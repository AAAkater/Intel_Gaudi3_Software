#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

struct NativeScalCommand
{
    std::vector<uint8_t> bytes;
    bool submittedDuringCapture = false;
    bool completesChunk = false;
    bool userCompletion = false;
};

struct NativeScalCommandBuffer
{
    std::vector<NativeScalCommand> commands;
    uint64_t completionIncrementsBeforeCommands = 0;
    bool empty() const { return commands.empty(); }
    void clear() { commands.clear(); completionIncrementsBeforeCommands = 0; }
};

struct NativeComputePatch
{
    uint32_t byteOffset = 0;
    uint32_t dependency = 0;
    uint32_t mask = 0;
    uint8_t sourceShift = 0;
    uint8_t destinationShift = 0;
};

struct NativeComputeWaitPacket
{
    std::vector<uint8_t> bytes;
    std::vector<NativeComputePatch> patches;
};

struct NativeComputePage
{
    std::vector<uint8_t> bytes;
    std::vector<NativeComputePatch> patches;
    uint64_t completionOffset = 0;
    uint64_t sourcePacketCount = 0;
    bool endsCompletion = false;
};

// Cold compilation of scheduler packets, completion offsets and NIC waits.
// Runtime publishes pages, never revisits recipe/segment descriptions.
struct NativeComputeProgram
{
    enum class ContinuationInput { Invalid, Ready, NeedsFence };

    static uint64_t continuationProducerOffset(uint64_t fullOffset, uint64_t prefixDelta)
    {
        return fullOffset > prefixDelta ? fullOffset - prefixDelta : 0;
    }

    static bool producerAvailableToPrefix(uint32_t producerSegment, uint32_t prefixComputes)
    {
        return producerSegment == UINT32_MAX || producerSegment < prefixComputes;
    }

    static ContinuationInput continuationInput(uint32_t expectedIndex, uint64_t expectedTarget,
                                               uint32_t currentIndex, uint64_t currentTarget,
                                               bool pendingWait)
    {
        if (!expectedTarget || currentIndex != expectedIndex || currentTarget < expectedTarget ||
            currentTarget >= (uint64_t(1) << 60))
            return ContinuationInput::Invalid;
        // Completed Bridge events need no wait. SCAL may also have emitted a
        // completion barrier while queuing waits at a CCB chunk boundary.
        return pendingWait ? ContinuationInput::NeedsFence : ContinuationInput::Ready;
    }

    std::vector<NativeComputePage> pages;
    std::vector<uint64_t> producerOffsets;
    uint64_t completionDelta = 0;
    uint64_t byteCount = 0;
    uint64_t maximumPendingBytes = 0;
    uint64_t sourcePacketCount = 0;
    bool externalInputCompletion = false;
    std::vector<size_t> segmentPageEnds;
    std::vector<size_t> segmentPageByteEnds;
    std::vector<uint64_t> segmentPagePacketEnds;
    std::vector<uint64_t> segmentCompletionOffsets;

    bool canReplayAt(uint64_t base) const
    {
        // Leave room for the optional input completion and the terminal CCB
        // retirement guard. Neither target may overflow the 60-bit long SO.
        const uint64_t margin = externalInputCompletion ? 2 : 1;
        return completionDelta < (uint64_t(1) << 60) - margin &&
               base < (uint64_t(1) << 60) - completionDelta - margin;
    }

    size_t lastWaitDependency() const
    {
        for (auto page = pages.rbegin(); page != pages.rend(); ++page)
            if (!page->patches.empty()) return page->patches.back().dependency;
        return std::numeric_limits<size_t>::max();
    }

    static NativeComputeProgram prepare(
        const std::vector<const NativeScalCommandBuffer*>& segments,
        const std::vector<uint32_t>& consumers,
        const std::vector<NativeComputeWaitPacket>& waitPackets,
        uint32_t alignment, uint64_t availableBytes, bool deferredTerminalRetirement = false,
        bool externalInputCompletion = false, const std::vector<uint32_t>& producers = {})
    {
        if (segments.empty() || consumers.empty() || consumers.size() > 128 || waitPackets.empty() ||
            alignment == 0 || (alignment & (alignment - 1)) != 0)
            throw std::invalid_argument("Invalid native compute plan shape");
        if (!producers.empty() && producers.size() != consumers.size())
            throw std::invalid_argument("Native producer/consumer coverage differs");
        for (size_t i = 0; i < consumers.size(); ++i)
        {
            if ((consumers[i] == 0 && !externalInputCompletion) || consumers[i] >= segments.size() ||
                (i && consumers[i] < consumers[i - 1]))
                throw std::invalid_argument("Native NIC consumers must follow ordered producers");
            if (!producers.empty() &&
                (producers[i] == UINT32_MAX ? !externalInputCompletion : producers[i] >= consumers[i]))
                throw std::invalid_argument("Native producer must finish before its consumer");
        }
        NativeComputeProgram plan;
        plan.externalInputCompletion = externalInputCompletion &&
            (producers.empty() ? consumers.front() == 0 :
             std::find(producers.begin(), producers.end(), UINT32_MAX) != producers.end());
        auto append = [&](const std::vector<uint8_t>& bytes, const std::vector<NativeComputePatch>& patches,
                          uint32_t dependency, bool completes) {
            if (bytes.empty() || bytes.size() > alignment)
                throw std::invalid_argument("Native packet exceeds command alignment");
            if (plan.pages.empty() || plan.pages.back().endsCompletion ||
                plan.pages.back().bytes.size() + bytes.size() > alignment)
                plan.pages.emplace_back();
            auto& page = plan.pages.back();
            const size_t start = page.bytes.size();
            page.bytes.insert(page.bytes.end(), bytes.begin(), bytes.end());
            for (auto patch : patches)
            {
                if (patch.byteOffset > bytes.size() || bytes.size() - patch.byteOffset < sizeof(uint32_t) ||
                    !patch.mask || patch.sourceShift > 45 || patch.destinationShift > 17)
                    throw std::invalid_argument("Invalid native compute relocation");
                patch.byteOffset += start;
                patch.dependency = dependency;
                page.patches.push_back(patch);
            }
            if (completes)
            {
                page.endsCompletion = true;
                page.completionOffset = ++plan.completionDelta;
            }
            plan.byteCount += bytes.size();
            ++page.sourcePacketCount;
            ++plan.sourcePacketCount;
        };
        size_t nextConsumer = 0;
        std::vector<uint64_t> segmentEnds(segments.size());
        for (size_t segment = 0; segment < segments.size(); ++segment)
        {
            const auto* source = segments[segment];
            if (!source || source->empty()) throw std::invalid_argument("Empty native compute segment");
            // Independent exchanges may share a compiled producer/consumer.
            // Retain a separate wait and completion relocation for every NIC
            // operation so the consumer cannot run after only the first one.
            while (nextConsumer < consumers.size() && segment == consumers[nextConsumer])
            {
                if (!plan.completionDelta && !plan.externalInputCompletion)
                    throw std::invalid_argument("NIC has no compute producer");
                const uint64_t producer = producers.empty() ? plan.completionDelta :
                    (producers[nextConsumer] == UINT32_MAX ? 0 : segmentEnds.at(producers[nextConsumer]));
                if (!producer && !plan.externalInputCompletion)
                    throw std::invalid_argument("Native producer has no completion target");
                plan.producerOffsets.push_back(producer);
                for (const auto& packet : waitPackets) append(packet.bytes, packet.patches, nextConsumer, false);
                ++nextConsumer;
            }
            if (source->completionIncrementsBeforeCommands >= (uint64_t(1) << 60) - plan.completionDelta)
                throw std::overflow_error("Native compute completion overflow");
            plan.completionDelta += source->completionIncrementsBeforeCommands;
            for (const auto& command : source->commands) append(command.bytes, {}, 0, command.completesChunk);
            segmentEnds[segment] = plan.completionDelta;
            plan.segmentPageEnds.push_back(plan.pages.size());
            plan.segmentPageByteEnds.push_back(plan.pages.back().bytes.size());
            plan.segmentPagePacketEnds.push_back(plan.pages.back().sourcePacketCount);
            plan.segmentCompletionOffsets.push_back(plan.completionDelta);
        }
        if (nextConsumer != consumers.size() || (!plan.pages.back().endsCompletion && !deferredTerminalRetirement) ||
            !plan.completionDelta || plan.completionDelta >= (uint64_t(1) << 60) - 1)
            throw std::invalid_argument("Native compute plan has no final completion");
        // A validated terminal scheduler fence release carries no HBM
        // references. Protect its CCB bytes through the next actual compute
        // completion. This is a retirement guard, not a new expected device
        // completion or an increment of the graph's mathematical work.
        uint64_t nextCompletion = plan.completionDelta + 1, pending = 0;
        for (auto it = plan.pages.rbegin(); it != plan.pages.rend(); ++it)
        {
            if (it->endsCompletion) { nextCompletion = it->completionOffset; pending = 0; }
            it->completionOffset = nextCompletion;
            pending += alignment;  // worst-case alignment per publication page
            plan.maximumPendingBytes = std::max(plan.maximumPendingBytes, pending + alignment);
        }
        // Every NIC microstream is published before compute. Every compute
        // page is also published immediately, so recycling can only wait for
        // an already-published prefix, provided a single job fits the CCB.
        if (plan.maximumPendingBytes > availableBytes)
            throw std::invalid_argument("Native compute job exceeds safe CCB publication capacity");
        return plan;
    }

    std::pair<NativeComputeProgram, NativeComputeProgram> split(
        size_t prefixPageCount, size_t prefixPageBytes, uint64_t prefixPagePackets,
        uint64_t prefixCompletionDelta,
        size_t collectiveSplit, size_t& suffixDependencyBegin) const
    {
        if (!prefixPageCount || prefixPageCount >= pages.size() || !prefixCompletionDelta ||
            prefixCompletionDelta >= completionDelta || !collectiveSplit ||
            collectiveSplit >= producerOffsets.size() || !prefixPageBytes ||
            prefixPageBytes > pages[prefixPageCount - 1].bytes.size() ||
            prefixPagePackets > pages[prefixPageCount - 1].sourcePacketCount)
            throw std::invalid_argument("Invalid native compute publication split");

        NativeComputeProgram prefix, suffix;
        prefix.pages.assign(pages.begin(), pages.begin() + prefixPageCount);
        suffix.pages.assign(pages.begin() + prefixPageCount, pages.end());
        const auto& shared = pages[prefixPageCount - 1];
        if (prefixPageBytes < shared.bytes.size())
        {
            NativeComputePage suffixHead = shared;
            std::vector<NativeComputePatch> prefixPatches, suffixPatches;
            for (auto patch : shared.patches)
            {
                if (patch.byteOffset + sizeof(uint32_t) <= prefixPageBytes)
                    prefixPatches.push_back(patch);
                else if (patch.byteOffset >= prefixPageBytes)
                {
                    patch.byteOffset -= prefixPageBytes;
                    suffixPatches.push_back(patch);
                }
                else
                    throw std::invalid_argument("Native relocation crosses the publication split");
            }
            auto& prefixTail = prefix.pages.back();
            prefixTail.bytes.resize(prefixPageBytes);
            prefixTail.patches = std::move(prefixPatches);
            prefixTail.sourcePacketCount = prefixPagePackets;
            prefixTail.endsCompletion = false;
            suffixHead.bytes.erase(suffixHead.bytes.begin(), suffixHead.bytes.begin() + prefixPageBytes);
            suffixHead.patches = std::move(suffixPatches);
            suffixHead.sourcePacketCount -= prefixPagePackets;
            if (suffixHead.bytes.empty() || !suffixHead.sourcePacketCount)
                throw std::invalid_argument("Native suffix has an empty split page");
            suffix.pages.insert(suffix.pages.begin(), std::move(suffixHead));
        }
        const auto& prefixTail = prefix.pages.back();
        if ((!prefixTail.endsCompletion && prefixTail.completionOffset != prefixCompletionDelta + 1) ||
            (prefixTail.endsCompletion && prefixTail.completionOffset != prefixCompletionDelta))
            throw std::invalid_argument("Native prefix tail has no bounded retirement completion");
        prefix.producerOffsets.assign(producerOffsets.begin(), producerOffsets.begin() + collectiveSplit);
        prefix.completionDelta = prefixCompletionDelta;
        prefix.externalInputCompletion = externalInputCompletion;

        suffixDependencyBegin = producerOffsets.size();
        for (const auto& page : prefix.pages)
            for (const auto& patch : page.patches)
                if (patch.dependency >= collectiveSplit)
                    throw std::invalid_argument("Native prefix waits on a suffix collective");
        for (const auto& page : suffix.pages)
            for (const auto& patch : page.patches)
                suffixDependencyBegin = std::min(suffixDependencyBegin, size_t(patch.dependency));
        if (suffixDependencyBegin > collectiveSplit || suffixDependencyBegin >= producerOffsets.size())
            throw std::invalid_argument("Native suffix lacks its boundary collective dependency");
        suffix.producerOffsets.assign(producerOffsets.begin() + suffixDependencyBegin, producerOffsets.end());
        suffix.completionDelta = completionDelta - prefixCompletionDelta;
        suffix.externalInputCompletion = false;
        for (auto& page : suffix.pages)
        {
            if (page.completionOffset <= prefixCompletionDelta)
                throw std::invalid_argument("Native suffix page retires before its split");
            page.completionOffset -= prefixCompletionDelta;
            for (auto& patch : page.patches)
            {
                if (patch.dependency < suffixDependencyBegin)
                    throw std::invalid_argument("Native suffix dependency precedes retained boundary");
                patch.dependency -= suffixDependencyBegin;
            }
        }
        auto summarize = [](NativeComputeProgram& program) {
            program.byteCount = 0;
            program.sourcePacketCount = 0;
            uint64_t nextCompletion = program.completionDelta + 1, pending = 0;
            for (auto& page : program.pages)
            {
                program.byteCount += page.bytes.size();
                program.sourcePacketCount += page.sourcePacketCount;
            }
            for (auto it = program.pages.rbegin(); it != program.pages.rend(); ++it)
            {
                if (it->endsCompletion) { nextCompletion = it->completionOffset; pending = 0; }
                it->completionOffset = nextCompletion;
                pending += it->bytes.size();
                program.maximumPendingBytes = std::max(program.maximumPendingBytes, pending + it->bytes.size());
            }
        };
        summarize(prefix);
        summarize(suffix);
        return {std::move(prefix), std::move(suffix)};
    }

    template<class Sync>
    static void writePage(uint8_t* output, const NativeComputePage& page, const Sync* completions)
    {
        std::memcpy(output, page.bytes.data(), page.bytes.size());
        for (const auto& patch : page.patches)
        {
            uint32_t word;
            std::memcpy(&word, output + patch.byteOffset, sizeof(word));
            const uint64_t value = completions[patch.dependency].targetValue >> patch.sourceShift;
            word = (word & ~patch.mask) | ((value << patch.destinationShift) & patch.mask);
            std::memcpy(output + patch.byteOffset, &word, sizeof(word));
        }
    }
};
