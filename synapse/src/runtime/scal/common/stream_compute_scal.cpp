#include "native_bounded_selection.h"
#include "native_program_window.h"
#include "stream_compute_scal.hpp"
#include "synapse_api.h"

#include "habana_global_conf_runtime.h"
#include "runtime/common/recipe/recipe_handle_impl.hpp"
#include "scal_event.hpp"
#include "synapse_common_types.h"
#include "runtime/common/recipe/recipe_dynamic_info.hpp"
#include "runtime/common/recipe/recipe_utils.hpp"
#include "defenders.h"
#include "device/device_mem_alloc.hpp"

#include "log_manager.h"

#include "runtime/scal/common/entities/scal_completion_group.hpp"
#include "runtime/scal/common/entities/scal_memory_pool.hpp"
#include "runtime/scal/common/native_compute_graph.hpp"
#include "runtime/scal/common/native_compute_graph_handle.hpp"
#include "runtime/common/streams/stream.hpp"
#include "runtime/scal/common/entities/scal_stream_compute_interface.hpp"
#include "global_statistics.hpp"

#include "profiler_api.hpp"

#include <cstring>
#include <limits>

NativeRecipeMemory::NativeRecipeMemory(ScalMemoryPool& globalPool, ScalMemoryPool& hbmSharedPool)
: m_globalPool(globalPool), m_hbmSharedPool(hbmSharedPool)
{
}

NativeRecipeMemory::~NativeRecipeMemory()
{
    if (m_initialized && !m_closed)
    {
        const synStatus status = close();
        if (status != synSuccess)
        {
            LOG_ERR(SYN_MEM_ALLOC, "Failed to release native recipe memory during destruction, status {}", status);
        }
    }
}

synStatus NativeRecipeMemory::initialize(uint64_t globalSize, uint64_t arcSize, uint64_t workspaceSize)
{
    if (m_initialized)
    {
        return synObjectAlreadyInitialized;
    }
    if (globalSize == 0 || arcSize == 0)
    {
        LOG_ERR(SYN_MEM_ALLOC,
                "Native recipe memory requires non-zero global and ARC sizes, got {}/{}",
                globalSize,
                arcSize);
        return synInvalidArgument;
    }

    // RecipeStaticProcessor places PROGRAM_DATA at offset zero and requires
    // that base to be 8 KiB aligned. The normal HbmGlblMemMgr aligns its
    // arena before handing out recipe chunks; direct SCAL pool allocations do
    // not inherit that contract, especially when the Bridge has first
    // reserved an arbitrary percentage of global HBM.
    constexpr uint64_t globalAlignment = 1ULL << 13;
    // Gaudi2 scheduler firmware encodes both static and dynamic ECB list
    // addresses in 256-byte chunks. ArcHbmMemMgr's generic minimum is 128,
    // but a standalone native allocation must preserve the stronger recipe
    // arena base contract or a zero-offset dynamic ECB list is rejected.
    constexpr uint64_t arcAlignment    = 256;
    if (!NativeProgramWindow::allocationSize(globalSize, false, m_globalAllocationBytes) ||
        arcSize > std::numeric_limits<uint64_t>::max() - (arcAlignment - 1) ||
        workspaceSize > std::numeric_limits<uint64_t>::max() - (globalAlignment - 1))
    {
        return synInvalidArgument;
    }
    m_arcAllocationBytes    = arcSize + arcAlignment - 1;

    synStatus status = m_globalPool.allocateDeviceMemory(m_globalAllocationBytes, m_globalBuffer);
    if (status != synSuccess)
    {
        return status;
    }

    uint32_t unusedGlobalCoreAddress = 0;
    uint64_t rawGlobalDeviceAddress = 0;
    status = m_globalPool.getDeviceMemoryAddress(
        m_globalBuffer, unusedGlobalCoreAddress, rawGlobalDeviceAddress);
    if (status != synSuccess)
    {
        m_globalPool.releaseDeviceMemory(m_globalBuffer);
        m_globalBuffer = nullptr;
        return status;
    }

    if (!NativeProgramWindow::selectAddress(
            rawGlobalDeviceAddress, m_globalAllocationBytes, globalSize, m_globalDeviceAddress))
    {
        // Native plans own program storage outside DeviceScal's normal
        // compute-stream arena. Preserve that arena's 4 GiB TPC-PC window
        // contract as well as PROGRAM_DATA alignment. Do not assume that
        // the replacement allocation will have the same raw address.
        status = m_globalPool.releaseDeviceMemory(m_globalBuffer);
        if (status != synSuccess) return status;
        m_globalBuffer = nullptr;
        NativeProgramWindow::allocationSize(globalSize, true, m_globalAllocationBytes);
        status = m_globalPool.allocateDeviceMemory(m_globalAllocationBytes, m_globalBuffer);
        if (status != synSuccess) return status;
        status = m_globalPool.getDeviceMemoryAddress(
            m_globalBuffer, unusedGlobalCoreAddress, rawGlobalDeviceAddress);
        if (status == synSuccess && !NativeProgramWindow::selectAddress(
                rawGlobalDeviceAddress, m_globalAllocationBytes, globalSize, m_globalDeviceAddress))
        {
            status = synInvalidArgument;
        }
        if (status != synSuccess)
        {
            m_globalPool.releaseDeviceMemory(m_globalBuffer);
            m_globalBuffer = nullptr;
            return status;
        }
        LOG_INFO(SYN_MEM_ALLOC,
                 "Native recipe program window padded: raw 0x{:x}, selected 0x{:x}, payload {}, allocation {}",
                 rawGlobalDeviceAddress, m_globalDeviceAddress, globalSize, m_globalAllocationBytes);
    }

    status = m_hbmSharedPool.allocateDeviceMemory(m_arcAllocationBytes, m_arcBuffer);
    if (status != synSuccess)
    {
        m_globalPool.releaseDeviceMemory(m_globalBuffer);
        m_globalBuffer = nullptr;
        return status;
    }

    uint32_t rawArcCoreAddress = 0;
    uint64_t rawArcDeviceAddress = 0;
    status = m_hbmSharedPool.getDeviceMemoryAddress(m_arcBuffer, rawArcCoreAddress, rawArcDeviceAddress);
    if (status != synSuccess)
    {
        m_hbmSharedPool.releaseDeviceMemory(m_arcBuffer);
        m_globalPool.releaseDeviceMemory(m_globalBuffer);
        m_arcBuffer    = nullptr;
        m_globalBuffer = nullptr;
        return status;
    }

    const uint64_t arcShift = (arcAlignment - (rawArcDeviceAddress & (arcAlignment - 1))) & (arcAlignment - 1);
    if ((static_cast<uint64_t>(rawArcCoreAddress) + arcShift) > std::numeric_limits<uint32_t>::max() ||
        ((static_cast<uint64_t>(rawArcCoreAddress) + arcShift) & (arcAlignment - 1)) != 0)
    {
        m_hbmSharedPool.releaseDeviceMemory(m_arcBuffer);
        m_globalPool.releaseDeviceMemory(m_globalBuffer);
        m_arcBuffer    = nullptr;
        m_globalBuffer = nullptr;
        return synFail;
    }
    m_arcDeviceAddress = rawArcDeviceAddress + arcShift;
    m_arcCoreAddress   = static_cast<uint32_t>(static_cast<uint64_t>(rawArcCoreAddress) + arcShift);

    // Bridge's shared workspace can grow or move during later prefill or
    // output-head launches. Captured descriptor addresses must instead point
    // to graph-owned storage, retained through the final graph completion.
    if (workspaceSize)
    {
        m_workspaceAllocationBytes = workspaceSize + globalAlignment - 1;
        status = m_globalPool.allocateDeviceMemory(m_workspaceAllocationBytes, m_workspaceBuffer);
        uint64_t rawWorkspaceAddress = 0;
        uint32_t unusedCoreAddress = 0;
        if (status == synSuccess)
            status = m_globalPool.getDeviceMemoryAddress(m_workspaceBuffer, unusedCoreAddress, rawWorkspaceAddress);
        if (status != synSuccess)
        {
            if (m_workspaceBuffer) m_globalPool.releaseDeviceMemory(m_workspaceBuffer);
            m_hbmSharedPool.releaseDeviceMemory(m_arcBuffer);
            m_globalPool.releaseDeviceMemory(m_globalBuffer);
            m_workspaceBuffer = m_arcBuffer = m_globalBuffer = nullptr;
            return status;
        }
        m_workspaceDeviceAddress = (rawWorkspaceAddress + globalAlignment - 1) & ~(globalAlignment - 1);
    }

    LOG_INFO(SYN_MEM_ALLOC,
             "Native recipe storage global raw/aligned 0x{:x}/0x{:x} bytes {}/{}; "
             "ARC raw/aligned 0x{:x}/0x{:x} core 0x{:x}/0x{:x} bytes {}/{}",
             rawGlobalDeviceAddress,
             m_globalDeviceAddress,
             globalSize,
             m_globalAllocationBytes,
             rawArcDeviceAddress,
             m_arcDeviceAddress,
             rawArcCoreAddress,
             m_arcCoreAddress,
             arcSize,
             m_arcAllocationBytes);
    m_initialized = true;
    return synSuccess;
}

synStatus NativeRecipeMemory::close()
{
    if (!m_initialized)
    {
        return synObjectNotInitialized;
    }
    if (m_closed)
    {
        return synSuccess;
    }

    synStatus firstFailure = synSuccess;
    if (m_workspaceBuffer != nullptr)
    {
        const synStatus status = m_globalPool.releaseDeviceMemory(m_workspaceBuffer);
        if (status != synSuccess) firstFailure = status;
        else m_workspaceBuffer = nullptr;
    }
    if (m_arcBuffer != nullptr)
    {
        const synStatus status = m_hbmSharedPool.releaseDeviceMemory(m_arcBuffer);
        if (status != synSuccess)
        {
            if (firstFailure == synSuccess) firstFailure = status;
        }
        else
        {
            m_arcBuffer = nullptr;
        }
    }
    if (m_globalBuffer != nullptr)
    {
        const synStatus status = m_globalPool.releaseDeviceMemory(m_globalBuffer);
        if (firstFailure == synSuccess && status != synSuccess)
        {
            firstFailure = status;
        }
        if (status == synSuccess)
        {
            m_globalBuffer = nullptr;
        }
    }

    if (firstFailure == synSuccess)
    {
        m_closed = true;
    }
    return firstFailure;
}

NativeComputeGraph::NativeComputeGraph(ScalStreamComputeInterface& stream,
                                       ScalMemoryPool&              globalPool,
                                       ScalMemoryPool&              hbmSharedPool)
: m_stream(stream),
  m_streamBase(dynamic_cast<ScalStreamBase&>(stream)),
  m_globalPool(globalPool),
  m_hbmSharedPool(hbmSharedPool)
{
}

NativeComputeGraph::~NativeComputeGraph()
{
    if (m_state != State::CLOSED)
    {
        const synStatus status = close();
        if (status != synSuccess)
        {
            LOG_ERR(SYN_STREAM, "Failed to close native compute graph during destruction, status {}", status);
        }
    }
}

synStatus NativeComputeGraph::beginCapture()
{
    if (m_state != State::CREATED)
    {
        return synInvalidArgument;
    }
    m_state = State::CAPTURING;
    return synSuccess;
}

CompiledSegment* NativeComputeGraph::addPendingSegment(
    uint64_t                      globalSize,
    uint64_t                      arcSize,
    uint64_t                      recipeSequence,
    const synLaunchTensorInfoExt* launchTensors,
    uint32_t                      launchTensorCount,
    uint64_t                      workspaceAddress,
    uint64_t                      workspaceSize,
    synStatus&                    status)
{
    status = synSuccess;
    if (m_state != State::CAPTURING || launchTensors == nullptr || launchTensorCount == 0)
    {
        status = synInvalidArgument;
        return nullptr;
    }

    CompiledSegment segment;
    segment.memory = std::make_unique<NativeRecipeMemory>(m_globalPool, m_hbmSharedPool);
    // Bound new native scratch residency independently of the model's KV pool.
    constexpr uint64_t maximumWorkspaceBytes = uint64_t(2) << 30;
    if (m_workspaceAllocationBytes > maximumWorkspaceBytes || workspaceSize > maximumWorkspaceBytes ||
        (workspaceSize && workspaceSize + 8191 > maximumWorkspaceBytes - m_workspaceAllocationBytes))
    {
        status = synOutOfDeviceMemory;
        m_state = State::INVALID;
        return nullptr;
    }
    status         = segment.memory->initialize(globalSize, arcSize, workspaceSize);
    if (status != synSuccess)
    {
        m_state = State::INVALID;
        return nullptr;
    }

    segment.bindings.reserve(launchTensorCount);
    for (uint32_t index = 0; index < launchTensorCount; ++index)
    {
        NativeTensorBinding binding;
        binding.tensorId      = launchTensors[index].tensorId;
        binding.deviceAddress = launchTensors[index].pTensorAddress;
        binding.tensorType    = launchTensors[index].tensorType;
        std::memcpy(binding.sizes, launchTensors[index].tensorSize, sizeof(binding.sizes));
        segment.bindings.push_back(binding);
    }
    segment.workspaceAddress = workspaceSize ? segment.memory->workspaceDeviceAddress() : workspaceAddress;
    segment.recipeSequence   = recipeSequence;
    m_globalProgramBytes += segment.memory->globalAllocationBytes();
    m_arcProgramBytes += segment.memory->arcAllocationBytes();
    m_workspaceAllocationBytes += segment.memory->workspaceAllocationBytes();
    m_segments.push_back(std::move(segment));
    return &m_segments.back();
}

void NativeComputeGraph::commitSegment(CompiledSegment& segment, NativeScalCommandBuffer&& commands)
{
    if (m_state != State::CAPTURING || commands.empty())
    {
        m_state = State::INVALID;
        return;
    }
    segment.commands = std::move(commands);
    segment.ready    = true;
}

void NativeComputeGraph::invalidate()
{
    m_state = State::INVALID;
}

synStatus NativeComputeGraph::endCapture()
{
    if (m_state != State::CAPTURING || m_segments.empty())
    {
        return synInvalidArgument;
    }
    for (const CompiledSegment& segment : m_segments)
    {
        if (!segment.ready || segment.commands.empty())
        {
            m_state = State::INVALID;
            return synFail;
        }
    }
    m_state = State::INSTANTIATED;
    return synSuccess;
}

synStatus NativeComputeGraph::preparePlan(const uint32_t* consumers, uint64_t count, uint32_t nicIndex,
                                         synNativeGraphExchangeReplay callback, void* context, const uint32_t* producers)
{
    if (m_state != State::INSTANTIATED || m_replayActive || m_jointPlan || !consumers || !callback || !context ||
        !count || count > 128 || nicIndex % 8 || (nicIndex >> 3) > 0xff)
    {
        std::fprintf(stderr, "NATIVE_PLAN_REJECT state=%u active=%u existing=%u count=%llu nic_index=%u callback=%u context=%u\n",
            unsigned(m_state), unsigned(m_replayActive), unsigned(bool(m_jointPlan)),
            static_cast<unsigned long long>(count), nicIndex, unsigned(callback != nullptr), unsigned(context != nullptr));
        return synInvalidArgument;
    }
    try
    {
        auto plan = std::make_unique<JointPlan>();
        std::vector<const NativeScalCommandBuffer*> buffers;
        for (const auto& segment : m_segments) buffers.push_back(&segment.commands);
        const bool terminalRelease = m_streamBase.nativeTerminalRelease(m_segments.back().commands);
        plan->program = NativeComputeProgram::prepare(buffers, {consumers, consumers + count},
            m_streamBase.prepareNativeWaitPackets(nicIndex), m_streamBase.nativeCommandAlignment(),
            m_streamBase.nativeAvailableBytes(), terminalRelease,
            producers ? std::find(producers, producers + count, UINT32_MAX) != producers + count : consumers[0] == 0,
            producers ? std::vector<uint32_t>(producers, producers + count) : std::vector<uint32_t>{});
        plan->nicLongSoIndex = nicIndex;
        plan->callback = callback;
        plan->context = context;
        plan->producers.resize(count);
        plan->completions.resize(count);
        plan->submissionScratch.resize(2);
        plan->statistics[1] = plan->program.pages.size();
        plan->statistics[2] = plan->program.sourcePacketCount;
        plan->statistics[3] = plan->program.byteCount;
        plan->statistics[4] = plan->program.completionDelta;
        plan->statistics[5] = plan->program.maximumPendingBytes;
        plan->statistics[8] = count;
        m_jointPlan = std::move(plan);
        return synSuccess;
    }
    catch (const std::exception& error)
    {
        LOG_ERR(SYN_STREAM, "Native compute plan prepare rejected: {}", error.what());
        std::fprintf(stderr, "NATIVE_PLAN_REJECT reason=%s\n", error.what());
        return synInvalidArgument;
    }
}

synStatus NativeComputeGraph::prepareSegmentedPlan(
    const uint32_t* producers, const uint32_t* consumers, uint64_t count,
    uint64_t prefixCollectives, uint32_t prefixComputes, uint32_t nicIndex,
    synNativeGraphExchangeReplay callback, void* context)
{
    if (m_state != State::INSTANTIATED || m_replayActive || m_jointPlan || !producers || !consumers ||
        !callback || !context || count < 2 || count > 128 || !prefixCollectives ||
        prefixCollectives >= count || nicIndex % 8 || (nicIndex >> 3) > 0xff)
        return synInvalidArgument;
    try
    {
        auto plan = std::make_unique<JointPlan>();
        std::vector<const NativeScalCommandBuffer*> buffers;
        for (const auto& segment : m_segments) buffers.push_back(&segment.commands);
        const bool terminalRelease = m_streamBase.nativeTerminalRelease(m_segments.back().commands);
        plan->program = NativeComputeProgram::prepare(
            buffers, {consumers, consumers + count}, m_streamBase.prepareNativeWaitPackets(nicIndex),
            m_streamBase.nativeCommandAlignment(), m_streamBase.nativeAvailableBytes(), terminalRelease,
            std::find(producers, producers + count, UINT32_MAX) != producers + count,
            std::vector<uint32_t>(producers, producers + count));
        const size_t splitSegment = prefixComputes;
        if (!splitSegment || splitSegment >= plan->program.segmentPageEnds.size())
            return synInvalidArgument;
        for (size_t n = 0; n < count; ++n)
            // UINT32_MAX is a stream-ordered device producer outside this
            // captured graph. Prefix replay binds it to the current compute
            // completion through externalInputCompletion.
            if ((n < prefixCollectives &&
                 !NativeComputeProgram::producerAvailableToPrefix(producers[n], splitSegment)) ||
                (n >= prefixCollectives && consumers[n] < splitSegment))
                return synInvalidArgument;
        const size_t prefixPages = plan->program.segmentPageEnds[splitSegment - 1];
        const uint64_t prefixDelta = plan->program.segmentCompletionOffsets[splitSegment - 1];
        auto parts = plan->program.split(prefixPages,
                                         plan->program.segmentPageByteEnds[splitSegment - 1],
                                         plan->program.segmentPagePacketEnds[splitSegment - 1],
                                         prefixDelta, prefixCollectives,
                                         plan->suffixDependencyBegin);
        plan->prefixProgram = std::move(parts.first);
        plan->suffixProgram = std::move(parts.second);
        plan->collectiveSplit = prefixCollectives;
        plan->nicLongSoIndex = nicIndex;
        plan->callback = callback;
        plan->context = context;
        plan->producers.resize(count);
        plan->completions.resize(count);
        plan->submissionScratch.resize(2);
        plan->statistics[1] = plan->program.pages.size();
        plan->statistics[2] = plan->program.sourcePacketCount;
        plan->statistics[3] = plan->program.byteCount;
        plan->statistics[4] = plan->program.completionDelta;
        plan->statistics[5] = plan->program.maximumPendingBytes;
        plan->statistics[8] = count;
        std::fprintf(stderr,
            "NATIVE_SEGMENTED_PLAN_READY split=%llu prefix_pages=%zu suffix_pages=%zu "
            "prefix_delta=%llu suffix_delta=%llu carry=%zu\n",
            static_cast<unsigned long long>(prefixCollectives), plan->prefixProgram.pages.size(),
            plan->suffixProgram.pages.size(),
            static_cast<unsigned long long>(plan->prefixProgram.completionDelta),
            static_cast<unsigned long long>(plan->suffixProgram.completionDelta),
            plan->suffixDependencyBegin);
        m_jointPlan = std::move(plan);
        return synSuccess;
    }
    catch (const std::exception& error)
    {
        LOG_ERR(SYN_STREAM, "Segmented native compute plan prepare rejected: {}", error.what());
        std::fprintf(stderr, "NATIVE_SEGMENTED_PLAN_REJECT reason=%s\n", error.what());
        return synInvalidArgument;
    }
}

synStatus NativeComputeGraph::prepareBoundedVariants(
    const uint32_t* minimum, uint64_t segments, const uint32_t* producers,
    const uint32_t* consumers, uint64_t collectives, uint32_t limit)
{
    if (m_state != State::INSTANTIATED || m_replayActive || m_replayCount || !m_jointPlan ||
        !minimum || !producers || !consumers || segments != m_segments.size() ||
        collectives != m_jointPlan->producers.size() || !limit || limit > 8 ||
        !m_jointPlan->boundedPrograms.empty() || m_jointPlan->collectiveSplit)
        return synInvalidArgument;
    try {
        std::vector<NativeComputeProgram> variants;
        for (uint32_t bound = 0; bound <= limit; ++bound) {
            const auto selection = NativeBoundedSelection::prepare(
                {minimum, minimum + segments}, {producers, producers + collectives},
                {consumers, consumers + collectives}, bound, limit);
            std::vector<const NativeScalCommandBuffer*> buffers;
            for (auto index : selection.segments) buffers.push_back(&m_segments[index].commands);
            const bool terminalRelease = m_streamBase.nativeTerminalRelease(*buffers.back());
            variants.push_back(NativeComputeProgram::prepare(
                buffers, selection.consumers, m_streamBase.prepareNativeWaitPackets(m_jointPlan->nicLongSoIndex),
                m_streamBase.nativeCommandAlignment(), m_streamBase.nativeAvailableBytes(), terminalRelease,
                std::find(producers, producers + collectives, UINT32_MAX) != producers + collectives,
                selection.producers));
        }
        m_jointPlan->boundedPrograms = std::move(variants);
        return synSuccess;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "NATIVE_BOUNDED_PLAN_REJECT reason=%s\n", error.what());
        return synInvalidArgument;
    }
}

synStatus NativeComputeGraph::selectBoundedVariant(uint32_t bound)
{
    if (m_state != State::INSTANTIATED || m_replayActive || !m_jointPlan ||
        bound >= m_jointPlan->boundedPrograms.size() || m_jointPlan->selectedBound != UINT32_MAX)
        return synInvalidArgument;
    m_jointPlan->selectedBound = bound;
    return synSuccess;
}

synStatus NativeComputeGraph::replayPlan(synNativeGraphSyncInfo& completion, uint64_t* statistics, uint64_t count)
{
    if (m_state != State::INSTANTIATED || m_replayActive || !m_jointPlan || !statistics || count != 12)
        return synInvalidArgument;
    auto& plan = *m_jointPlan;
    if (!plan.boundedPrograms.empty() && plan.selectedBound >= plan.boundedPrograms.size())
        return synInvalidArgument;
    const auto& program = plan.boundedPrograms.empty() ? plan.program : plan.boundedPrograms[plan.selectedBound];
    plan.selectedBound = UINT32_MAX;
    plan.statistics[1] = program.pages.size();
    plan.statistics[2] = program.sourcePacketCount;
    plan.statistics[3] = program.byteCount;
    plan.statistics[4] = program.completionDelta;
    plan.statistics[5] = program.maximumPendingBytes;
    if (!program.canReplayAt(m_streamBase.getTargetLongSo(0).m_targetValue)) return synInvalidArgument;
    m_replayActive = true;
    auto fail = [&]() { m_state = State::INVALID; m_replayActive = false; return synFail; };
    try
    {
        if (program.externalInputCompletion && m_streamBase.prevCmdIsWait())
        {
            // A partial decoder graph can begin with NIC, using a projection
            // from its preceding group. A bare input wait does not advance the
            // compute SO; use the normal device barrier to publish readiness.
            // The queue lock is already held. No host synchronization occurs.
            ScalLongSyncObject inputReady;
            if (m_streamBase.addBarrierOrEmptyPdma(inputReady) != synSuccess) return fail();
            if (!plan.statistics[0]) std::fprintf(stderr,
                "NATIVE_INPUT_READY_FENCE target=%llu scope=graph_boundary_outside_page_submission_count\n",
                static_cast<unsigned long long>(inputReady.m_targetValue));
        }
        const ScalLongSyncObject base = m_streamBase.getTargetLongSo(0);
        if (!base.m_targetValue || !program.canReplayAt(base.m_targetValue)) return fail();
        // QueueComputeScal holds the compute queue lock across reservation,
        // NIC publication and compute publication. No other launch can consume
        // these offsets between the three operations.
        const ScalLongSyncObject final = m_streamBase.getIncrementedLongSo(true, program.completionDelta);
        for (size_t n = 0; n < plan.producers.size(); ++n)
            plan.producers[n] = {uint32_t(base.m_index), base.m_targetValue + program.producerOffsets[n]};
        ++plan.statistics[11];
        if (plan.callback(plan.context, plan.producers.data(), plan.producers.size(), plan.completions.data()) != 0)
            return fail();
        for (size_t n = 0; n < plan.completions.size(); ++n)
            if (plan.completions[n].longSoIndex != plan.nicLongSoIndex || !plan.completions[n].targetValue ||
                plan.completions[n].targetValue >= (uint64_t(1) << 60) ||
                (n && plan.completions[n].targetValue != plan.completions[n - 1].targetValue + 1)) return fail();
        uint64_t submissions = 0;
        if (m_streamBase.publishNativeComputeProgram(program, plan.completions.data(), base,
                                                     plan.submissionScratch, submissions) != synSuccess) return fail();
        if (!plan.statistics[0]) plan.statistics[9] = final.m_targetValue;
        plan.statistics[10] = final.m_targetValue;
        ++plan.statistics[0];
        plan.statistics[6] += submissions;
        plan.statistics[7] = submissions;
        m_lastCompletion = final;
        ++m_replayCount;
        m_replayActive = false;
        completion = {uint32_t(final.m_index), final.m_targetValue};
        std::copy(plan.statistics.begin(), plan.statistics.end(), statistics);
        return synSuccess;
    }
    catch (...) { return fail(); }
}

synStatus NativeComputeGraph::replayPlanPrefix(
    synNativeGraphSyncInfo& completion, uint64_t* statistics, uint64_t count)
{
    if (m_state != State::INSTANTIATED || m_replayActive || !m_jointPlan || !statistics || count != 12 ||
        m_jointPlan->collectiveSplit == 0 || m_jointPlan->prefixProgram.pages.empty())
        return synInvalidArgument;
    auto& plan = *m_jointPlan;
    if (!plan.prefixProgram.canReplayAt(m_streamBase.getTargetLongSo(0).m_targetValue))
        return synInvalidArgument;
    m_replayActive = true;
    auto fail = [&]() { m_state = State::INVALID; m_replayActive = false; return synFail; };
    try
    {
        if (plan.prefixProgram.externalInputCompletion && m_streamBase.prevCmdIsWait())
        {
            ScalLongSyncObject inputReady;
            if (m_streamBase.addBarrierOrEmptyPdma(inputReady) != synSuccess) return fail();
        }
        const ScalLongSyncObject base = m_streamBase.getTargetLongSo(0);
        if (!base.m_targetValue || !plan.prefixProgram.canReplayAt(base.m_targetValue)) return fail();
        const ScalLongSyncObject prefixFinal =
            m_streamBase.getIncrementedLongSo(true, plan.prefixProgram.completionDelta);
        for (size_t n = 0; n < plan.collectiveSplit; ++n)
            plan.producers[n] = {uint32_t(base.m_index), base.m_targetValue + plan.program.producerOffsets[n]};
        ++plan.statistics[11];
        if (plan.callback(plan.context, plan.producers.data(), plan.collectiveSplit,
                          plan.completions.data()) != 0)
            return fail();
        for (size_t n = 0; n < plan.collectiveSplit; ++n)
            if (plan.completions[n].longSoIndex != plan.nicLongSoIndex ||
                !plan.completions[n].targetValue || plan.completions[n].targetValue >= (uint64_t(1) << 60) ||
                (n && plan.completions[n].targetValue != plan.completions[n - 1].targetValue + 1))
                return fail();
        uint64_t submissions = 0;
        if (m_streamBase.publishNativeComputeProgram(plan.prefixProgram, plan.completions.data(), base,
                                                     plan.submissionScratch, submissions) != synSuccess)
            return fail();
        plan.statistics[6] += submissions;
        plan.statistics[7] = submissions;
        plan.segmentedBase = base;
        completion = {uint32_t(prefixFinal.m_index), prefixFinal.m_targetValue};
        std::copy(plan.statistics.begin(), plan.statistics.end(), statistics);
        return synSuccess;
    }
    catch (...) { return fail(); }
}

synStatus NativeComputeGraph::replayPlanFinish(
    synNativeGraphSyncInfo& completion, uint64_t* statistics, uint64_t count)
{
    if (m_state != State::INSTANTIATED || !m_replayActive || !m_jointPlan || !statistics || count != 12 ||
        m_jointPlan->collectiveSplit == 0 || m_jointPlan->suffixProgram.pages.empty() ||
        m_jointPlan->segmentedBase.m_targetValue == 0)
        return synInvalidArgument;
    auto& plan = *m_jointPlan;
    auto fail = [&]() { m_state = State::INVALID; m_replayActive = false; return synFail; };
    try
    {
        ScalLongSyncObject suffixBase = m_streamBase.getTargetLongSo(0);
        const auto input = NativeComputeProgram::continuationInput(
            uint32_t(plan.segmentedBase.m_index),
            plan.segmentedBase.m_targetValue + plan.prefixProgram.completionDelta,
            uint32_t(suffixBase.m_index), suffixBase.m_targetValue, m_streamBase.prevCmdIsWait());
        if (input == NativeComputeProgram::ContinuationInput::Invalid)
            return fail();
        // Turn an outstanding input wait into a producer SO for the first
        // suffix exchange. A retired Bridge event is already ready and may
        // legitimately leave no wait packet or extra compute completion.
        if (input == NativeComputeProgram::ContinuationInput::NeedsFence &&
            m_streamBase.addBarrierOrEmptyPdma(suffixBase) != synSuccess)
            return fail();
        if (!plan.suffixProgram.canReplayAt(suffixBase.m_targetValue))
            return fail();
        const ScalLongSyncObject final =
            m_streamBase.getIncrementedLongSo(true, plan.suffixProgram.completionDelta);
        for (size_t n = plan.collectiveSplit; n < plan.producers.size(); ++n)
            plan.producers[n] = {uint32_t(suffixBase.m_index),
                                 suffixBase.m_targetValue +
                                     NativeComputeProgram::continuationProducerOffset(
                                         plan.program.producerOffsets[n], plan.prefixProgram.completionDelta)};
        ++plan.statistics[11];
        if (plan.callback(plan.context, plan.producers.data() + plan.collectiveSplit,
                          plan.producers.size() - plan.collectiveSplit,
                          plan.completions.data() + plan.collectiveSplit) != 0)
            return fail();
        for (size_t n = plan.collectiveSplit; n < plan.completions.size(); ++n)
            if (plan.completions[n].longSoIndex != plan.nicLongSoIndex ||
                !plan.completions[n].targetValue || plan.completions[n].targetValue >= (uint64_t(1) << 60) ||
                plan.completions[n].targetValue != plan.completions[n - 1].targetValue + 1)
                return fail();
        uint64_t submissions = 0;
        if (m_streamBase.publishNativeComputeProgram(
                plan.suffixProgram, plan.completions.data() + plan.suffixDependencyBegin, suffixBase,
                plan.submissionScratch, submissions) != synSuccess)
            return fail();
        if (!plan.statistics[0]) plan.statistics[9] = final.m_targetValue;
        plan.statistics[10] = final.m_targetValue;
        ++plan.statistics[0];
        plan.statistics[6] += submissions;
        plan.statistics[7] = submissions;
        m_lastCompletion = final;
        ++m_replayCount;
        m_replayActive = false;
        plan.segmentedBase = LongSoEmpty;
        completion = {uint32_t(final.m_index), final.m_targetValue};
        std::copy(plan.statistics.begin(), plan.statistics.end(), statistics);
        return synSuccess;
    }
    catch (...) { return fail(); }
}

synStatus NativeComputeGraph::replay()
{
    synStatus status = beginReplay();
    if (status != synSuccess)
    {
        return status;
    }

    ScalLongSyncObject completion = LongSoEmpty;
    for (size_t index = 0; index < m_segments.size(); ++index)
    {
        const bool submitLast = index + 1 == m_segments.size();
        status = replaySegment(index, nullptr, submitLast, completion);
        if (status != synSuccess)
        {
            return status;
        }
    }

    return synSuccess;
}

synStatus NativeComputeGraph::beginReplay()
{
    if (m_state != State::INSTANTIATED || m_replayActive || m_jointPlan)
    {
        return synObjectNotInitialized;
    }
    m_nextReplaySegment = 0;
    m_replayActive      = true;
    return synSuccess;
}

synStatus NativeComputeGraph::replaySegment(size_t                    segmentIndex,
                                            const ScalLongSyncObject* dependency,
                                            bool                      submit,
                                            ScalLongSyncObject&       completion)
{
    if (m_state != State::INSTANTIATED || !m_replayActive || segmentIndex != m_nextReplaySegment ||
        segmentIndex >= m_segments.size())
    {
        return synInvalidArgument;
    }
    const bool isLast = segmentIndex + 1 == m_segments.size();
    if (submit != isLast)
    {
        // A unified replay stages every segment and publishes the compute CCB
        // only after the final segment has been appended.
        m_state        = State::INVALID;
        m_replayActive = false;
        return synInvalidArgument;
    }

    if (dependency != nullptr && dependency->m_targetValue != 0)
    {
        const synStatus waitStatus = m_streamBase.longSoWaitOnDevice(*dependency, false /* isUserReq */);
        if (waitStatus != synSuccess)
        {
            m_state        = State::INVALID;
            m_replayActive = false;
            return waitStatus;
        }
    }

    completion = LongSoEmpty;
    const synStatus status = m_streamBase.replayNativeCommands(m_segments[segmentIndex].commands,
                                                               false /* preserveCapturedSubmissions */,
                                                               submit,
                                                               completion);
    if (status != synSuccess)
    {
        m_state        = State::INVALID;
        m_replayActive = false;
        return status;
    }
    if (!(completion != LongSoEmpty))
    {
        m_state        = State::INVALID;
        m_replayActive = false;
        return synFail;
    }

    ++m_nextReplaySegment;
    if (isLast)
    {
        m_lastCompletion = completion;
        ++m_replayCount;
        m_replayActive = false;
    }

    return synSuccess;
}

synStatus NativeComputeGraph::close()
{
    if (m_state == State::CLOSED)
    {
        return synSuccess;
    }

    synStatus status = synSuccess;
    if (!m_segments.empty())
    {
        status = m_stream.longSoWaitForLast(true, std::numeric_limits<uint64_t>::max(), __FUNCTION__);
        if (status != synSuccess)
        {
            return status;
        }
    }

    for (CompiledSegment& segment : m_segments)
    {
        if (segment.memory != nullptr && segment.memory->initialized() && !segment.memory->closed())
        {
            status = segment.memory->close();
            if (status != synSuccess)
            {
                return status;
            }
        }
    }
    m_segments.clear();
    m_state = State::CLOSED;
    return synSuccess;
}

QueueComputeScal::QueueComputeScal(const BasicQueueInfo&           rBasicQueueInfo,
                                   ScalStreamComputeInterface*     pScalStream,
                                   const ComputeCompoundResources* pComputeResources,
                                   synDeviceType                   deviceType,
                                   const ScalDevSpecificInfo&      rDevSpecificInfo,
                                   DevMemoryAllocInterface&        devMemoryAlloc,
                                   ScalMemoryPool&                 globalPool,
                                   ScalMemoryPool&                 hbmSharedPool)
: QueueBaseScalCommon(rBasicQueueInfo, pScalStream),
  m_deviceType(deviceType),
  m_rDevSpecificInfo(rDevSpecificInfo),
  m_memMgrs(m_scalStream->getName(), devMemoryAlloc),
  m_launchTracker(GCFG_NUM_OF_CSDC_TO_CHK.value()),
  m_devMemoryAlloc(devMemoryAlloc),
  m_globalPool(globalPool),
  m_hbmSharedPool(hbmSharedPool),
  m_computeResources(*pComputeResources)
{
    // Currently, we will have internal-signal for the TX-Command
    // Hence, in case that is the compute-stream, this will need to be avoided
    HB_ASSERT(!(pComputeResources->m_pTxCommandsStream->isComputeStream()),
              "scal_stream_copy_xxx::memcopy is needed to be updated to supprt this mode");
}

QueueComputeScal::~QueueComputeScal() {}

std::unique_ptr<NativeComputeGraph> QueueComputeScal::createNativeComputeGraph()
{
    ScalStreamComputeInterface* stream = dynamic_cast<ScalStreamComputeInterface*>(m_scalStream);
    if (stream == nullptr)
    {
        return nullptr;
    }
    return std::make_unique<NativeComputeGraph>(*stream, m_globalPool, m_hbmSharedPool);
}

synStatus QueueComputeScal::beginNativeComputeGraphCapture(NativeComputeGraph& graph)
{
    std::lock_guard<std::timed_mutex> lock(m_userOpLock);
    if (m_activeNativeGraph != nullptr)
    {
        return synObjectAlreadyInitialized;
    }
    const synStatus status = graph.beginCapture();
    if (status == synSuccess)
    {
        m_activeNativeGraph = &graph;
    }
    return status;
}

synStatus QueueComputeScal::endNativeComputeGraphCapture(NativeComputeGraph& graph)
{
    std::lock_guard<std::timed_mutex> lock(m_userOpLock);
    if (m_activeNativeGraph != &graph)
    {
        return synInvalidArgument;
    }
    const synStatus status = graph.endCapture();
    m_activeNativeGraph    = nullptr;
    return status;
}

synStatus QueueComputeScal::abortNativeComputeGraphCapture(NativeComputeGraph& graph)
{
    std::lock_guard<std::timed_mutex> lock(m_userOpLock);
    if (m_activeNativeGraph != &graph)
    {
        return synInvalidArgument;
    }
    graph.invalidate();
    m_activeNativeGraph = nullptr;
    return synSuccess;
}

synStatus QueueComputeScal::replayNativeComputeGraph(NativeComputeGraph& graph)
{
    std::lock_guard<std::timed_mutex> lock(m_userOpLock);
    if (m_activeNativeGraph != nullptr)
    {
        return synInvalidArgument;
    }
    return graph.replay();
}

synStatus QueueComputeScal::beginNativeComputeGraphReplay(NativeComputeGraph& graph)
{
    std::lock_guard<std::timed_mutex> lock(m_userOpLock);
    if (m_activeNativeGraph != nullptr)
    {
        return synInvalidArgument;
    }
    return graph.beginReplay();
}

synStatus QueueComputeScal::replayNativeComputeGraphSegment(NativeComputeGraph&       graph,
                                                            size_t                    segmentIndex,
                                                            const ScalLongSyncObject* dependency,
                                                            bool                      submit,
                                                            ScalLongSyncObject&       completion)
{
    std::lock_guard<std::timed_mutex> lock(m_userOpLock);
    if (m_activeNativeGraph != nullptr)
    {
        return synInvalidArgument;
    }
    return graph.replaySegment(segmentIndex, dependency, submit, completion);
}

synStatus QueueComputeScal::prepareNativeComputePlan(NativeComputeGraph& graph, const uint32_t* consumers,
    uint64_t count, uint32_t nicIndex, synNativeGraphExchangeReplay callback, void* context, const uint32_t* producers)
{
    std::lock_guard<std::timed_mutex> lock(m_userOpLock);
    if (m_activeNativeGraph != nullptr) return synInvalidArgument;
    return graph.preparePlan(consumers, count, nicIndex, callback, context, producers);
}

synStatus QueueComputeScal::prepareSegmentedNativeComputePlan(
    NativeComputeGraph& graph, const uint32_t* producers, const uint32_t* consumers,
    uint64_t count, uint64_t prefixCollectives, uint32_t prefixComputes, uint32_t nicIndex,
    synNativeGraphExchangeReplay callback, void* context)
{
    std::lock_guard<std::timed_mutex> lock(m_userOpLock);
    if (m_activeNativeGraph != nullptr) return synInvalidArgument;
    return graph.prepareSegmentedPlan(producers, consumers, count, prefixCollectives,
                                      prefixComputes, nicIndex, callback, context);
}

synStatus QueueComputeScal::replayNativeComputePlan(NativeComputeGraph& graph, synNativeGraphSyncInfo& completion,
                                                 uint64_t* statistics, uint64_t count)
{
    std::lock_guard<std::timed_mutex> lock(m_userOpLock);
    if (m_activeNativeGraph != nullptr) return synInvalidArgument;
    return graph.replayPlan(completion, statistics, count);
}

synStatus QueueComputeScal::replayNativeComputePlanPrefix(
    NativeComputeGraph& graph, synNativeGraphSyncInfo& completion,
    uint64_t* statistics, uint64_t count)
{
    std::lock_guard<std::timed_mutex> lock(m_userOpLock);
    if (m_activeNativeGraph != nullptr) return synInvalidArgument;
    return graph.replayPlanPrefix(completion, statistics, count);
}

synStatus QueueComputeScal::replayNativeComputePlanFinish(
    NativeComputeGraph& graph, synNativeGraphSyncInfo& completion,
    uint64_t* statistics, uint64_t count)
{
    std::lock_guard<std::timed_mutex> lock(m_userOpLock);
    if (m_activeNativeGraph != nullptr) return synInvalidArgument;
    return graph.replayPlanFinish(completion, statistics, count);
}

namespace {
class NativeComputeReplayJob final : public StreamJob
{
public:
    enum class Part { Full, Prefix, Finish };
    NativeComputeReplayJob(InternalNativeComputeGraphHandle& owner, Part part)
    : StreamJob(JobType::COMPUTE), m_owner(owner), m_part(part) {}

    synStatus run(QueueInterface* queue) override
    {
        if (queue != m_owner.queue || !m_owner.replayCompletion || !m_owner.replayStatistics)
            return synInvalidArgument;
        if (m_part == Part::Prefix)
            return m_owner.queue->replayNativeComputePlanPrefix(*m_owner.graph, *m_owner.replayCompletion,
                                                                m_owner.replayStatistics,
                                                                m_owner.replayStatisticCount);
        if (m_part == Part::Finish)
            return m_owner.queue->replayNativeComputePlanFinish(*m_owner.graph, *m_owner.replayCompletion,
                                                                m_owner.replayStatistics,
                                                                m_owner.replayStatisticCount);
        return m_owner.queue->replayNativeComputePlan(*m_owner.graph, *m_owner.replayCompletion,
                                                      m_owner.replayStatistics,
                                                      m_owner.replayStatisticCount);
    }
private:
    InternalNativeComputeGraphHandle& m_owner;
    Part m_part;
};
}

extern "C" synStatus SYN_API_CALL synNativeComputeGraphPreparePlan(synNativeComputeGraphHandle handle,
    const uint32_t* consumers, uint64_t count, uint32_t nicIndex, synNativeGraphExchangeReplay callback, void* context)
{
    if (!handle || !handle->queue || !handle->graph || !handle->stream) return synInvalidArgument;
    std::lock_guard<std::mutex> lock(handle->replayMutex);
    if (handle->preparedReplayJob) return synInvalidArgument;
    const synStatus status = handle->queue->prepareNativeComputePlan(*handle->graph, consumers, count, nicIndex,
                                                                    callback, context);
    if (status != synSuccess) return status;
    // Instantiated once. Stream::addJob runs this same object synchronously;
    // no per-recipe executor or per-replay job allocation is introduced.
    handle->preparedReplayJob =
        std::make_unique<NativeComputeReplayJob>(*handle, NativeComputeReplayJob::Part::Full);
    return synSuccess;
}

extern "C" synStatus SYN_API_CALL synNativeComputeGraphPreparePlanV2(synNativeComputeGraphHandle handle,
    const uint32_t* producers, const uint32_t* consumers, uint64_t count, uint32_t nicIndex, synNativeGraphExchangeReplay callback, void* context)
{
    if (!producers || !handle || !handle->queue || !handle->graph || !handle->stream) return synInvalidArgument;
    std::lock_guard<std::mutex> lock(handle->replayMutex);
    if (handle->preparedReplayJob) return synInvalidArgument;
    const synStatus status = handle->queue->prepareNativeComputePlan(*handle->graph, consumers, count, nicIndex,
                                                                    callback, context, producers);
    if (status != synSuccess) return status;
    // Instantiated once. Stream::addJob runs this same object synchronously;
    // no per-recipe executor or per-replay job allocation is introduced.
    handle->preparedReplayJob =
        std::make_unique<NativeComputeReplayJob>(*handle, NativeComputeReplayJob::Part::Full);
    return synSuccess;
}

extern "C" synStatus SYN_API_CALL synNativeComputeGraphPrepareSegmentedPlanV2(
    synNativeComputeGraphHandle handle, const uint32_t* producers, const uint32_t* consumers,
    uint64_t count, uint64_t prefixCollectives, uint32_t nicIndex,
    synNativeGraphExchangeReplay callback, void* context)
{
    if (!consumers || !prefixCollectives || prefixCollectives >= count) return synInvalidArgument;
    return synNativeComputeGraphPrepareSegmentedPlanV3(
        handle, producers, consumers, count, prefixCollectives, consumers[prefixCollectives],
        nicIndex, callback, context);
}

extern "C" synStatus SYN_API_CALL synNativeComputeGraphPrepareSegmentedPlanV3(
    synNativeComputeGraphHandle handle, const uint32_t* producers, const uint32_t* consumers,
    uint64_t count, uint64_t prefixCollectives, uint32_t prefixComputes, uint32_t nicIndex,
    synNativeGraphExchangeReplay callback, void* context)
{
    if (!producers || !handle || !handle->queue || !handle->graph || !handle->stream)
        return synInvalidArgument;
    std::lock_guard<std::mutex> lock(handle->replayMutex);
    if (handle->preparedReplayJob || handle->preparedPrefixReplayJob || handle->preparedFinishReplayJob)
        return synInvalidArgument;
    const synStatus status = handle->queue->prepareSegmentedNativeComputePlan(
        *handle->graph, producers, consumers, count, prefixCollectives, prefixComputes, nicIndex, callback, context);
    if (status != synSuccess) return status;
    handle->preparedReplayJob =
        std::make_unique<NativeComputeReplayJob>(*handle, NativeComputeReplayJob::Part::Full);
    handle->preparedPrefixReplayJob =
        std::make_unique<NativeComputeReplayJob>(*handle, NativeComputeReplayJob::Part::Prefix);
    handle->preparedFinishReplayJob =
        std::make_unique<NativeComputeReplayJob>(*handle, NativeComputeReplayJob::Part::Finish);
    return synSuccess;
}

extern "C" synStatus SYN_API_CALL synNativeComputeGraphPrepareBoundedPlanV1(
    synNativeComputeGraphHandle handle, const uint32_t* producers, const uint32_t* consumers,
    uint64_t collectives, const uint32_t* minimum, uint64_t segments, uint32_t limit,
    uint32_t nicIndex, synNativeGraphExchangeReplay callback, void* context)
{
    const auto status = synNativeComputeGraphPreparePlanV2(
        handle, producers, consumers, collectives, nicIndex, callback, context);
    if (status != synSuccess) return status;
    std::lock_guard<std::mutex> lock(handle->replayMutex);
    return handle->graph->prepareBoundedVariants(minimum, segments, producers, consumers, collectives, limit);
}

extern "C" synStatus SYN_API_CALL synNativeComputeGraphGetWorkspaceBytes(synNativeComputeGraphHandle handle,
                                                                        uint64_t* bytes)
{
    if (!handle || !handle->graph || !bytes) return synInvalidArgument;
    std::lock_guard<std::mutex> lock(handle->replayMutex);
    *bytes = handle->graph->workspaceAllocationBytes();
    return synSuccess;
}

extern "C" synStatus SYN_API_CALL synNativeComputeGraphReplayPlan(synNativeComputeGraphHandle handle,
    synNativeGraphSyncInfo* completion, uint64_t* statistics, uint64_t count)
{
    if (!handle || !handle->queue || !handle->graph || !handle->stream || !handle->preparedReplayJob ||
        !completion || !statistics || count != 12) return synInvalidArgument;
    std::lock_guard<std::mutex> lock(handle->replayMutex);
    handle->replayCompletion = completion;
    handle->replayStatistics = statistics;
    handle->replayStatisticCount = count;
    synStatus status;
    try {
        // The logical stream drains pending device waits and performs the
        // NETWORK/DMA -> COMPUTE handoff once for the entire graph. Bypassing
        // it leaves both input waits and final event queue ownership stale.
        status = handle->stream->addJob(handle->preparedReplayJob);
    } catch (...) { status = synFail; }
    handle->replayCompletion = nullptr;
    handle->replayStatistics = nullptr;
    handle->replayStatisticCount = 0;
    if (status != synSuccess) handle->graph->invalidate();
    return status;
}

static synStatus replayNativeComputeGraphPart(
    synNativeComputeGraphHandle handle, std::unique_ptr<StreamJob>& job,
    synNativeGraphSyncInfo* completion, uint64_t* statistics, uint64_t count, uint32_t bound = UINT32_MAX)
{
    if (!handle || !handle->queue || !handle->graph || !handle->stream || !job ||
        !completion || !statistics || count != 12) return synInvalidArgument;
    std::lock_guard<std::mutex> lock(handle->replayMutex);
    if (bound != UINT32_MAX && handle->graph->selectBoundedVariant(bound) != synSuccess)
        return synInvalidArgument;
    handle->replayCompletion = completion;
    handle->replayStatistics = statistics;
    handle->replayStatisticCount = count;
    synStatus status;
    try { status = handle->stream->addJob(job); }
    catch (...) { status = synFail; }
    handle->replayCompletion = nullptr;
    handle->replayStatistics = nullptr;
    handle->replayStatisticCount = 0;
    if (status != synSuccess) handle->graph->invalidate();
    return status;
}

extern "C" synStatus SYN_API_CALL synNativeComputeGraphReplayBoundedPlanV1(
    synNativeComputeGraphHandle handle, uint32_t bound, synNativeGraphSyncInfo* completion,
    uint64_t* statistics, uint64_t count)
{
    if (!handle || bound > 8) return synInvalidArgument;
    return replayNativeComputeGraphPart(handle, handle->preparedReplayJob, completion, statistics, count, bound);
}

extern "C" synStatus SYN_API_CALL synNativeComputeGraphReplayPlanPrefix(
    synNativeComputeGraphHandle handle, synNativeGraphSyncInfo* completion,
    uint64_t* statistics, uint64_t count)
{
    if (!handle) return synInvalidArgument;
    return replayNativeComputeGraphPart(handle, handle->preparedPrefixReplayJob,
                                        completion, statistics, count);
}

extern "C" synStatus SYN_API_CALL synNativeComputeGraphReplayPlanFinish(
    synNativeComputeGraphHandle handle, synNativeGraphSyncInfo* completion,
    uint64_t* statistics, uint64_t count)
{
    if (!handle) return synInvalidArgument;
    return replayNativeComputeGraphPart(handle, handle->preparedFinishReplayJob,
                                        completion, statistics, count);
}

synStatus QueueComputeScal::eventRecord(EventInterface& rEventInterface, synStreamHandle streamHandle)
{
    LOG_DEBUG(SYN_STREAM, "{} Stream {}", HLLOG_FUNC, m_basicQueueInfo.getDescription());
#ifdef DISABLE_SYNC_ON_DEV
    synchronizeStream(streamHandle);
    return synSuccess;
#endif
    ScalEvent& rScalEvent = dynamic_cast<ScalEvent&>(rEventInterface);
    rScalEvent.clearState();
    rScalEvent.pStreamIfScal = this;

    // handle the case where last cmd on the stream is 'wait'
    addCompletionAfterWait();
    return m_scalStream->eventRecord(true, rScalEvent);
}

synStatus QueueComputeScal::eventWait(const EventInterface& rEventInterface,
                                      const unsigned int    flags,
                                      synStreamHandle       streamHandle)
{
    LOG_DEBUG(SYN_STREAM, "{} Stream {}", HLLOG_FUNC, m_basicQueueInfo.getDescription());

#ifdef DISABLE_SYNC_ON_DEV
    return synSuccess;
#endif

    const ScalEvent& tmpScalEvent = dynamic_cast<const ScalEvent&>(rEventInterface);
    // defend against overriding the event while we're working on it
    ScalEvent rcScalEvent = tmpScalEvent;

    ScalLongSyncObject longSo = rcScalEvent.longSo;
    if (rcScalEvent.isOnHclStream())
    {
        longSo.m_index       = rcScalEvent.hclSyncInfo.long_so_index;
        longSo.m_targetValue = rcScalEvent.hclSyncInfo.targetValue;
    }

    synStatus status;
    {
        std::lock_guard<std::timed_mutex> lock(m_userOpLock);
        status = m_scalStream->longSoWaitOnDevice(longSo, true);
    }

    return status;
}

synStatus QueueComputeScal::query()
{
    // handle the case where last cmd on the stream is 'wait'
    addCompletionAfterWait();
    return m_scalStream->longSoWaitForLast(true, 0, __FUNCTION__);
}

synStatus QueueComputeScal::synchronize(synStreamHandle streamHandle, bool isUserRequest)
{
    return waitForLastLongSo(isUserRequest);
}

// this function can be called ONLY after successfull recipeLauncher->launch(launchInfo);
void QueueComputeScal::updateSfgEventsScalStream(EventWithMappedTensorDB& events)
{
    for (auto& event : events)
    {
        // validation is done beforehand
        auto scalEvent           = static_cast<ScalEvent*>(event.get());
        scalEvent->pStreamIfScal = this;
    }
}

/*
 ***************************************************************************************************
 *   @brief init() inits the recipeBuffs (used to move/cache the recipe in the HBM)
 *
 *   @param  Node
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus QueueComputeScal::init(PreAllocatedStreamMemoryAll& preAllocatedInfo)
{
    if (m_initDone == true)
    {
        LOG_CRITICAL_T(SYN_STREAM,
                       "Stream {:x} m_scalStream {:x} steam {} already init",
                       TO64(this),
                       TO64(m_scalStream),
                       m_basicQueueInfo.getDescription());
        return synFail;
    }

    m_hbmGlblSize    = preAllocatedInfo.global.Size;
    m_hbmGlblAddr    = preAllocatedInfo.global.AddrDev;
    m_arcHbmSize     = preAllocatedInfo.shared.Size;
    m_arcHbmAddrCore = preAllocatedInfo.shared.AddrCore;
    m_arcHbmAddrDev  = preAllocatedInfo.shared.AddrDev;

    {
        synStatus status = initMemMgrs();
        if (status != synSuccess)
        {
            return status;
        }
    }

    m_initDone = true;
    return synSuccess;
}

/*
 ***************************************************************************************************
 *   @brief initMemMgrs() inits all the memory managers
 *
 *   @param  None
 *   @return None
 *
 ***************************************************************************************************
 */
synStatus QueueComputeScal::initMemMgrs()
{
    m_memMgrs.hbmGlblMemMgr.init(m_hbmGlblAddr, m_hbmGlblSize);
    m_memMgrs.arcHbmMemMgr.init(m_arcHbmAddrCore, m_arcHbmAddrDev, m_arcHbmSize);
    return m_memMgrs.mappedMemMgr.init();
}

/*
 ***************************************************************************************************
 * get hbm global memory for one program (simultaneously HBM_BUFFS_AMOUNT can reside in this memory)
 *
 * @param maxGlbHbmMemorySize
 * @return chunk size for one program
 ***************************************************************************************************
 */
uint64_t QueueComputeScal::getHbmGlblMaxRecipeSize(uint64_t maxGlbHbmMemorySize)
{
    return HbmGlblMemMgr::getMaxRecipeSize(maxGlbHbmMemorySize);
}

/*
 ***************************************************************************************************
 * get hbm shared memory for one program (simultaneously HBM_BUFFS_AMOUNT can reside in this memory)
 *
 * @param maxSharedHbmMemorySize
 * @return chunk size for one program
 ***************************************************************************************************
 */
uint64_t QueueComputeScal::getHbmSharedMaxRecipeSize(uint64_t maxArcHbmMemorySize)
{
    return ArcHbmMemMgr::getMaxRecipeSize(maxArcHbmMemorySize);
}

/*
 ***************************************************************************************************
 * get getDynamicShapeProcessor
 * @return DynamicRecipe
 ***************************************************************************************************
 */
DynamicRecipe* QueueComputeScal::getDynamicShapeProcessor(const LaunchInfo& launchInfo)
{
    uint64_t recipeId        = launchInfo.pRecipeHandle->recipeSeqNum;
    auto     dsdProcessorItr = m_dynamicShapeProcessor.find(recipeId);

    if (dsdProcessorItr != m_dynamicShapeProcessor.end())
    {
        LOG_DEBUG_T(SYN_STREAM, "DSD processor for recipe 0x{:x} already on stream", (uint64_t)recipeId);
    }
    else
    {
        const DeviceAgnosticRecipeInfo& deviceAgnostcInfo   = launchInfo.pRecipeHandle->deviceAgnosticRecipeHandle;
        const RecipeAddrPatcher*        pRecipeAddrPatcher  = &deviceAgnostcInfo.m_recipeStaticInfoScal.recipeAddrPatcher;
        std::unique_ptr<DynamicRecipe>  dsdProcessor        =
            std::make_unique<DynamicRecipe>(launchInfo.pRecipeHandle->basicRecipeHandle,
                                            deviceAgnostcInfo,
                                            &deviceAgnostcInfo.m_recipeStaticInfoScal.recipeDsdPpInfo.getDsdDCPatchingInfo(),
                                            pRecipeAddrPatcher);
        m_dynamicShapeProcessor[recipeId] = std::move(dsdProcessor);
    }
    return m_dynamicShapeProcessor[recipeId].get();
}

synStatus QueueComputeScal::launch(const synLaunchTensorInfoExt* launchTensorsInfo,
                                   uint32_t                      launchTensorsAmount,
                                   uint64_t                      workspaceAddress,
                                   InternalRecipeHandle*         pRecipeHandle,
                                   uint64_t                      assertAsyncMappedAddress,
                                   uint32_t                      flags,
                                   EventWithMappedTensorDB&      events,
                                   uint8_t                       apiId)
{
    CHECK_POINTER(SYN_STREAM, pRecipeHandle, "pRecipeHandle", synFail);
    basicRecipeInfo& basicRecipeHandle = pRecipeHandle->basicRecipeHandle;
    HB_ASSERT_PTR(basicRecipeHandle.recipe);

    const uint64_t recipeHandle = (uint64_t)pRecipeHandle;
    const uint64_t recipeSeqNum = pRecipeHandle->recipeSeqNum;
    const uint64_t seqId        = QueueComputeUtils::getLaunchSeqId();
    const uint64_t inFlight     = m_launchTracker.size();

    STAT_GLBL_COLLECT(seqId, seqId);
    STAT_GLBL_COLLECT(pRecipeHandle->recipeSeqNum, recipeId);

    PROFILER_COLLECT_TIME()

    LOG_DEBUG_T(
        SYN_STREAM,
        "{}: Stream {:#x} recipeHandle {:#x} recipeSeqNum {} seqId {} inFlight {} workspaceAddress {:#x} apiId {}",
        HLLOG_FUNC,
        TO64(this),
        recipeHandle,
        recipeSeqNum,
        seqId,
        inFlight,
        workspaceAddress,
        apiId);

    if (QueueComputeUtils::isRecipeEmpty(*pRecipeHandle))
    {
        return synSuccess;
    }

    synStatus status = QueueComputeUtils::prepareLaunch(m_basicQueueInfo,
                                                        launchTensorsInfo,
                                                        launchTensorsAmount,
                                                        pRecipeHandle,
                                                        events,
                                                        flags);
    if (status != synSuccess)
    {
        return status;
    }

    LaunchInfo launchInfo(launchTensorsInfo,
                          launchTensorsAmount,
                          workspaceAddress,
                          assertAsyncMappedAddress,
                          pRecipeHandle,
                          flags,
                          m_rDevSpecificInfo,
                          events,
                          apiId);

    status = launch(launchInfo);

    if (status != synSuccess)
    {
        LOG_ERR_T(SYN_STREAM, "SCAL launch failed with status {}", status);
        return status;
    }

    ProfilerApi::setHostProfilerApiId(apiId);

    const std::string profilerStr(
        "enqueueWithExternalEvents recipeHandle " + fmt::format(FMT_COMPILE("{:#x}"), recipeHandle) + " recipeSeqNum " +
        std::to_string(recipeSeqNum) + " seqId " + std::to_string(seqId) + " inFlight " + std::to_string(inFlight));
    PROFILER_MEASURE_TIME(profilerStr.c_str());

    return synSuccess;
}

/*
 ***************************************************************************************************
 *   @brief launch() Does the launch on the stream. It gets the HBM addresses from the recipeBuffs
 *   and sends the needed commands to scal to do the work.
 *   Based on some habana global config flags it also:
 *   read the recipe from the HBM and compares it to the recipe from the GC
 *   Dumps the recipe to the log
 *   Skips sending the commands to scal (for testing)
 *
 *   @param  LaunchInfo - all information needed for this specific launch
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus QueueComputeScal::launch(const LaunchInfo& launchInfo)
{
    std::lock_guard<std::timed_mutex> lock(m_userOpLock);

    STAT_GLBL_START(scalChkCompletion);
    uint64_t numOfCompletionsCopy, numOfCompletionsCompute;
    m_launchTracker.checkForCompletion(numOfCompletionsCopy, numOfCompletionsCompute);
    STAT_GLBL_COLLECT_TIME(scalChkCompletion, globalStatPointsEnum::scalChkCompletion);

    DynamicRecipe* dsdProcessor = nullptr;
    bool           isDsd        = RecipeUtils::isDsd(launchInfo.pRecipeHandle->basicRecipeHandle);

    if (isDsd)
    {
        dsdProcessor = getDynamicShapeProcessor(launchInfo);
    }

    ScalStreamComputeInterface* pComputeScalStream = dynamic_cast<ScalStreamComputeInterface*>(m_scalStream);
    VERIFY_IS_NULL_POINTER(SYN_STREAM, pComputeScalStream, "pComputeScalStream");

    CompiledSegment* nativeSegment = nullptr;
    if (m_activeNativeGraph != nullptr)
    {
        if (isDsd || RecipeUtils::isIH2DRecipe(launchInfo.pRecipeHandle->basicRecipeHandle.recipe))
        {
            LOG_ERR(SYN_STREAM, "Native compute graph capture supports static recipes without IH2D");
            m_activeNativeGraph->invalidate();
            return synUnsupported;
        }
        synStatus nativeStatus = synSuccess;
        const RecipeStaticInfoScal& staticInfo =
            launchInfo.pRecipeHandle->deviceAgnosticRecipeHandle.m_recipeStaticInfoScal;
        const recipe_t& capturedRecipe = *launchInfo.pRecipeHandle->basicRecipeHandle.recipe;
        // The reserved PROGRAM range used by section validation can extend
        // beyond the downloaded execution blobs. A standalone allocation must
        // own that entire range, as well as the physical SCAL recipe sections.
        // The usual program arena hides this difference in its larger chunks.
        uint64_t nativeGlobalSize = staticInfo.m_glbHbmSizeTotal;
        for (const auto section : {PROGRAM_DATA, NON_PATCHABLE})
        {
            const uint64_t size = capturedRecipe.workspace_sizes[
                section == PROGRAM_DATA ? MEMORY_ID_RESERVED_FOR_PROGRAM_DATA : MEMORY_ID_RESERVED_FOR_PROGRAM];
            const uint64_t offset = staticInfo.recipeSections[section].offsetHbm;
            if (size > std::numeric_limits<uint64_t>::max() - offset)
            {
                m_activeNativeGraph->invalidate();
                return synInvalidArgument;
            }
            nativeGlobalSize = std::max(nativeGlobalSize, offset + size);
        }
        nativeSegment = m_activeNativeGraph->addPendingSegment(nativeGlobalSize,
                                                               staticInfo.m_arcHbmSize,
                                                               launchInfo.pRecipeHandle->recipeSeqNum,
                                                               launchInfo.pEnqueueTensorsInfo,
                                                               launchInfo.pEnqueueTensorsInfoAmount,
                                                               launchInfo.workspaceAddress,
                                                               launchInfo.pRecipeHandle->basicRecipeHandle.recipe
                                                                   ->workspace_sizes[MEMORY_ID_RESERVED_FOR_WORKSPACE],
                                                               nativeStatus);
        if (nativeStatus != synSuccess || nativeSegment == nullptr)
        {
            m_activeNativeGraph->invalidate();
            return nativeStatus == synSuccess ? synFail : nativeStatus;
        }
    }

    std::unique_ptr<RecipeLauncher> recipeLauncher = std::make_unique<RecipeLauncher>(pComputeScalStream,
                                                                                      &m_computeResources,
                                                                                      m_devMemoryAlloc,
                                                                                      m_memMgrs,
                                                                                      m_launchTracker,
                                                                                      launchInfo.pRecipeHandle,
                                                                                      dsdProcessor,
                                                                                      ++m_runningId,
                                                                                      launchInfo.m_apiId,
                                                                                      nativeSegment == nullptr ? nullptr : nativeSegment->memory.get(),
                                                                                      nativeSegment == nullptr ? nullptr : &nativeSegment->commands);

    const LaunchInfo retainedLaunchInfo(launchInfo.pEnqueueTensorsInfo, launchInfo.pEnqueueTensorsInfoAmount,
        nativeSegment ? nativeSegment->workspaceAddress : launchInfo.workspaceAddress,
        launchInfo.m_assertAsyncMappedAddress, const_cast<InternalRecipeHandle*>(launchInfo.pRecipeHandle),
        launchInfo.launchFlags, launchInfo.devSpecificInfo, launchInfo.events, launchInfo.m_apiId);
    synStatus status = recipeLauncher->launch(retainedLaunchInfo);
    if (status != synSuccess)
    {
        if (m_activeNativeGraph != nullptr)
        {
            m_activeNativeGraph->invalidate();
        }
        LOG_ERR(SYN_STREAM,
                "{}: recipeLauncher launch failed, recipe seq {}",
                HLLOG_FUNC,
                launchInfo.pRecipeHandle->recipeSeqNum);
        // in case of failure after enqueue, add launcher to tracker anyway
        if (status == synFailedSectionValidation)
        {
            m_launchTracker.add(std::move(recipeLauncher));
        }
        return status;
    }

    if (nativeSegment != nullptr)
    {
        if (nativeSegment->commands.empty())
        {
            m_activeNativeGraph->invalidate();
            return synFail;
        }
        nativeSegment->ready = true;
    }

    updateSfgEventsScalStream(launchInfo.events);
    m_launchTracker.add(std::move(recipeLauncher));

    return synSuccess;
}

/*
 ***************************************************************************************************
 *   @brief getMappedMemorySize() returns the size of mapped memory used by this stream
 *
 *   The size returned is of the global-hbm memory size (mapped memory mgr), not arc-shared memory
 *
 *   @return status
 *
 ***************************************************************************************************
 */
synStatus QueueComputeScal::getMappedMemorySize(uint64_t& mappedMemorySize) const
{
    mappedMemorySize = m_memMgrs.mappedMemMgr.getMappedMemorySize();
    return synSuccess;
}

void QueueComputeScal::notifyRecipeRemoval(InternalRecipeHandle& rRecipeHandle)
{
    std::lock_guard<std::timed_mutex> lock(m_userOpLock);

    m_launchTracker.checkRecipeCompletion(rRecipeHandle);

    RecipeSeqId recipeSeqId(rRecipeHandle.recipeSeqNum);
    m_memMgrs.mappedMemMgr.removeId(recipeSeqId);

    auto recipeProcessorsIter = m_dynamicShapeProcessor.find(rRecipeHandle.recipeSeqNum);
    if (recipeProcessorsIter != m_dynamicShapeProcessor.end())
    {
        m_dynamicShapeProcessor.erase(recipeProcessorsIter);
    }
}

void QueueComputeScal::notifyAllRecipeRemoval()
{
    std::lock_guard<std::timed_mutex> lock(m_userOpLock);

    m_launchTracker.checkForCompletionAll();

    m_memMgrs.mappedMemMgr.removeAllId();

    m_dynamicShapeProcessor.clear();
}

/**
 * log all recipes related to the current stream
 * @param oldestRecipeOnly if true, logs only the oldest recipe (=the one that failed). otherwise, logs all related
 * recipes.
 */
void QueueComputeScal::dfaUniqStreamInfo(bool               oldestRecipeOnly,
                                         uint64_t           currentLongSo,
                                         bool               dumpRecipe,
                                         const std::string& callerMsg)
{
    if (!oldestRecipeOnly)
    {
        m_launchTracker.dfaLogRecipesDesc(oldestRecipeOnly, currentLongSo, dumpRecipe, callerMsg, true);
    }

    LOG_INFO(SYN_DEV_FAIL, "- Related recipe(s) -");

    m_launchTracker.dfaLogRecipesDesc(oldestRecipeOnly, currentLongSo, dumpRecipe, callerMsg, false);
}

synStatus QueueComputeScal::getDynamicShapesTensorInfoArray(synRecipeHandle             recipeHandle,
                                                            std::vector<tensor_info_t>& tensorInfoArray) const
{
    VERIFY_IS_NULL_POINTER(SYN_API, recipeHandle, "recipeHandle");
    uint64_t recipeId        = recipeHandle->recipeSeqNum;
    auto     dsdProcessorItr = m_dynamicShapeProcessor.find(recipeId);

    if (dsdProcessorItr == m_dynamicShapeProcessor.end())
    {
        LOG_DEBUG_T(SYN_STREAM, "DSD processor for recipe 0x{:x} doesn't exist", (uint64_t)recipeId);
        return synInvalidArgument;
    }

    tensorInfoArray = dsdProcessorItr->second.get()->getDynamicShapesTensorInfoArray();
    return synSuccess;
}

std::set<ScalStreamCopyInterface*> QueueComputeScal::dfaGetQueueScalStreams()
{
    std::set<ScalStreamCopyInterface*> set;

    set.insert(m_scalStream);
    set.insert(m_computeResources.m_pTxCommandsStream);
    set.insert(m_computeResources.m_pDev2DevCommandsStream);
    set.insert(m_computeResources.m_pRxCommandsStream);

    return set;
}
