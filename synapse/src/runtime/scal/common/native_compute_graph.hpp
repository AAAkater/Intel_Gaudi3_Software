#pragma once

#include "runtime/scal/common/entities/scal_stream_base.hpp"
#include "runtime/scal/common/native_recipe_memory.hpp"
#include "synapse_api_types.h"

#include <cstdint>
#include <memory>
#include <vector>
#include <array>

class ScalMemoryPool;
class ScalStreamComputeInterface;

struct NativeTensorBinding
{
    uint64_t      tensorId      = 0;
    uint64_t      deviceAddress = 0;
    synTensorType tensorType    = DATA_TENSOR;
    TSize         sizes[HABANA_DIM_MAX] {};
};

struct CompiledSegment
{
    std::unique_ptr<NativeRecipeMemory> memory;
    NativeScalCommandBuffer             commands;
    std::vector<NativeTensorBinding>    bindings;
    uint64_t                            workspaceAddress = 0;
    uint64_t                            recipeSequence   = 0;
    bool                                ready            = false;
};

class NativeComputeGraph
{
public:
    enum class State
    {
        CREATED,
        CAPTURING,
        INSTANTIATED,
        INVALID,
        CLOSED
    };

    NativeComputeGraph(ScalStreamComputeInterface& stream,
                       ScalMemoryPool&              globalPool,
                       ScalMemoryPool&              hbmSharedPool);
    ~NativeComputeGraph();

    NativeComputeGraph(const NativeComputeGraph&)            = delete;
    NativeComputeGraph& operator=(const NativeComputeGraph&) = delete;

    synStatus beginCapture();
    synStatus endCapture();
    synStatus replay();
    synStatus beginReplay();
    synStatus replaySegment(size_t                    segmentIndex,
                            const ScalLongSyncObject* dependency,
                            bool                      submit,
                            ScalLongSyncObject&       completion);
    synStatus close();
    synStatus preparePlan(const uint32_t* consumerSegments, uint64_t count, uint32_t nicLongSoIndex,
                          synNativeGraphExchangeReplay callback, void* context, const uint32_t* producers = nullptr);
    synStatus prepareSegmentedPlan(const uint32_t* producerSegments, const uint32_t* consumerSegments,
                                   uint64_t count, uint64_t prefixCollectives, uint32_t prefixComputes,
                                   uint32_t nicLongSoIndex,
                                   synNativeGraphExchangeReplay callback, void* context);
    synStatus prepareBoundedVariants(const uint32_t* minimum, uint64_t segments,
                                     const uint32_t* producers, const uint32_t* consumers,
                                     uint64_t collectives, uint32_t limit);
    synStatus selectBoundedVariant(uint32_t bound);
    synStatus replayPlan(synNativeGraphSyncInfo& completion, uint64_t* statistics, uint64_t count);
    synStatus replayPlanPrefix(synNativeGraphSyncInfo& completion, uint64_t* statistics, uint64_t count);
    synStatus replayPlanFinish(synNativeGraphSyncInfo& completion, uint64_t* statistics, uint64_t count);

    CompiledSegment* addPendingSegment(uint64_t                      globalSize,
                                       uint64_t                      arcSize,
                                       uint64_t                      recipeSequence,
                                       const synLaunchTensorInfoExt* launchTensors,
                                       uint32_t                      launchTensorCount,
                                       uint64_t                      workspaceAddress,
                                       uint64_t                      workspaceSize,
                                       synStatus&                    status);
    void commitSegment(CompiledSegment& segment, NativeScalCommandBuffer&& commands);
    void invalidate();

    State state() const { return m_state; }
    uint64_t replayCount() const { return m_replayCount; }
    size_t segmentCount() const { return m_segments.size(); }
    uint64_t globalProgramBytes() const { return m_globalProgramBytes; }
    uint64_t arcProgramBytes() const { return m_arcProgramBytes; }
    uint64_t workspaceAllocationBytes() const { return m_workspaceAllocationBytes; }
    const ScalLongSyncObject& lastCompletion() const { return m_lastCompletion; }

private:
    struct JointPlan
    {
        NativeComputeProgram program;
        std::vector<NativeComputeProgram> boundedPrograms;
        uint32_t selectedBound = UINT32_MAX;
        NativeComputeProgram prefixProgram;
        NativeComputeProgram suffixProgram;
        uint32_t nicLongSoIndex = 0;
        synNativeGraphExchangeReplay callback = nullptr;
        void* context = nullptr;
        std::vector<synNativeGraphSyncInfo> producers, completions;
        std::vector<CommandSubmissionData> submissionScratch;
        std::array<uint64_t, 12> statistics {};
        size_t collectiveSplit = 0;
        size_t suffixDependencyBegin = 0;
        ScalLongSyncObject segmentedBase = LongSoEmpty;
    };
    std::unique_ptr<JointPlan> m_jointPlan;
    ScalStreamComputeInterface& m_stream;
    ScalStreamBase&             m_streamBase;
    ScalMemoryPool&             m_globalPool;
    ScalMemoryPool&             m_hbmSharedPool;
    std::vector<CompiledSegment> m_segments;
    ScalLongSyncObject          m_lastCompletion = LongSoEmpty;
    State                       m_state          = State::CREATED;
    uint64_t                    m_replayCount    = 0;
    uint64_t                    m_globalProgramBytes = 0;
    uint64_t                    m_arcProgramBytes    = 0;
    size_t                      m_nextReplaySegment = 0;
    bool                        m_replayActive      = false;
    uint64_t                    m_workspaceAllocationBytes = 0;
};
