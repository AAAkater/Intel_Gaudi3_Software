#pragma once

#include "stream_base_scal.hpp"
#include "utils.h"

#include "runtime/scal/common/recipe_launcher/mem_mgrs.hpp"
#include "runtime/scal/common/recipe_launcher/recipe_launcher.hpp"

#include <array>

struct ScalEvent;
struct PreAllocatedStreamMemoryAll;
class DevMemoryAllocInterface;
class DynamicRecipe;
class ScalStreamComputeInterface;
class ScalMemoryPool;
class NativeComputeGraph;

class QueueComputeScal : public QueueBaseScalCommon
{
public:
    friend class ScalStreamTest;
    friend class SynScalLaunchDummyRecipe;

    QueueComputeScal(const BasicQueueInfo&           rBasicQueueInfo,
                     ScalStreamComputeInterface*     pScalStream,
                     const ComputeCompoundResources* pComputeResources,
                     synDeviceType                   deviceType,
                     const ScalDevSpecificInfo&      rDevSpecificInfo,
                     DevMemoryAllocInterface&        devMemoryAlloc,
                     ScalMemoryPool&                 globalPool,
                     ScalMemoryPool&                 hbmSharedPool);

    virtual ~QueueComputeScal();

    virtual synStatus getMappedMemorySize(uint64_t& mappedMemorySize) const override;

    virtual synStatus eventRecord(EventInterface& rEventInterface, synStreamHandle streamHandle) override;

    virtual synStatus
    eventWait(const EventInterface& rEventInterface, const unsigned int flags, synStreamHandle streamHandle) override;

    virtual synStatus query() override;

    virtual synStatus synchronize(synStreamHandle streamHandle, bool isUserRequest) override;

    synStatus init(PreAllocatedStreamMemoryAll& preAllocatedInfo);

    virtual synStatus memcopy(internalMemcopyParams& memcpyParams,
                              const internalDmaDir   direction,
                              bool                   isUserRequest,
                              QueueInterface*        pPreviousStream,
                              const uint64_t         overrideMemsetVal,
                              bool                   inspectCopiedContent,
                              SpRecipeProgramBuffer* pRecipeProgramBuffer,
                              uint8_t                apiId) override
    {
        return synFail;
    }

    virtual synStatus launch(const synLaunchTensorInfoExt* launchTensorsInfo,
                             uint32_t                      launchTensorsAmount,
                             uint64_t                      workspaceAddress,
                             InternalRecipeHandle*         pRecipeHandle,
                             uint64_t                      assertAsyncMappedAddress,
                             uint32_t                      flags,
                             EventWithMappedTensorDB&      events,
                             uint8_t                       apiId) override;

    virtual void notifyRecipeRemoval(InternalRecipeHandle& rRecipeHandle);

    virtual void    notifyAllRecipeRemoval();
    static uint64_t getHbmGlblMaxRecipeSize(uint64_t maxGlbHbmMemorySize);
    static uint64_t getHbmSharedMaxRecipeSize(uint64_t maxArcHbmMemorySize);

    virtual void dfaUniqStreamInfo(bool               oldestRecipeOnly,
                                   uint64_t           currentLongSo,
                                   bool               dumpRecipe,
                                   const std::string& callerMsg) override;

    virtual synStatus getDynamicShapesTensorInfoArray(synRecipeHandle             recipeHandle,
                                                      std::vector<tensor_info_t>& tensorInfoArray) const;

    std::unique_ptr<NativeComputeGraph> createNativeComputeGraph();
    synStatus beginNativeComputeGraphCapture(NativeComputeGraph& graph);
    synStatus endNativeComputeGraphCapture(NativeComputeGraph& graph);
    synStatus abortNativeComputeGraphCapture(NativeComputeGraph& graph);
    synStatus replayNativeComputeGraph(NativeComputeGraph& graph);
    synStatus beginNativeComputeGraphReplay(NativeComputeGraph& graph);
    synStatus prepareNativeComputePlan(NativeComputeGraph& graph, const uint32_t* consumers,
        uint64_t count, uint32_t nicIndex, synNativeGraphExchangeReplay callback, void* context, const uint32_t* producers = nullptr);
    synStatus prepareSegmentedNativeComputePlan(NativeComputeGraph& graph, const uint32_t* producers,
        const uint32_t* consumers, uint64_t count, uint64_t prefixCollectives, uint32_t prefixComputes, uint32_t nicIndex,
        synNativeGraphExchangeReplay callback, void* context);
    synStatus replayNativeComputePlan(NativeComputeGraph& graph, synNativeGraphSyncInfo& completion,
        uint64_t* statistics, uint64_t count);
    synStatus replayNativeComputePlanPrefix(NativeComputeGraph& graph, synNativeGraphSyncInfo& completion,
        uint64_t* statistics, uint64_t count);
    synStatus replayNativeComputePlanFinish(NativeComputeGraph& graph, synNativeGraphSyncInfo& completion,
        uint64_t* statistics, uint64_t count);
    synStatus replayNativeComputeGraphSegment(NativeComputeGraph&       graph,
                                              size_t                    segmentIndex,
                                              const ScalLongSyncObject* dependency,
                                              bool                      submit,
                                              ScalLongSyncObject&       completion);

private:
    synStatus initMemMgrs();
    synStatus      launch(const LaunchInfo& launchInfo);
    DynamicRecipe* getDynamicShapeProcessor(const LaunchInfo& launchInfo);

    void updateSfgEventsScalStream(EventWithMappedTensorDB& events);
    virtual std::set<ScalStreamCopyInterface*> dfaGetQueueScalStreams() override;


    const synDeviceType        m_deviceType;
    const ScalDevSpecificInfo& m_rDevSpecificInfo;

    MemMgrs       m_memMgrs;
    LaunchTracker m_launchTracker;

    DevMemoryAllocInterface& m_devMemoryAlloc;
    ScalMemoryPool&          m_globalPool;
    ScalMemoryPool&          m_hbmSharedPool;

    uint64_t m_runningId = 0;

    uint64_t m_hbmGlblAddr = 0;
    uint64_t m_hbmGlblSize = 0;

    uint32_t m_arcHbmAddrCore = 0;
    uint64_t m_arcHbmAddrDev  = 0;
    uint64_t m_arcHbmSize     = 0;

    bool m_initDone = false;

    std::unordered_map<uint64_t, std::unique_ptr<DynamicRecipe>> m_dynamicShapeProcessor;

    const ComputeCompoundResources m_computeResources;
    NativeComputeGraph*             m_activeNativeGraph = nullptr;
};
