#pragma once

#include "runtime/scal/common/native_compute_graph.hpp"
#include "runtime/common/streams/stream_job.hpp"
#include "infra/containers/slot_map.hpp"

#include <memory>
#include <mutex>

class QueueComputeScal;
class Stream;

struct InternalNativeComputeGraphHandle
{
    QueueComputeScal*                   queue        = nullptr;
    synStreamHandle                     streamHandle = nullptr;
    std::unique_ptr<NativeComputeGraph> graph;
    SlotMapItemSptr<Stream>              stream;
    std::unique_ptr<StreamJob>           preparedReplayJob;
    std::unique_ptr<StreamJob>           preparedPrefixReplayJob;
    std::unique_ptr<StreamJob>           preparedFinishReplayJob;
    std::mutex                          replayMutex;
    synNativeGraphSyncInfo*              replayCompletion = nullptr;
    uint64_t*                           replayStatistics = nullptr;
    uint64_t                            replayStatisticCount = 0;
};
