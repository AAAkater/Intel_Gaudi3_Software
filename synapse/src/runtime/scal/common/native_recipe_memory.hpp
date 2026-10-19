#pragma once

#include "runtime/scal/common/infra/scal_includes.hpp"
#include "synapse_common_types.h"

#include <cstdint>

class ScalMemoryPool;

// Device program storage owned by a native graph instance. Unlike the normal
// per-stream N-buffer allocator, these allocations are never recycled between
// launches. They are released only after the graph's last completion target has
// finished, so captured ECB addresses remain valid for every replay.
class NativeRecipeMemory
{
public:
    NativeRecipeMemory(ScalMemoryPool& globalPool, ScalMemoryPool& hbmSharedPool);
    ~NativeRecipeMemory();

    NativeRecipeMemory(const NativeRecipeMemory&)            = delete;
    NativeRecipeMemory& operator=(const NativeRecipeMemory&) = delete;

    synStatus initialize(uint64_t globalSize, uint64_t arcSize, uint64_t workspaceSize = 0);
    synStatus close();

    bool initialized() const { return m_initialized; }
    bool closed() const { return m_closed; }

    uint64_t globalDeviceAddress() const { return m_globalDeviceAddress; }
    uint64_t arcDeviceAddress() const { return m_arcDeviceAddress; }
    uint32_t arcCoreAddress() const { return m_arcCoreAddress; }
    uint64_t globalAllocationBytes() const { return m_globalAllocationBytes; }
    uint64_t arcAllocationBytes() const { return m_arcAllocationBytes; }
    uint64_t workspaceDeviceAddress() const { return m_workspaceDeviceAddress; }
    uint64_t workspaceAllocationBytes() const { return m_workspaceAllocationBytes; }

private:
    ScalMemoryPool& m_globalPool;
    ScalMemoryPool& m_hbmSharedPool;

    scal_buffer_handle_t m_globalBuffer = nullptr;
    scal_buffer_handle_t m_arcBuffer    = nullptr;
    uint64_t             m_globalDeviceAddress = 0;
    uint64_t             m_arcDeviceAddress    = 0;
    uint32_t             m_arcCoreAddress      = 0;
    uint64_t             m_globalAllocationBytes = 0;
    uint64_t             m_arcAllocationBytes    = 0;
    bool                 m_initialized         = false;
    bool                 m_closed              = false;
    scal_buffer_handle_t m_workspaceBuffer = nullptr;
    uint64_t             m_workspaceDeviceAddress = 0;
    uint64_t             m_workspaceAllocationBytes = 0;
};
