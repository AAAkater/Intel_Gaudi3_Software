// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <limits>

namespace NativeProgramWindow
{
constexpr uint64_t alignment = 1ULL << 13;
constexpr uint64_t window = 1ULL << 32;

inline bool allocationSize(uint64_t payload, bool padded, uint64_t& bytes)
{
    if (payload == 0 || payload >= window) return false;
    // If an aligned payload crosses a window, the distance to the next
    // window is at most the payload. Twice its size plus alignment
    // therefore suffices even when a replacement allocation moves.
    bytes = (padded ? 2 * payload : payload) + alignment - 1;
    return true;
}

inline bool selectAddress(uint64_t raw, uint64_t bytes, uint64_t payload, uint64_t& address)
{
    if (payload == 0 || payload >= window || bytes < payload ||
        raw > std::numeric_limits<uint64_t>::max() - (bytes - 1)) return false;
    uint64_t shift = (alignment - (raw & (alignment - 1))) & (alignment - 1);
    if (shift > bytes - payload) return false;
    const uint64_t remaining = window - ((raw + shift) & (window - 1));
    // Match the normal compute arena's exclusive-end check, including
    // a final instruction whose subsequent PC could wrap at the boundary.
    if (payload >= remaining) shift += remaining;
    if (shift > bytes - payload) return false;
    address = raw + shift;
    return true;
}
}  // namespace NativeProgramWindow
