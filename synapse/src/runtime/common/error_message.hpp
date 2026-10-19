#pragma once

#include "synapse_common_types.h"

namespace synapse::api_error
{
void        reset() noexcept;
void        set(synStatus status, const char* function = nullptr, const char* detail = nullptr) noexcept;
synStatus   getLastError() noexcept;
const char* getErrorString(synStatus status) noexcept;
const char* getLastErrorMessage() noexcept;
}  // namespace synapse::api_error
