// SPDX-License-Identifier: MIT
// Preserve the logger v1_0 data types and forward to its current named-module overload.
#include "hl_logger/hllog.hpp"
#include "hl_logger/impl/hllog_internal_api.hpp"

#include <string_view>
#include <utility>

namespace hl_logger::internal::v1_0
{
HLLOG_API ResourceGuard registerLazyLogsHandler(LazyLogsHandler handler, std::string_view moduleName);

ResourceGuard registerLazyLogsHandler(LazyLogsHandler handler)
{
    return registerLazyLogsHandler(std::move(handler), "synapse-recovered");
}
}  // namespace hl_logger::internal::v1_0
