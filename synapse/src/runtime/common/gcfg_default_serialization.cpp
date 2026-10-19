// Keep default-value serialization instantiated with the complete toString
// overload set. In particular a string must never convert to SizeParam.
#include "hl_gcfg/hlgcfg_item.hpp"

template std::string hl_gcfg::GcfgItemImpl<std::string, std::string>::getDefaultValuesStr() const;
template std::string hl_gcfg::GcfgItemImpl<int64_t, int64_t>::getDefaultValuesStr() const;
template std::string hl_gcfg::GcfgItemImpl<uint64_t, uint64_t>::getDefaultValuesStr() const;
template std::string hl_gcfg::GcfgItemImpl<bool, bool>::getDefaultValuesStr() const;
template std::string hl_gcfg::GcfgItemImpl<float, float>::getDefaultValuesStr() const;
