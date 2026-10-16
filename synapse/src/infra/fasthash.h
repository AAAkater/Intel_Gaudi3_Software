#pragma once

#include <string>
#include <cstdint>

uint64_t fasthash(const std::string& s);
uint64_t fasthash(const void* ptr, size_t len);
