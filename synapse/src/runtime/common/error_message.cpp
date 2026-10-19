#include "error_message.hpp"

#include <string>

namespace synapse::api_error
{
namespace
{
struct ErrorState
{
    synStatus   status = synSuccess;
    std::string message;
};

thread_local ErrorState state;
}  // namespace

const char* getErrorString(synStatus status) noexcept
{
    switch (status)
    {
        case synSuccess:                      return "Success";
        case synInvalidArgument:              return "Invalid argument";
        case synCbFull:                       return "Command buffer is full";
        case synOutOfHostMemory:              return "Out of host memory";
        case synOutOfDeviceMemory:            return "Out of device memory";
        case synObjectAlreadyInitialized:     return "Object already initialized";
        case synObjectNotInitialized:         return "Object not initialized";
        case synCommandSubmissionFailure:     return "Command submission failed";
        case synNoDeviceFound:                return "Device not found";
        case synDeviceTypeMismatch:           return "Device type mismatch";
        case synFailedToInitializeCb:         return "Failed to initialize command buffer";
        case synFailedToFreeCb:               return "Failed to free command buffer";
        case synFailedToMapCb:                return "Failed to map command buffer";
        case synFailedToUnmapCb:              return "Failed to unmap command buffer";
        case synFailedToAllocateDeviceMemory: return "Failed to allocate device memory";
        case synFailedToFreeDeviceMemory:     return "Failed to free device memory";
        case synFailedNotEnoughDevicesFound:  return "Not enough devices found";
        case synDeviceReset:                  return "Device reset";
        case synUnsupported:                  return "Unsupported operation";
        case synWrongParamsFile:              return "Invalid recipe file";
        case synDeviceAlreadyAcquired:        return "Device already acquired";
        case synNameIsAlreadyUsed:            return "Name is already used";
        case synBusy:                         return "Device is busy";
        case synAllResourcesTaken:            return "All resources are taken";
        case synUnavailable:                  return "Resource unavailable";
        case synInvalidTensorDimensions:      return "Invalid tensor dimensions";
        case synFail:                         return "Generic failure";
        case synOutOfResources:               return "Out of resources";
        case synUninitialized:                return "Synapse is uninitialized";
        case synAlreadyInitialized:           return "Synapse is already initialized";
        case synFailedSectionValidation:      return "Section validation failed";
        case synSynapseTerminated:            return "Synapse terminated";
        case synAssertAsync:                  return "Asynchronous assertion";
        case synInvalidEventHandle:           return "Invalid event handle";
        case synMappingNotFound:              return "Mapping not found";
        case synFailedDynamicPatching:         return "Dynamic patching failed";
        case synFailedStaticPatching:          return "Static patching failed";
        case synFailedToSubmitWorkload:        return "Failed to submit workload";
        case synInvalidSectionsDefinition:    return "Invalid sections definition";
        case synInvalidTensorProperties:      return "Invalid tensor properties";
        case synFailHccl:                     return "HCCL failure";
        case synFailedToCollectTime:           return "Failed to collect time";
        case synTimeout:                       return "Timeout";
        case synResourceBadUsage:              return "Invalid resource usage";
    }
    return "Unknown Synapse status";
}

void reset() noexcept
{
    state.status = synSuccess;
    state.message.clear();
}

void set(synStatus status, const char* function, const char* detail) noexcept
{
    state.status = status;
    state.message.clear();
    if (status == synSuccess) return;

    try
    {
        if (function != nullptr && function[0] != '\0')
        {
            state.message.append(function);
            state.message.append(": ");
        }
        if (detail != nullptr && detail[0] != '\0')
        {
            state.message.append(detail);
        }
        else
        {
            state.message.append(getErrorString(status));
        }
    }
    catch (...)
    {
        state.message.clear();
    }
}

synStatus getLastError() noexcept
{
    return state.status;
}

const char* getLastErrorMessage() noexcept
{
    return state.message.empty() ? nullptr : state.message.c_str();
}
}  // namespace synapse::api_error
