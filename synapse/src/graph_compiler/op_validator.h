#pragma once
#ifndef _OP_VALIDATOR_H_
#define _OP_VALIDATOR_H_

#include "defs.h"
#include <string_view>

namespace gc::ops
{
struct OpInfo;
class TensorValidationContext;
class OpValidationContext;

enum class ValidationResult
{
    SUCCESS,
    GUID_NOT_FOUND,
    INCOMPATIBLE_INPUT_COUNT,
    INCOMPATIBLE_INPUT_DIMENSION,
    INCOMPATIBLE_INPUT_SIZE,
    INCOMPATIBLE_OUTPUT_COUNT,
    INCOMPATIBLE_OUTPUT_DIMENSION,
    INCOMPATIBLE_OUTPUT_SIZE,
    INCOMPATIBLE_DATA_TYPE,
    GUID_MISSING_DYNAMIC_SUPPORT,
    GUID_HAS_NO_SHAPE_TENSOR_INPUT,
    GUID_HAS_NO_H2D_TENSOR_INPUT,
    DS_NDIMS_NOT_SUPPORTED,
    FAILURE
};

enum class ValidationVersion
{
    V1    = 1,
    V2    = 2,
    QUERY = 3
};

class OpValidator
{
public:
    struct ScopedSkipValidation
    {
        ScopedSkipValidation();
        ~ScopedSkipValidation();
        ScopedSkipValidation(const ScopedSkipValidation&) = delete;
        ScopedSkipValidation(ScopedSkipValidation&&)      = delete;
        ScopedSkipValidation& operator=(const ScopedSkipValidation&) = delete;
        ScopedSkipValidation& operator=(ScopedSkipValidation&&) = delete;
        bool                  m_prevMode;
    };

    ~OpValidator()                        = default;
    OpValidator()                         = delete;
    OpValidator(const OpValidator& other) = delete;
    OpValidator(OpValidator&& other)      = delete;
    OpValidator& operator=(const OpValidator& other) = delete;
    OpValidator& operator=(OpValidator&& other) = delete;

    /**
     * @brief Validate input (guid,context,device) and return status status accordingly
     */
    static ValidationResult validateOp(const std::string&       guid,
                                       const OpValidationContext& ovc,
                                       synDeviceType             deviceType,
                                       ValidationVersion         version = ValidationVersion::V1);

protected:
    OpValidator(const std::string& guid, const OpValidationContext& ctx, synDeviceType deviceType);
    typedef ValidationResult (*ValidationFunction)(const OpInfo& gcOpInfo, const OpValidationContext& opValCtx);

    /**
     * @brief Enable mode setter
     *
     * @param mode
     */
    static void setEnabled(bool mode);

    /**
     * @brief Enable mode getter
     *
     * @return mode
     */
    static bool enabled();

    /**
     * @brief Get the validated op context
     *
     */
    const auto& getTestedOpCtx() const { return m_opCtx; }

    /**
     * @brief Get guid string
     *
     * @return const auto&
     */
    const auto& getGuid() const { return m_guid; }

    /**
     * @brief Get the validated device type
     *
     */
    const auto getDevice() const { return m_device; }

    /**
     * @brief Test whether tested type is included in input mask or not
     *
     */
    static bool isSupportedDatatype(synDataType testedType, unsigned supportedTypesMask);

    /**
     * @brief Run all validations and return a retcode accordingly
     *
     */
    ValidationResult runValidations(ValidationVersion version) const;
    ValidationResult runQueryValidations() const;

    // Validations
    static ValidationResult validateCustom(const OpInfo& gcOpInfo, const OpValidationContext& opValCtx);
    static ValidationResult validateNumOperands(const OpInfo& gcOpInfo, const OpValidationContext& opValCtx);
    static ValidationResult validateRank(const OpInfo& gcOpInfo, const OpValidationContext& opValCtx);
    static ValidationResult validateDatatypes(const OpInfo& gcOpInfo, const OpValidationContext& opValCtx);
    static ValidationResult validateDynShapesSupport(const OpInfo& gcOpInfo, const OpValidationContext& opValCtx);
    static ValidationResult validateShapeTensorInputs(const OpInfo& gcOpInfo, const OpValidationContext& opValCtx);
    static ValidationResult validateH2DTensorInputs(const OpInfo& gcOpInfo, const OpValidationContext& opValCtx);
    static ValidationResult validateDynShapesNDimsSupport(const OpInfo& gcOpInfo,
                                                          const OpValidationContext& opValCtx);

    const std::string                                                    m_guid;
    const OpValidationContext&                                           m_opCtx;
    const synDeviceType                                                  m_device;
    static const std::vector<std::pair<ValidationFunction, std::string_view>> s_validationsV1;
    static const std::vector<std::pair<ValidationFunction, std::string_view>> s_validationsV2;
    static bool s_enabled;  // used to internally disable validation when necessary as a temporary workaround
};

using ScopedSkipValidation = OpValidator::ScopedSkipValidation;

/**
 * @brief Internal representation for synapse operation context
 */
class OpValidationContext
{
public:
    OpValidationContext()                                 = default;
    OpValidationContext(const OpValidationContext& other) = default;
    OpValidationContext(OpValidationContext&& other)      = default;
    OpValidationContext& operator=(const OpValidationContext& other) = default;
    OpValidationContext& operator=(OpValidationContext&& other) = default;
    ~OpValidationContext()                                      = default;

    // Non-const and const operand context getters
    const std::vector<TensorValidationContext>& getInputs() const;
    const std::vector<TensorValidationContext>& getOutputs() const;
    std::vector<TensorValidationContext>&       getInputs();
    std::vector<TensorValidationContext>&       getOutputs();

    bool supportsDynamicShapes() const { return m_supportsDynamicShapes; }
    bool requiresShapeTensor() const { return m_requiresShapeTensor; }
    bool requiresH2DTensor() const { return m_requiresH2DTensor; }
    unsigned queryBitMap() const { return m_queryBitMap; }
    unsigned* resultBitMap() const { return m_resultBitMap; }

    void setSupportsDynamicShapes(bool value) { m_supportsDynamicShapes = value; }
    void setRequiresShapeTensor(bool value) { m_requiresShapeTensor = value; }
    void setRequiresH2DTensor(bool value) { m_requiresH2DTensor = value; }
    void setQuery(unsigned queryBitMap, unsigned* resultBitMap)
    {
        m_queryBitMap  = queryBitMap;
        m_resultBitMap = resultBitMap;
    }

    // Whether context object is empty or not
    bool empty() const;


private:
    // Non-const operands context getter
    std::vector<TensorValidationContext>& getOperandsCtx(bool isInput);

    std::vector<TensorValidationContext> m_input;
    std::vector<TensorValidationContext> m_output;
    bool                                 m_supportsDynamicShapes = false;
    bool                                 m_requiresShapeTensor   = false;
    bool                                 m_requiresH2DTensor     = false;
    unsigned                             m_queryBitMap           = 0;
    unsigned*                            m_resultBitMap          = nullptr;
};

/**
 * @brief Contains tensor  attributes required for validation
 */
class TensorValidationContext
{
public:
    TensorValidationContext(unsigned rank, synDataType dtype, synTensorType tensorType)
    : m_rank(rank), m_dtype(dtype), m_tensorType(tensorType)
    {
    }

    TensorValidationContext() : m_rank(0), m_dtype(syn_type_na), m_tensorType(TENSOR_TYPE_INVALID) {}

    ~TensorValidationContext()                                    = default;
    TensorValidationContext(const TensorValidationContext& other) = default;
    TensorValidationContext(TensorValidationContext&& other)      = default;
    TensorValidationContext& operator=(const TensorValidationContext& other) = default;
    TensorValidationContext& operator=(TensorValidationContext&& other) = default;

    unsigned      getRank() const { return m_rank; }
    synDataType   getDatatype() const { return m_dtype; }
    synTensorType getTensorType() const { return m_tensorType; }
    bool          isDataTensor() const { return getTensorType() == DATA_TENSOR; }
    bool empty() const { return (m_rank == 0 && m_dtype == syn_type_na && m_tensorType == TENSOR_TYPE_INVALID); }

private:
    unsigned      m_rank;
    synDataType   m_dtype;
    synTensorType m_tensorType;
};

}  // namespace gc::ops

#endif  // _OP_VALIDATOR_H_
