#include "updl/updl_nn_utils_udl.h"

#include "updl/updl_kernels_support.h"
#include "updl/updl_utility.h"

int64_t updl_udl_bound40(int64_t input) {
#ifdef UPDL_UDL_SHIFT_ONLY_MODE
    const int64_t upper = 0x0000007FFFFFFFFFLL;
    const int64_t lower = -0x0000008000000000LL;
    if (input > upper) {
        return upper;
    }
    if (input < lower) {
        return lower;
    }
    return input;
#else
    return input;
#endif
}

int16_t updl_udl_bound16(int64_t input) {
#ifdef UPDL_UDL_SHIFT_ONLY_MODE
    if (input > INT16_MAX) {
        return INT16_MAX;
    }
    if (input < INT16_MIN) {
        return INT16_MIN;
    }
    return (int16_t)input;
#else
    if (input > INT16_MAX) {
        return INT16_MAX;
    }
    if (input < INT16_MIN) {
        return INT16_MIN;
    }
    return (int16_t)input;
#endif
}

int64_t updl_udl_right_shift_round64(int64_t input, uint8_t rshift) {
    if (rshift == 0) {
        return input;
    }
    int64_t carry = (input >> (rshift - 1)) & 0x1;
    return (input >> rshift) + carry;
}

int64_t updl_udl_activation64(int64_t input, atype_t activation) {
    switch (activation) {
        case Atype_relu:
            return (input > 0) ? input : 0;
        case Atype_leakyrelu:
            return input;
        case Atype_linear:
        case Atype_none:
        default:
            return input;
    }
}

int16_t updl_udl_finalize(int64_t sum,
                          int16_t bias_val,
                          atype_t activation,
                          int32_t eff_multiplier,
                          int16_t eff_shift,
                          int16_t output_zp,
                          int32_t eff_bias_multiplier,
                          int16_t eff_bias_shift) {
#ifdef UPDL_UDL_SHIFT_ONLY_MODE
    (void)eff_multiplier;
    (void)output_zp;
    (void)eff_bias_multiplier;

    sum = updl_udl_bound40(sum);

    int64_t bias64 = (int64_t)bias_val;
    if (bias_val != 0 && eff_bias_shift != 0) {
        if (eff_bias_shift < 0) {
            bias64 <<= (-eff_bias_shift);
        } else {
            updl_Warning("Positive eff_bias_shift (%d) not supported in hardware mode; truncating\n",
                         eff_bias_shift);
            bias64 >>= eff_bias_shift;
        }
    }
    sum += bias64;
    sum = updl_udl_bound40(sum);

    sum = updl_udl_activation64(sum, activation);

    if (eff_shift < 0) {
        updl_Warning("Negative eff_shift (%d) not supported in hardware mode; clamping to 0\n",
                     eff_shift);
        eff_shift = 0;
    }

    uint8_t pos_shift = (uint8_t)eff_shift;
    uint8_t neg_shift = pos_shift;
    if (activation == Atype_leakyrelu) {
        int16_t candidate = eff_shift + 3;
        if (candidate < 0) {
            updl_Warning("Invalid leaky ReLU negative shift (%d); clamping to 0\n", candidate);
            candidate = 0;
        }
        neg_shift = (uint8_t)candidate;
    }

    uint8_t selected_shift = pos_shift;
    if (activation == Atype_leakyrelu && sum < 0) {
        selected_shift = neg_shift;
    }
    sum = updl_udl_right_shift_round64(sum, selected_shift);

    return updl_udl_bound16(sum);
#else
    int64_t scaled_bias = updl_scale_bias(bias_val, eff_bias_multiplier, eff_bias_shift);
    sum += scaled_bias;
    return updl_quantize_pipeline(sum, activation, eff_multiplier, eff_shift, output_zp);
#endif
}
