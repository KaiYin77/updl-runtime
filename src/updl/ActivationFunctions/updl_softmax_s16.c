/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#include "updl/updl_kernels_support.h"
#include "updl/updl_operator.h"

#include <stdint.h>

/**
 * @brief Simple exponential approximation for softmax
 * Uses bit shifting for 2^x approximation: exp(x) ≈ 2^(x * 1.44)
 * Input: x_shifted value (after subtracting max)
 * Output: approximated exp(x) scaled for integer arithmetic
 */
static int32_t _exp_approx(int32_t x_shifted) {
    // For large negative values, return near zero
    if (x_shifted < -16384) return 1; // Very small positive value
    
    // For positive values, clamp to prevent overflow
    if (x_shifted > 16384) return INT32_MAX / 1024; // Large but manageable value
    
    // Use bit shifting approximation: 2^(x/4096) where x is in quantized units
    // This gives us reasonable dynamic range for softmax
    if (x_shifted >= 0) {
        // For positive values, use left shift (multiplication)
        int32_t shift_amount = x_shifted / 2048; // Scale down for reasonable shifts
        if (shift_amount > 15) shift_amount = 15; // Limit to prevent overflow
        return 1024 << shift_amount; // Base value with exponential scaling
    } else {
        // For negative values, use right shift (division)
        int32_t shift_amount = (-x_shifted) / 2048;
        if (shift_amount > 15) return 1; // Very small value
        return 1024 >> shift_amount;
    }
}

/**
 * @brief Softmax activation function for int16 quantized tensors
 * 
 * Implements numerically stable softmax: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
 * Uses improved fixed-point arithmetic optimized for embedded systems.
 * 
 * @param[in]  input         Input tensor (int16, quantized)
 * @param[out] output        Output tensor (int16, quantized)  
 * @param[in]  size          Number of elements in the tensor
 * @param[in]  eff_multiplier Requantization multiplier
 * @param[in]  eff_shift     Requantization shift
 * @param[in]  input_zp      Input zero point
 * @param[in]  output_zp     Output zero point
 * @return 0 on success, non-zero on error
 */
uint8_t updl_softmax_s16(int16_t *input, int16_t *output, uint32_t size,
                         int32_t eff_multiplier, int16_t eff_shift,
                         int16_t input_zp, int16_t output_zp)
{
    if (!input || !output || size == 0 || size > 256) {
        return 1; // Invalid parameters
    }

    // Step 1: Find maximum value for numerical stability
    int32_t max_val = (int32_t)input[0] - (int32_t)input_zp;
    for (uint32_t i = 1; i < size; i++) {
        int32_t val = (int32_t)input[i] - (int32_t)input_zp;
        if (val > max_val) {
            max_val = val;
        }
    }

    // Step 2: First pass - compute sum of exp(x_i - max)
    int64_t sum = 0;
    
    for (uint32_t i = 0; i < size; i++) {
        int32_t x = (int32_t)input[i] - (int32_t)input_zp;
        int32_t shifted = x - max_val; // x - max for numerical stability
        
        // Compute exp using simple approximation (Python: exp_x = np.exp(x_shifted))
        int32_t exp_val = _exp_approx(shifted);
        sum += exp_val;
    }

    // Step 3: Handle edge case
    if (sum == 0) {
        // Edge case: set uniform distribution
        int32_t uniform_val = 32767 / size;
        for (uint32_t i = 0; i < size; i++) {
            output[i] = updl_clamp_s16(uniform_val + output_zp);
        }
        return 0;
    }

    // Step 4: Second pass - compute final softmax values (Python: return exp_x / np.sum(exp_x))    
    for (uint32_t i = 0; i < size; i++) {
        int32_t x = (int32_t)input[i] - (int32_t)input_zp;
        int32_t shifted = x - max_val; // Recompute shifted value
        
        // Recompute exp value (same as first pass)
        int32_t exp_val = _exp_approx(shifted);
        
        // Compute probability as: prob_i = exp_i / sum (following Python reference)
        int64_t numerator = (int64_t)exp_val * 16384; // Scale to int16 max
        int32_t prob_val = (int32_t)(numerator / sum);
        
        // Convert to output quantization with zero point
        int32_t final_val = prob_val + output_zp;
        output[i] = updl_clamp_s16(final_val);
    }
    return 0; // Success
}