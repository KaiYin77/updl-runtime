/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#include "updl/updl_kernels_support.h"
#include "updl/updl_operator.h"

#include <stdint.h>
#include <math.h>

/**
 * @brief Softmax activation function for int16 quantized tensors
 * 
 * Implements numerically stable softmax using floating-point arithmetic internally:
 * softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
 * 
 * The function:
 * 1. Dequantizes int16 inputs to float32 using: real_value = input_scale * (quantized - input_zp)
 * 2. Computes softmax using standard exp() function
 * 3. Quantizes float32 outputs back to int16 using: quantized = real_value / output_scale + output_zp
 * 
 * This approach provides better numerical accuracy compared to fixed-point arithmetic,
 * as softmax is very sensitive to quantization errors.
 * 
 * @param[in]  input         Input tensor (int16, quantized)
 * @param[out] output        Output tensor (int16, quantized)  
 * @param[in]  size          Number of elements in the tensor
 * @param[in]  input_scale   Input dequantization scale
 * @param[in]  input_zp      Input zero point
 * @param[in]  output_scale  Output quantization scale
 * @param[in]  output_zp     Output zero point
 * @return 0 on success, non-zero on error
 */
uint8_t updl_softmax_s16(int16_t *input, int16_t *output, uint32_t size,
                         float input_scale, int16_t input_zp,
                         float output_scale, int16_t output_zp)
{
    if (!input || !output || size == 0) {
        updl_Error("Softmax: Invalid parameters (input=%p, output=%p, size=%u)\n", 
                   (void*)input, (void*)output, size);
        return 1; // Invalid parameters
    }
    
    if (size > 256) {
        updl_Error("Softmax: Size %u exceeds maximum supported size of 256\n", size);
        return 1; // Size exceeds buffer limit
    }

    // Temporary buffer for floating-point calculations
    float exp_values[256]; // Max size is 256 as validated above
    
    // Step 1: Dequantize inputs to real values and find maximum for numerical stability
    // Dequantization formula: real_value = input_scale * (quantized_value - input_zp)
    float max_val = input_scale * ((float)input[0] - (float)input_zp);
    for (uint32_t i = 1; i < size; i++) {
        float val = input_scale * ((float)input[i] - (float)input_zp);
        if (val > max_val) {
            max_val = val;
        }
    }

    // Step 2: Compute exp(x_i - max) and sum using floating-point arithmetic
    float sum = 0.0f;
    for (uint32_t i = 0; i < size; i++) {
        float x = input_scale * ((float)input[i] - (float)input_zp);
        float shifted = x - max_val; // Subtract max for numerical stability
        
        // Compute exp using standard math library
        exp_values[i] = expf(shifted);
        sum += exp_values[i];
    }

    // Step 3: Handle edge case
    if (sum == 0.0f || !isfinite(sum)) {
        // Edge case: set uniform distribution
        // For uniform distribution, each probability = 1/size
        float uniform_prob = 1.0f / (float)size;
        int32_t uniform_quantized = (int32_t)(uniform_prob / output_scale + 0.5f) + output_zp;
        for (uint32_t i = 0; i < size; i++) {
            output[i] = updl_clamp_s16(uniform_quantized);
        }
        return 0;
    }

    // Step 4: Compute softmax probabilities and quantize to int16
    // Softmax produces probabilities in [0, 1] range
    for (uint32_t i = 0; i < size; i++) {
        // Compute probability: exp(x_i - max) / sum
        float prob = exp_values[i] / sum;
        
        // Quantize to int16 using output scale and zero point
        // Quantization formula: quantized_value = round(real_value / output_scale) + output_zp
        int32_t quantized_val = (int32_t)(prob / output_scale + 0.5f) + output_zp;
        output[i] = updl_clamp_s16(quantized_val);
    }
    
    return 0; // Success
}