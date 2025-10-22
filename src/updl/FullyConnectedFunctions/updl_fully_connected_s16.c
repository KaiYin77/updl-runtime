/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#include "updl/updl_kernels.h"
#include "updl/updl_kernels_support.h"
#include "updl/updl_operator.h"

#include <assert.h>

/**
 * @brief Basic s16 fully-connected layer function
 * 
 * Following CMSIS-NN approach:
 * - Vector by matrix multiplication using transposed weights
 * - Single batch processing (batch_size = 1 always at runtime)
 * - Optimized for int16 activations with int16 weights
 */
uint8_t updl_fully_connected_s16(
    int16_t *input, int16_t *output, int16_t *weights, int16_t *bias,
    uint32_t input_features, uint32_t output_features,
    atype_t activation,
    int32_t eff_multiplier, int16_t eff_shift,
    int16_t input_zp, int16_t weight_zp, int16_t output_zp,
    int32_t eff_bias_multiplier, int16_t eff_bias_shift)
{

    for (uint32_t out_feat = 0; out_feat < output_features; out_feat++) {
        int64_t sum = 0;
        const int16_t *weight_row = weights + out_feat * input_features;

        for (uint32_t in_feat = 0; in_feat < input_features; in_feat++) {
            int32_t inp = (int32_t)input[in_feat] - (int32_t)input_zp;
            int32_t wgt = (int32_t)weight_row[in_feat] - (int32_t)weight_zp;
            sum += (int64_t)inp * (int64_t)wgt;
        }

        // Add bias with proper scaling
        if (bias) {
            int64_t scaled_bias = updl_scale_bias(bias[out_feat], eff_bias_multiplier, eff_bias_shift);
            sum += scaled_bias;
        }
        
        // Check if we need dynamic scaling
        if (sum > INT32_MAX || sum < INT32_MIN) {
            // Use original path with dynamic scaling for overflow cases
            int16_t dynamic_shift = 0;
            int32_t raw_sum = updl_clamp_s32_with_scaling(sum, &dynamic_shift);
            int32_t activated = updl_activation(raw_sum, activation);
            int16_t adjusted_shift = eff_shift - dynamic_shift;
            int32_t quantized = updl_requantize(activated, eff_multiplier, adjusted_shift);
            quantized += output_zp;
            output[out_feat] = updl_clamp_s16(quantized);
        } else {
            // Use optimized pipeline for normal cases (most common)
            output[out_feat] = updl_quantize_pipeline(sum, activation, eff_multiplier, eff_shift, output_zp);
        }
    }
    return 0;
}


