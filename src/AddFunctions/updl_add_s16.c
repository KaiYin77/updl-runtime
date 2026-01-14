/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#include "updl/updl_kernels.h"
#include "updl/updl_kernels_support.h"
#include "updl/updl_nn_utils_udl.h"
#include "updl/updl_operator.h"

#include <assert.h>

/**
 * @brief Basic s16 element-wise addition function
 *
 * Following CMSIS-NN approach:
 * - Element-wise addition with proper requantization
 * - Supports multiple input tensors for residual connections
 * - int16 activations with int16 outputs
 */
uint8_t updl_add_s16(
    int16_t **input_buffers, uint32_t num_inputs, int16_t *output,
    uint32_t tensor_size,
    atype_t activation,
    int32_t eff_multiplier, int16_t eff_shift,
    int16_t *input_zps, int16_t output_zp,
    float *input_scales, float output_scale)
{
    if (!input_buffers || !output || num_inputs == 0) {
        return 1;
    }

    for (uint32_t i = 0; i < tensor_size; i++) {
        int64_t sum = 0;

        // Sum all input tensors with dequantization
        for (uint32_t inp = 0; inp < num_inputs; inp++) {
            if (input_buffers[inp]) {
                // Dequantize input: real_value = (quantized - zero_point) * scale
                int32_t dequantized = ((int32_t)input_buffers[inp][i] - (int32_t)input_zps[inp]);

                // Scale to output scale domain: dequantized * (input_scale / output_scale)
                float scale_ratio = input_scales[inp] / output_scale;
                int64_t scaled_value = (int64_t)(dequantized * scale_ratio);

                sum += scaled_value;
                sum = updl_udl_bound40(sum);
            }
        }

        // Requantize and apply activation
        // For Add operations, no bias is typically used
        int16_t bias_val = 0;
        output[i] = updl_udl_finalize(
            sum, bias_val, activation, eff_multiplier, eff_shift, output_zp,
            1, 0); // No bias multiplier/shift for Add
    }

    return 0;
}

