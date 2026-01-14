/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#include "updl/updl_kernels.h"
#include "updl/updl_kernels_support.h"
#include "updl/updl_nn_utils_udl.h"
#include "updl/updl_operator.h"

#include <assert.h>
#include <math.h>

#ifndef UPDL_ADD_USE_FLOAT_ACC
#define UPDL_ADD_USE_FLOAT_ACC 1
#endif

/**
 * @brief Basic s16 element-wise addition function
 *
 * Following CMSIS-NN approach:
 * - Element-wise addition with proper requantization
 * - Supports multiple input tensors for residual connections
 * - int16 activations with int16 outputs
 */
uint8_t updl_add_s16(int16_t **input_buffers, uint32_t num_inputs, int16_t *output,
                     uint32_t tensor_size, atype_t activation, int32_t eff_multiplier,
                     int16_t eff_shift, int16_t *input_zps, int16_t output_zp,
                     float *input_scales, float output_scale) {
    (void)eff_multiplier; // Not used for Add
    (void)eff_shift;

    if (!input_buffers || !output || num_inputs == 0 || !input_scales ||
        !input_zps || output_scale <= 0.0f) {
        return 1;
    }

    for (uint32_t i = 0; i < tensor_size; i++) {
        int32_t activated;
#if UPDL_ADD_USE_FLOAT_ACC
        float sum_fp = 0.0f;
        for (uint32_t inp = 0; inp < num_inputs && inp < 4; inp++) {
            if (!input_buffers[inp]) {
                continue;
            }
            int32_t centered =
                (int32_t)input_buffers[inp][i] - (int32_t)input_zps[inp];
            sum_fp += (float)centered * input_scales[inp];
        }

        int32_t sum_q =
            (int32_t)lrintf(sum_fp / output_scale);
        activated = updl_activation(sum_q, activation);
#else
        int64_t sum = 0;
        requant_params_t scale_params[4] = {0};
        for (uint32_t inp = 0; inp < num_inputs && inp < 4; inp++) {
            if (!input_buffers[inp]) {
                continue;
            }
            float scale_ratio = input_scales[inp] / output_scale;
            scale_params[inp] = updl_scale_to_multiplier_shift(scale_ratio);
        }

        for (uint32_t inp = 0; inp < num_inputs; inp++) {
            if (inp >= 4 || !input_buffers[inp]) {
                continue;
            }

            int32_t centered =
                (int32_t)input_buffers[inp][i] - (int32_t)input_zps[inp];

            int32_t scaled =
                updl_requantize(centered, scale_params[inp].multiplier,
                                scale_params[inp].shift);
            sum += (int64_t)scaled;
            sum = updl_udl_bound40(sum);
        }

        activated = updl_activation(updl_clamp_s32(sum), activation);
#endif

        int32_t with_zp = activated + (int32_t)output_zp;
        output[i] = updl_clamp_s16(with_zp);
    }

    return 0;
}
