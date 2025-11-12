/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#include "updl/updl_kernels.h"
#include "updl/updl_kernels_support.h"
#include "updl/updl_nn_utils_udl.h"
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
            sum = updl_udl_bound40(sum);
        }

        int16_t bias_val = bias ? bias[out_feat] : 0;
        output[out_feat] = updl_udl_finalize(
            sum, bias_val, activation, eff_multiplier, eff_shift, output_zp,
            eff_bias_multiplier, eff_bias_shift);
    }
    return 0;
}

