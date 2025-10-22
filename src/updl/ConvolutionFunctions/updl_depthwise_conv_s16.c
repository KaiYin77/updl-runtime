/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#include "updl/updl_kernels.h"
#include "updl/updl_kernels_support.h"
#include "updl/updl_operator.h"
#include "updl/updl_utility.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Optimized depthwise convolution for 64x25x5 input (CHW) with 64x3x3 weights (depthwise format)
 * Specialized for: depth_multiplier = 1, kernel_size = 3x3, stride = 1x1
 * Follows the same pattern as updl_depthwise_conv_s16 with proper padding calculation
 */
uint8_t updl_depthwise_conv_64x3x3_s16_64x25x5(
    int16_t *input, int16_t *output, int16_t *weights, int16_t *bias,
    ptype_t padding, atype_t activation,
    int32_t eff_multiplier, int16_t eff_shift,
    int16_t input_zp, int16_t weight_zp, int16_t output_zp,
    int32_t eff_bias_multiplier, int16_t eff_bias_shift)
{
    // Fixed dimensions for this highly optimized kernel
    const uint32_t input_height = 25, input_width = 5, input_channels = 64;
    const uint32_t output_height = 25, output_width = 5;
    const uint32_t input_hw_size = 125;  // 25 * 5, compile-time constant
    const uint32_t output_hw_size = 125; // 25 * 5, compile-time constant

    // Pre-compute zero-point corrections (hoisted out of loops)
    const int32_t input_zp_i32 = (int32_t)input_zp;
    const int32_t weight_zp_i32 = (int32_t)weight_zp;
    const int32_t output_zp_i32 = (int32_t)output_zp;

    // Calculate padding once for all channels (stride=1, kernel=3x3)
    uint32_t pad_top = 0, pad_left = 0;
    if (padding == Ptype_same) {
        pad_top = 1;  // For kernel=3, stride=1: pad = (3-1)/2 = 1
        pad_left = 1; // For kernel=3, stride=1: pad = (3-1)/2 = 1
    }
    
    // Compile-time constants for inner loop bounds
    const int32_t pad_top_i32 = (int32_t)pad_top;
    const int32_t pad_left_i32 = (int32_t)pad_left;
    
    // OPTIMIZATION 1: Eliminate depth_multiplier loop (depth_multiplier=1 is constant)
    // OPTIMIZATION 2: Channel-wise processing with maximum cache efficiency
    for (uint32_t ch = 0; ch < 64; ch++) {
        const uint32_t ch_input_base = ch * input_hw_size;
        const uint32_t ch_output_base = ch * output_hw_size;
        
        // Pre-load all 9 weights for this channel (3x3 kernel)
        const int16_t *w_ch = weights + ch * 9; // weights[ch * 9 + ky*3 + kx]
        
        // OPTIMIZATION 3: Unroll 3x3 kernel and pre-compute weight terms
        const int32_t w00 = (int32_t)w_ch[0] - weight_zp_i32; // [0][0]
        const int32_t w01 = (int32_t)w_ch[1] - weight_zp_i32; // [0][1]
        const int32_t w02 = (int32_t)w_ch[2] - weight_zp_i32; // [0][2]
        const int32_t w10 = (int32_t)w_ch[3] - weight_zp_i32; // [1][0]
        const int32_t w11 = (int32_t)w_ch[4] - weight_zp_i32; // [1][1]
        const int32_t w12 = (int32_t)w_ch[5] - weight_zp_i32; // [1][2]
        const int32_t w20 = (int32_t)w_ch[6] - weight_zp_i32; // [2][0]
        const int32_t w21 = (int32_t)w_ch[7] - weight_zp_i32; // [2][1]
        const int32_t w22 = (int32_t)w_ch[8] - weight_zp_i32; // [2][2]
        
        // Pre-compute bias term
        const int64_t scaled_bias = updl_scale_bias(bias[ch], eff_bias_multiplier, eff_bias_shift);
        
        // OPTIMIZATION 4: Spatial loop optimization - process row by row for better cache locality
        for (uint32_t out_y = 0; out_y < output_height; out_y++) {
            const int32_t in_y_base = (int32_t)out_y - pad_top_i32;
            
            // Pre-compute input row pointers (with bounds checking)
            const int16_t *in_row0 = NULL, *in_row1 = NULL, *in_row2 = NULL;
            const int32_t in_y0 = in_y_base + 0;
            const int32_t in_y1 = in_y_base + 1;
            const int32_t in_y2 = in_y_base + 2;
            
            if (in_y0 >= 0 && in_y0 < 25) in_row0 = input + ch_input_base + in_y0 * 5;
            if (in_y1 >= 0 && in_y1 < 25) in_row1 = input + ch_input_base + in_y1 * 5;
            if (in_y2 >= 0 && in_y2 < 25) in_row2 = input + ch_input_base + in_y2 * 5;
            
            // OPTIMIZATION 5: Unroll x-direction for width=5 (small width allows full unroll)
            for (uint32_t out_x = 0; out_x < 5; out_x++) {
                const int32_t in_x_base = (int32_t)out_x - pad_left_i32;
                
                int64_t sum = 0;
                
                // OPTIMIZATION 6: Manually unroll 3x3 convolution with inline bounds checking
                // Row 0 of kernel
                const int32_t in_x0 = in_x_base + 0;
                const int32_t in_x1 = in_x_base + 1;
                const int32_t in_x2 = in_x_base + 2;
                
                // Row 0: y-1
                if (in_row0) {
                    if (in_x0 >= 0 && in_x0 < 5) sum += (int64_t)((int32_t)in_row0[in_x0] - input_zp_i32) * w00;
                    if (in_x1 >= 0 && in_x1 < 5) sum += (int64_t)((int32_t)in_row0[in_x1] - input_zp_i32) * w01;
                    if (in_x2 >= 0 && in_x2 < 5) sum += (int64_t)((int32_t)in_row0[in_x2] - input_zp_i32) * w02;
                }
                
                // Row 1: y
                if (in_row1) {
                    if (in_x0 >= 0 && in_x0 < 5) sum += (int64_t)((int32_t)in_row1[in_x0] - input_zp_i32) * w10;
                    if (in_x1 >= 0 && in_x1 < 5) sum += (int64_t)((int32_t)in_row1[in_x1] - input_zp_i32) * w11;
                    if (in_x2 >= 0 && in_x2 < 5) sum += (int64_t)((int32_t)in_row1[in_x2] - input_zp_i32) * w12;
                }
                
                // Row 2: y+1
                if (in_row2) {
                    if (in_x0 >= 0 && in_x0 < 5) sum += (int64_t)((int32_t)in_row2[in_x0] - input_zp_i32) * w20;
                    if (in_x1 >= 0 && in_x1 < 5) sum += (int64_t)((int32_t)in_row2[in_x1] - input_zp_i32) * w21;
                    if (in_x2 >= 0 && in_x2 < 5) sum += (int64_t)((int32_t)in_row2[in_x2] - input_zp_i32) * w22;
                }
                
                // Add pre-computed bias
                sum += scaled_bias;

                // OPTIMIZATION 7: Use optimized 5-function pipeline (eliminates 40,000 function calls)
                output[ch_output_base + out_y * 5 + out_x] = updl_quantize_pipeline(
                    sum, activation, eff_multiplier, eff_shift, output_zp);
            }
        }
    }
    return 0;
}


/**
 * @brief General depthwise convolution (fallback for non-optimized cases)
 */
uint8_t updl_depthwise_conv_s16(
    int16_t *input, int16_t *output, int16_t *weights, int16_t *bias,
    uint32_t input_height, uint32_t input_width, uint32_t input_channels,
    uint32_t output_height, uint32_t output_width, uint32_t output_channels,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t depth_multiplier,
    uint32_t stride_x, uint32_t stride_y, ptype_t padding,
    atype_t activation,
    int32_t eff_multiplier, int16_t eff_shift,
    int16_t input_zp, int16_t weight_zp, int16_t output_zp,
    int32_t eff_bias_multiplier, int16_t eff_bias_shift)
{
    // CHW layout dimensions
    const uint32_t input_hw_size = input_height * input_width;
    const uint32_t output_hw_size = output_height * output_width;
    const uint32_t kernel_volume = kernel_height * kernel_width;

    // Calculate padding values based on padding type
    uint32_t pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0;
    if (padding == Ptype_same) {
        // Calculate height padding
        updl_calculate_padding(input_height, kernel_height, stride_y, &pad_top, &pad_bottom);
        // Calculate width padding  
        updl_calculate_padding(input_width, kernel_width, stride_x, &pad_left, &pad_right);
    }
    
    // CHW-optimized loop order: process channels first for better cache locality
    for (uint32_t input_ch = 0; input_ch < input_channels; input_ch++) {
        const uint32_t input_ch_base = input_ch * input_hw_size;
        
        for (uint32_t depth_idx = 0; depth_idx < depth_multiplier; depth_idx++) {
            uint32_t output_ch = input_ch * depth_multiplier + depth_idx;
            const uint32_t output_ch_base = output_ch * output_hw_size;
            // I1HW weight layout: weights[input_ch][depth_idx][ky][kx]
            const int16_t *weight_base = weights + input_ch * depth_multiplier * kernel_volume + depth_idx * kernel_volume;
            
            for (uint32_t out_y = 0; out_y < output_height; out_y++) {
                for (uint32_t out_x = 0; out_x < output_width; out_x++) {
                    int32_t in_y_start = (int32_t)(out_y * stride_y) - (int32_t)pad_top;
                    int32_t in_x_start = (int32_t)(out_x * stride_x) - (int32_t)pad_left;
                    int64_t sum = 0;

                    for (uint32_t ky = 0; ky < kernel_height; ky++) {
                        for (uint32_t kx = 0; kx < kernel_width; kx++) {
                            // Calculate input position
                            int32_t in_y = in_y_start + (int32_t)ky;
                            int32_t in_x = in_x_start + (int32_t)kx;
                            
                            // Bounds check - use zero padding for out-of-bounds
                            int16_t input_val = 0;
                            if (in_y >= 0 && in_y < (int32_t)input_height && 
                                in_x >= 0 && in_x < (int32_t)input_width) {
                                // CHW input indexing: input[input_ch][in_y][in_x]
                                input_val = input[input_ch_base + in_y * input_width + in_x];
                            }
                            
                            // I1HW weight indexing: weights[input_ch][depth_idx][ky][kx]
                            int16_t weight_val = weight_base[ky * kernel_width + kx];

                            // Accumulate with zero-point correction
                            int32_t inp = (int32_t)input_val - (int32_t)input_zp;
                            int32_t wgt = (int32_t)weight_val - (int32_t)weight_zp;
                            sum += (int64_t)inp * (int64_t)wgt;
                        }
                    }

                    // Scale bias from bias_scale to accumulator scale using updl_scale_bias
                    int64_t scaled_bias = updl_scale_bias(bias[output_ch], eff_bias_multiplier, eff_bias_shift);
                    sum += scaled_bias;

                    // CHW output indexing: output[output_ch][out_y][out_x] - use optimized pipeline
                    output[output_ch_base + out_y * output_width + out_x] = updl_quantize_pipeline(
                        sum, activation, eff_multiplier, eff_shift, output_zp);
                }
            }
        }
    }
    return 0;
}
