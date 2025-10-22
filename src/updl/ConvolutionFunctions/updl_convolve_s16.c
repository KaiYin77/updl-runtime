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

// Wrapper function removed - updl_kernels.c calls core implementations directly

/**
 * @brief Naive s16 convolution implementation without any optimizations
 * 
 * Input format: CHW (Channel-Height-Width)
 * Weight format: OIHW (Output-Input-Height-Width)
 * 
 * Simple nested loops with basic bounds checking
 */
uint8_t updl_convolve_s16(int16_t *input, int16_t *output, int16_t *weights, int16_t *bias,
                          uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                          uint32_t output_height, uint32_t output_width, uint32_t output_channels,
                          uint32_t kernel_height, uint32_t kernel_width,
                          uint32_t stride_x, uint32_t stride_y, ptype_t padding,
                          atype_t activation,
                          int32_t eff_multiplier, int16_t eff_shift,
                          int16_t input_zp, int16_t weight_zp, int16_t output_zp,
                          int32_t eff_bias_multiplier, int16_t eff_bias_shift)
{
    const uint32_t input_hw_size = input_height * input_width;
    const uint32_t output_hw_size = output_height * output_width;
    
    // Calculate padding values based on padding type
    uint32_t pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0;
    if (padding == Ptype_same) {
        // Calculate height padding
        updl_calculate_padding(input_height, kernel_height, stride_y, &pad_top, &pad_bottom);
        // Calculate width padding  
        updl_calculate_padding(input_width, kernel_width, stride_x, &pad_left, &pad_right);
    }
    
    // Naive nested loops - output spatial positions
    for (uint32_t out_y = 0; out_y < output_height; out_y++) {
        for (uint32_t out_x = 0; out_x < output_width; out_x++) {
            // Output channels
            for (uint32_t out_ch = 0; out_ch < output_channels; out_ch++) {
                int64_t sum = 0;
                
                // Input channels
                for (uint32_t in_ch = 0; in_ch < input_channels; in_ch++) {
                    // Kernel spatial dimensions
                    for (uint32_t ky = 0; ky < kernel_height; ky++) {
                        for (uint32_t kx = 0; kx < kernel_width; kx++) {
                            // Calculate input position
                            int32_t in_y = (int32_t)(out_y * stride_y) + (int32_t)ky - (int32_t)pad_top;
                            int32_t in_x = (int32_t)(out_x * stride_x) + (int32_t)kx - (int32_t)pad_left;
                            
                            // Bounds check - use zero padding for out-of-bounds
                            int16_t input_val = 0;
                            if (in_y >= 0 && in_y < (int32_t)input_height && 
                                in_x >= 0 && in_x < (int32_t)input_width) {
                                // CHW input indexing: input[in_ch][in_y][in_x]
                                input_val = input[in_ch * input_hw_size + in_y * input_width + in_x];
                            }
                            
                            // OIHW weight indexing: weights[out_ch][in_ch][ky][kx]
                            int16_t weight_val = weights[out_ch * input_channels * kernel_height * kernel_width + 
                                                        in_ch * kernel_height * kernel_width + 
                                                        ky * kernel_width + kx];

                            // Accumulate with zero-point correction
                            int32_t inp = (int32_t)input_val - (int32_t)input_zp;
                            int32_t wgt = (int32_t)weight_val - (int32_t)weight_zp;
                            sum += (int64_t)inp * (int64_t)wgt;
                        }
                    }
                }
                
                // Scale bias from bias_scale to accumulator scale using updl_scale_bias
                int64_t scaled_bias = updl_scale_bias(bias[out_ch], eff_bias_multiplier, eff_bias_shift);
                sum += scaled_bias;

                // CHW output indexing: output[out_ch][out_y][out_x] - use optimized pipeline
                output[out_ch * output_hw_size + out_y * output_width + out_x] = updl_quantize_pipeline(
                    sum, activation, eff_multiplier, eff_shift, output_zp);
            }
        }
    }
    
    return 0;
}




/**
 * @brief UPDL optimized 1x1 convolution for 64x25x5 input (CHW) with 64x64x1x1 weights (OIHW)
 * Uses 2x2 batching, bias pre-computation, and register reuse for maximum scalar performance
 */
uint8_t updl_convolve_64x1x1x64_s16_64x25x5(int16_t *input, int16_t *output, int16_t *weights, int16_t *bias,
                                       ptype_t padding, atype_t activation, int32_t eff_multiplier, int16_t eff_shift, 
                                       int16_t input_zp, int16_t weight_zp, int16_t output_zp,
                                       int32_t eff_bias_multiplier, int16_t eff_bias_shift)
{
    // Fixed dimensions based on function name - CHW layout optimized
    const uint32_t input_height = 25;
    const uint32_t input_width = 5;
    const uint32_t input_channels = 64;
    const uint32_t output_channels = 64;
    const uint32_t kernel_height = 1;
    const uint32_t kernel_width = 1;
    const uint32_t num_pixels = input_height * input_width;  // 25 * 5 = 125
    const uint32_t input_hw_size = num_pixels;
    const uint32_t output_hw_size = num_pixels;
    
    // Calculate padding (though 1x1 convolution doesn't typically need padding)
    uint32_t pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0;
    if (padding == Ptype_same) {
        updl_calculate_padding(input_height, kernel_height, 1, &pad_top, &pad_bottom);
        updl_calculate_padding(input_width, kernel_width, 1, &pad_left, &pad_right);
    }
    
    // CHW-optimized loop order: process output channels first for better cache locality
    for (uint32_t out_ch_pair = 0; out_ch_pair < output_channels; out_ch_pair += 2) {
        // OIHW weight layout: weights[out_ch][input_ch][1][1] = weights[out_ch * input_channels + input_ch]
        const int16_t *weight_ch0 = weights + out_ch_pair * input_channels;
        const int16_t *weight_ch1 = weights + (out_ch_pair + 1) * input_channels;

        // CHW output layout: output[ch][y][x] = output[ch * 125 + y*5 + x]
        int16_t *output_ptr0 = output + out_ch_pair * num_pixels;
        int16_t *output_ptr1 = output + (out_ch_pair + 1) * num_pixels;

        // Process pixels in pairs for better throughput
        uint32_t pixel_pairs = num_pixels >> 1;
        for (uint32_t pixel_pair = 0; pixel_pair < pixel_pairs; pixel_pair++) {
            uint32_t pixel_idx0 = pixel_pair * 2;
            uint32_t pixel_idx1 = pixel_pair * 2 + 1;

            int64_t sum00 = 0, sum01 = 0;
            int64_t sum10 = 0, sum11 = 0;

            const int16_t *wgt0_ptr = weight_ch0;
            const int16_t *wgt1_ptr = weight_ch1;

            // CHW input layout: process all channels for these two pixels
            for (int32_t ch = 0; ch < input_channels; ch++) {
                // CHW input indexing: input[ch][y][x] = input[ch * 125 + y*5 + x]
                int32_t inp0_val = (int32_t)input[ch * num_pixels + pixel_idx0] - input_zp;
                int32_t inp1_val = (int32_t)input[ch * num_pixels + pixel_idx1] - input_zp;
                int32_t wgt0_val = (int32_t)(*wgt0_ptr++) - (int32_t)weight_zp;
                int32_t wgt1_val = (int32_t)(*wgt1_ptr++) - (int32_t)weight_zp;

                sum00 += (int64_t)inp0_val * (int64_t)wgt0_val;
                sum01 += (int64_t)inp0_val * (int64_t)wgt1_val;
                sum10 += (int64_t)inp1_val * (int64_t)wgt0_val;
                sum11 += (int64_t)inp1_val * (int64_t)wgt1_val;
            }

            // Add bias if present with proper scaling
            if (bias) {
                int64_t b0 = updl_scale_bias(bias[out_ch_pair], eff_bias_multiplier, eff_bias_shift);
                int64_t b1 = updl_scale_bias(bias[out_ch_pair + 1], eff_bias_multiplier, eff_bias_shift);
                sum00 += b0; sum01 += b1;
                sum10 += b0; sum11 += b1;
            }

            // Use optimized pipeline instead of 5 separate function calls
            output_ptr0[pixel_idx0] = updl_quantize_pipeline(sum00, activation, eff_multiplier, eff_shift, output_zp);
            output_ptr1[pixel_idx0] = updl_quantize_pipeline(sum01, activation, eff_multiplier, eff_shift, output_zp);
            output_ptr0[pixel_idx1] = updl_quantize_pipeline(sum10, activation, eff_multiplier, eff_shift, output_zp);
            output_ptr1[pixel_idx1] = updl_quantize_pipeline(sum11, activation, eff_multiplier, eff_shift, output_zp);
        }

        // Handle leftover pixel (if num_pixels is odd) - CHW layout
        if (num_pixels % 2) {
            uint32_t last_pixel_idx = num_pixels - 1;

            int64_t sum0 = 0, sum1 = 0;
            const int16_t *wgt0_ptr = weight_ch0;
            const int16_t *wgt1_ptr = weight_ch1;

            // CHW input layout: process all channels for the last pixel
            for (int32_t ch = 0; ch < input_channels; ch++) {
                // CHW input indexing: input[ch][last_pixel]
                int32_t inp = (int32_t)input[ch * num_pixels + last_pixel_idx] - input_zp;
                int32_t w0 = (int32_t)(*wgt0_ptr++) - (int32_t)weight_zp;
                int32_t w1 = (int32_t)(*wgt1_ptr++) - (int32_t)weight_zp;
                sum0 += (int64_t)inp * (int64_t)w0;
                sum1 += (int64_t)inp * (int64_t)w1;
            }

            int64_t b0 = updl_scale_bias(bias[out_ch_pair], eff_bias_multiplier, eff_bias_shift);
            int64_t b1 = updl_scale_bias(bias[out_ch_pair + 1], eff_bias_multiplier, eff_bias_shift);
            sum0 += b0;
            sum1 += b1;

            // Use optimized pipeline instead of 5 separate function calls
            output_ptr0[last_pixel_idx] = updl_quantize_pipeline(sum0, activation, eff_multiplier, eff_shift, output_zp);
            output_ptr1[last_pixel_idx] = updl_quantize_pipeline(sum1, activation, eff_multiplier, eff_shift, output_zp);
        }
    }

    return 0;
}


/**
 * @brief UPDL optimized convolution for 1x49x10 input (CHW) with 64x1x10x4 weights (OIHW)
 * Uses optimized memory access following the same pattern as updl_convolve_s16
 * Fixed dimensions: input 1x49x10, kernel 64x1x10x4
 */
uint8_t updl_convolve_64x1x10x4_s16_1x49x10(int16_t *input, int16_t *output, int16_t *weights, int16_t *bias,
                                   uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                                   uint32_t output_height, uint32_t output_width, uint32_t output_channels,
                                   uint32_t kernel_height, uint32_t kernel_width,
                                   uint32_t stride_x, uint32_t stride_y, ptype_t padding,
                                   atype_t activation, 
                                   int32_t eff_multiplier, int16_t eff_shift,
                                   int16_t input_zp, int16_t weight_zp, int16_t output_zp,
                                   int32_t eff_bias_multiplier, int16_t eff_bias_shift)
{
    // Validate expected fixed dimensions for this optimized kernel
    if (input_height != 49 || input_width != 10 || input_channels != 1 ||
        kernel_height != 10 || kernel_width != 4 || output_channels != 64) {
        // Dimensions don't match expected values for this optimized kernel
        return 1;
    }
    
    const uint32_t input_hw_size = input_height * input_width;
    const uint32_t output_hw_size = output_height * output_width;
    
    // Calculate padding values based on padding type
    uint32_t pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0;
    if (padding == Ptype_same) {
        updl_calculate_padding(input_height, kernel_height, stride_y, &pad_top, &pad_bottom);
        updl_calculate_padding(input_width, kernel_width, stride_x, &pad_left, &pad_right);
    }
    
    // Optimized convolution loop - following updl_convolve_s16 pattern
    for (uint32_t out_y = 0; out_y < output_height; out_y++) {
        for (uint32_t out_x = 0; out_x < output_width; out_x++) {
            for (uint32_t out_ch = 0; out_ch < output_channels; out_ch++) {
                int64_t sum = 0;
                
                // Single input channel convolution
                for (uint32_t in_ch = 0; in_ch < input_channels; in_ch++) {
                    for (uint32_t ky = 0; ky < kernel_height; ky++) {
                        for (uint32_t kx = 0; kx < kernel_width; kx++) {
                            // Calculate input position
                            int32_t in_y = (int32_t)(out_y * stride_y) + (int32_t)ky - (int32_t)pad_top;
                            int32_t in_x = (int32_t)(out_x * stride_x) + (int32_t)kx - (int32_t)pad_left;
                            
                            // Bounds check - use zero padding for out-of-bounds
                            int16_t input_val = 0;
                            if (in_y >= 0 && in_y < (int32_t)input_height && 
                                in_x >= 0 && in_x < (int32_t)input_width) {
                                // CHW input indexing: input[in_ch][in_y][in_x]
                                input_val = input[in_ch * input_hw_size + in_y * input_width + in_x];
                            }
                            
                            // OIHW weight indexing: weights[out_ch][in_ch][ky][kx]
                            int16_t weight_val = weights[out_ch * input_channels * kernel_height * kernel_width + 
                                                        in_ch * kernel_height * kernel_width + 
                                                        ky * kernel_width + kx];

                            // Accumulate with zero-point correction
                            int32_t inp = (int32_t)input_val - (int32_t)input_zp;
                            int32_t wgt = (int32_t)weight_val - (int32_t)weight_zp;
                            sum += (int64_t)inp * (int64_t)wgt;
                        }
                    }
                }
                
                // Scale bias from bias_scale to accumulator scale using updl_scale_bias
                int64_t scaled_bias = updl_scale_bias(bias[out_ch], eff_bias_multiplier, eff_bias_shift);
                sum += scaled_bias;

                // CHW output indexing: output[out_ch][out_y][out_x] - use optimized pipeline
                output[out_ch * output_hw_size + out_y * output_width + out_x] = updl_quantize_pipeline(
                    sum, activation, eff_multiplier, eff_shift, output_zp);
            }
        }
    }
    return 0;
}
