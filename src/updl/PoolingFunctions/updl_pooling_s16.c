/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#include "updl/updl_kernels.h"
#include "updl/updl_kernels_support.h"
#include "updl/updl_operator.h"
#include "updl/updl_utility.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Max pooling 2D for s16 data
 * 
 * Following CMSIS-NN approach with optimized loop order for cache locality
 * Input/Output format: [Channels][Height][Width] (CHW layout)
 */
uint8_t updl_max_pool_s16(int16_t *input, int16_t *output,
                          uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                          uint32_t output_height, uint32_t output_width, uint32_t output_channels,
                          uint32_t kernel_height, uint32_t kernel_width,
                          uint32_t stride_x, uint32_t stride_y, ptype_t padding) {
    
    assert(input_channels == output_channels);
    assert(stride_x != 0 && stride_y != 0);

    // Calculate padding values based on padding type
    uint32_t pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0;
    if (padding == Ptype_same) {
        // Calculate height padding
        updl_calculate_padding(input_height, kernel_height, stride_y, &pad_top, &pad_bottom);
        // Calculate width padding  
        updl_calculate_padding(input_width, kernel_width, stride_x, &pad_left, &pad_right);
    }
    // CHW-optimized loop order: channels first for better cache locality
    const uint32_t input_hw_size = input_height * input_width;
    const uint32_t output_hw_size = output_height * output_width;
    
    for (uint32_t k = 0; k < output_channels; k++) {
        const uint32_t input_ch_base = k * input_hw_size;
        const uint32_t output_ch_base = k * output_hw_size;
        
        for (uint32_t j = 0; j < output_height; j++) {
            for (uint32_t i = 0; i < output_width; i++) {
                // Calculate input coordinates for this output position
                int32_t in_y_start = (int32_t)(j * stride_y) - (int32_t)pad_top;
                int32_t in_x_start = (int32_t)(i * stride_x) - (int32_t)pad_left;
                int16_t max_val = INT16_MIN;

                // Find maximum over kernel window for this channel
                for (uint32_t kj = 0; kj < kernel_height; kj++) {
                    for (uint32_t ki = 0; ki < kernel_width; ki++) {
                        // Calculate actual input coordinates
                        int32_t in_y = in_y_start + (int32_t)kj;
                        int32_t in_x = in_x_start + (int32_t)ki;

                        // Check if we're within input bounds
                        if (in_y >= 0 && in_y < (int32_t)input_height && 
                            in_x >= 0 && in_x < (int32_t)input_width) {
                            // CHW input layout: input[k][in_y][in_x]
                            uint32_t in_position = input_ch_base + in_y * input_width + in_x;
                            max_val = input[in_position] > max_val ? input[in_position] : max_val;
                        }
                        // For pooling, out-of-bounds values are typically treated as
                        // -infinity for max pooling but since we initialize max to INT16_MIN,
                        // we effectively ignore out-of-bounds values
                    }
                }

                // Store result in CHW layout: output[k][j][i]
                uint32_t out_position = output_ch_base + j * output_width + i;
                output[out_position] = max_val;
            }
        }
    }

    return 0;
}

/**
 * @brief Average pooling 2D for s16 data
 * 
 * Following CMSIS-NN approach with proper handling of padding
 * Input/Output format: [Channels][Height][Width] (CHW layout)
 */
uint8_t updl_avg_pool_s16(int16_t *input, int16_t *output,
                          uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                          uint32_t output_height, uint32_t output_width, uint32_t output_channels,
                          uint32_t kernel_height, uint32_t kernel_width,
                          uint32_t stride_x, uint32_t stride_y, ptype_t padding,
                          int32_t eff_multiplier, int16_t eff_shift,
                          int16_t input_zp, int16_t output_zp) {
    
    assert(input_channels == output_channels);
    assert(stride_x != 0 && stride_y != 0);

    // Calculate padding values based on padding type
    uint32_t pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0;
    if (padding == Ptype_same) {
        // Calculate height padding
        updl_calculate_padding(input_height, kernel_height, stride_y, &pad_top, &pad_bottom);
        // Calculate width padding  
        updl_calculate_padding(input_width, kernel_width, stride_x, &pad_left, &pad_right);
    }
    // CHW-optimized loop order: channels first for better cache locality  
    const uint32_t input_hw_size = input_height * input_width;
    const uint32_t output_hw_size = output_height * output_width;
    


    for (uint32_t k = 0; k < output_channels; k++) {
        const uint32_t input_ch_base = k * input_hw_size;
        const uint32_t output_ch_base = k * output_hw_size;
        
        for (uint32_t j = 0; j < output_height; j++) {
            for (uint32_t i = 0; i < output_width; i++) {
                // Calculate input coordinates for this output position
                int32_t in_y_start = (int32_t)(j * stride_y) - (int32_t)pad_top;
                int32_t in_x_start = (int32_t)(i * stride_x) - (int32_t)pad_left;
                int64_t sum = 0;  // Use int64_t to prevent overflow during accumulation
                uint32_t valid_count = 0;

                // Calculate average over kernel window for this channel
                for (uint32_t kj = 0; kj < kernel_height; kj++) {
                    for (uint32_t ki = 0; ki < kernel_width; ki++) {
                        // Calculate actual input coordinates
                        int32_t in_y = in_y_start + (int32_t)kj;
                        int32_t in_x = in_x_start + (int32_t)ki;

                        // Check if we're within input bounds
                        if (in_y >= 0 && in_y < (int32_t)input_height && 
                            in_x >= 0 && in_x < (int32_t)input_width) {
                            // CHW input layout: input[k][in_y][in_x]
                            uint32_t in_position = input_ch_base + in_y * input_width + in_x;
                            int16_t input_val = input[in_position];
                            
                            // Apply zero-point correction like in convolution
                            int32_t corrected_val = (int32_t)input_val - (int32_t)input_zp;
                            sum += corrected_val;
                            valid_count++;
                        }
                    }
                }

                // For average pooling, divide by the number of valid (non-padded) values and requantize
                int16_t final_value;
                if (valid_count > 0) {
                    // Calculate average in accumulator precision
                    int64_t avg_sum = sum / (int64_t)valid_count;
                    
                    // Clamp to 32-bit for requantization (similar to convolution)
                    int32_t raw_avg = updl_clamp_s32(avg_sum);
                    
                    // Apply requantization with multiplier and shift
                    int32_t quantized = updl_requantize(raw_avg, eff_multiplier, eff_shift);
                    
                    // Add output zero-point
                    quantized += output_zp;
                    
                    // Clamp to int16 range
                    final_value = updl_clamp_s16(quantized);
                } else {
                    final_value = output_zp;  // Use output zero-point for empty regions
                }

                // Store result in CHW layout: output[k][j][i]
                uint32_t out_position = output_ch_base + j * output_width + i;
                output[out_position] = final_value;
            }
        }
    }

    return 0;
}

/**
 * @brief Global average pooling for s16 data
 * 
 * Reduces spatial dimensions to 1x1 by averaging over entire spatial extent
 */
uint8_t updl_global_avg_pool_s16(int16_t *input, int16_t *output,
                                 uint32_t input_height, uint32_t input_width, uint32_t input_channels) {
    
    const uint32_t total_pixels = input_height * input_width;
    
    // Process each channel independently
    for (uint32_t ch = 0; ch < input_channels; ch++) {
        register int64_t sum = 0;
        
        // Sum all values for this channel across spatial dimensions
        // CHW layout: input[ch][h][w]
        const uint32_t ch_base = ch * input_height * input_width;
        for (uint32_t h = 0; h < input_height; h++) {
            for (uint32_t w = 0; w < input_width; w++) {
                uint32_t in_position = ch_base + h * input_width + w;
                sum += input[in_position];
            }
        }
        
        // Calculate average
        output[ch] = (int16_t)(sum / total_pixels);
    }
    
    return 0;
}