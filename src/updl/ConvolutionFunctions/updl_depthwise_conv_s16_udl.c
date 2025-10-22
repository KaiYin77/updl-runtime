/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file updl_depthwise_conv_s16_udl.c
 * @brief Hardware dispatch functions for DepthwiseConv2D operations
 * 
 * This file provides hardware acceleration hooks for DepthwiseConv2D operations,
 * interfacing with the UDL (Upbeat Deep Learning) hardware accelerator.
 * Since the hardware doesn't natively support depthwise convolution, we compose
 * it using multiple regular convolution calls (one per input channel).
 */

#include "updl/updl_kernels.h"
#include "updl/updl_kernels_support.h"
#include "updl/updl_operator.h"
#include <stdlib.h>  // For malloc/free
#include <string.h>  // For memcpy
#include <stdbool.h> // For bool type

// Hardware accelerator includes (conditional compilation)
#ifdef USE_UPDL_UDL_KERNEL
#include "metal.h"
#include "metal-platform.h"
#include "metal/udl.h"
#include "metal/upt_isr.h"
#endif

/**
 * @brief Hardware dispatch function for DepthwiseConv2D operations
 * 
 * Implements depthwise convolution by decomposing it into multiple regular
 * convolutions (one per input channel) and composing the results.
 * 
 * @param input Input tensor data
 * @param output Output tensor data  
 * @param weights Weight data (depthwise format)
 * @param bias Bias data
 * @param input_height Input height dimension
 * @param input_width Input width dimension
 * @param input_channels Input channel dimension
 * @param output_height Output height dimension
 * @param output_width Output width dimension
 * @param output_channels Output channel dimension
 * @param kernel_height Kernel height
 * @param kernel_width Kernel width
 * @param depth_multiplier Depth multiplier (typically 1)
 * @param stride_x Stride in x direction
 * @param stride_y Stride in y direction
 * @param padding Padding type
 * @param activation Activation function
 * @param eff_multiplier Effective multiplier for quantization
 * @param eff_shift Effective shift for quantization
 * @param input_zp Input zero point
 * @param weight_zp Weight zero point
 * @param output_zp Output zero point
 * @param eff_bias_multiplier Effective bias multiplier
 * @param eff_bias_shift Effective bias shift
 * @return 0 on success, non-zero on error
 */
uint8_t updl_depthwise_conv_s16_udl(int16_t *input, int16_t *output, int16_t *weights, int16_t *bias,
                                    uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                                    uint32_t output_height, uint32_t output_width, uint32_t output_channels,
                                    uint32_t kernel_height, uint32_t kernel_width, uint32_t depth_multiplier,
                                    uint32_t stride_x, uint32_t stride_y, ptype_t padding,
                                    atype_t activation,
                                    int32_t eff_multiplier, int16_t eff_shift,
                                    int16_t input_zp, int16_t weight_zp, int16_t output_zp,
                                    int32_t eff_bias_multiplier, int16_t eff_bias_shift)
{
#ifdef USE_UPDL_UDL_KERNEL
    
    // Validate depthwise convolution constraints
    if (depth_multiplier != 1) {
        updl_Warning("Hardware accelerator only supports depth_multiplier=1, got %u\n", depth_multiplier);
        return 1; // Fallback to software
    }
    
    // Get UDL hardware device (assuming UDL1_1 is available)
    metal_udl_Type *udl_device = upt_udl_get_device(0); // NAON_UDL_1_1
    if (!udl_device) {
        updl_Error("udl_device not found");
        return 1; // Hardware not available, fallback to software
    }
    
    // Reset UDL module
    metal_udl_reset(udl_device);
    
    // WORKAROUND: Handle padding in software (same as regular conv)
    int16_t *padded_input = input;
    uint32_t padded_height = input_height;
    uint32_t padded_width = input_width;
    
    // Calculate padding values if SAME padding is requested
    uint32_t pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0;
    if (padding == Ptype_same) {
        updl_calculate_padding(input_height, kernel_height, stride_y, &pad_top, &pad_bottom);
        updl_calculate_padding(input_width, kernel_width, stride_x, &pad_left, &pad_right);
        
        padded_height = input_height + pad_top + pad_bottom;
        padded_width = input_width + pad_left + pad_right;
        
        // Allocate temporary buffer for padded input (if padding needed)
        if (pad_top > 0 || pad_bottom > 0 || pad_left > 0 || pad_right > 0) {
            size_t padded_size = padded_height * padded_width * input_channels * sizeof(int16_t);
            padded_input = (int16_t*)malloc(padded_size);
            if (!padded_input) {
                updl_Error("Failed to allocate padded input buffer\n");
                return 1;
            }
            
            // Apply zero-padding with zero-point value
            for (uint32_t c = 0; c < input_channels; c++) {
                for (uint32_t h = 0; h < padded_height; h++) {
                    for (uint32_t w = 0; w < padded_width; w++) {
                        uint32_t padded_idx = c * padded_height * padded_width + h * padded_width + w;
                        
                        // Check if we're in the padding region
                        if (h < pad_top || h >= (pad_top + input_height) ||
                            w < pad_left || w >= (pad_left + input_width)) {
                            // Padding region - use zero-point value
                            padded_input[padded_idx] = input_zp;
                        } else {
                            // Original input region
                            uint32_t orig_h = h - pad_top;
                            uint32_t orig_w = w - pad_left;
                            uint32_t orig_idx = c * input_height * input_width + orig_h * input_width + orig_w;
                            padded_input[padded_idx] = input[orig_idx];
                        }
                    }
                }
            }
        }
    }
    
    // WORKAROUND 2: Handle stride > 1 by using stride=1 in hardware then subsampling
    uint32_t hw_output_height = output_height;
    uint32_t hw_output_width = output_width;
    bool need_subsampling = false;
    
    if (stride_y > 1 || stride_x > 1) {
        // Calculate intermediate output size with stride=1
        hw_output_height = padded_height - kernel_height + 1;
        hw_output_width = padded_width - kernel_width + 1;
        need_subsampling = true;        
    }
    
    // DEPTHWISE COMPOSITION: Process each input channel separately
    // Since depthwise conv processes each channel independently, we can use
    // the hardware's regular convolution with input_channels=1, output_channels=1
    
    // Size calculations for memory operations (kept for future reference)
    const size_t single_channel_output_size = hw_output_height * hw_output_width * sizeof(int16_t);
    
    // Hardware may require contiguous memory layout - use temporary aligned buffers
    // Allocate 4-byte aligned temporary buffers for single-channel processing
    const size_t single_channel_input_size_bytes = padded_height * padded_width * sizeof(int16_t);
    const size_t single_channel_weight_size_bytes = kernel_height * kernel_width * sizeof(int16_t);
    
    // Ensure sizes are multiples of 4 for alignment
    size_t aligned_input_size = (single_channel_input_size_bytes + 3) & ~3;
    size_t aligned_weight_size = (single_channel_weight_size_bytes + 3) & ~3;
    
    // Use portable aligned allocation
    int16_t *temp_input_buffer = (int16_t*)aligned_malloc(4, aligned_input_size);
    int16_t *temp_weight_buffer = (int16_t*)aligned_malloc(4, aligned_weight_size);
    int16_t *temp_bias_buffer = (int16_t*)aligned_malloc(4, 4); // Single bias value, 4-byte aligned
    
    if (!temp_input_buffer || !temp_weight_buffer || !temp_bias_buffer) {
        updl_Error("Failed to allocate aligned temporary buffers\n");
        if (padded_input != input) free(padded_input);
        if (temp_input_buffer) aligned_free(temp_input_buffer);
        if (temp_weight_buffer) aligned_free(temp_weight_buffer);
        if (temp_bias_buffer) aligned_free(temp_bias_buffer);
        return 1;
    }
    
    // Setup UPT-based convolution parameters for hardware (single channel)
    upt_nn_conv_params_Type upt_conv_params;
    upt_nn_dims_Type upt_input_dims;
    upt_nn_dims_Type upt_weight_dims;
    upt_nn_dims_Type upt_output_dims;
    
    // Configure dimensions for single-channel hardware operations
    upt_input_dims.w = padded_width;
    upt_input_dims.h = padded_height;
    upt_input_dims.c = 1;  // Single channel per hardware call
    
    upt_weight_dims.w = kernel_width;
    upt_weight_dims.h = kernel_height;
    
    upt_output_dims.w = hw_output_width;
    upt_output_dims.h = hw_output_height;
    upt_output_dims.c = 1;  // Single channel per hardware call
    
    // Configure convolution parameters
    upt_conv_params.stride.w = stride_x;
    upt_conv_params.stride.h = 1;  // Always use stride=1 for hardware
    
    // Map quantization parameters to hardware format
    int8_t calib_acc_shift = 0;
    upt_conv_params.acc_rshift = calib_acc_shift;
    
    // Convert signed eff_bias_shift to positive left shift value for hardware
    if (eff_bias_shift < 0) {
        upt_conv_params.bias_lshift = (uint8_t)(-eff_bias_shift);
    } else {
        upt_conv_params.bias_lshift = 0;
        updl_Warning("Positive eff_bias_shift (%d) not directly supported by hardware, using bias_lshift=0\n", eff_bias_shift);
    }
    
    upt_conv_params.leaky_pos_rshift = 0;
    upt_conv_params.leaky_neg_rshift = 0;
    
    // Map activation type to hardware format
    switch (activation) {
        case Atype_relu:
            upt_conv_params.activation = udl_atype_relu;
            upt_conv_params.leaky_pos_rshift = eff_shift;
            upt_conv_params.leaky_neg_rshift = eff_shift; 
            break;
        case Atype_leakyrelu:
            upt_conv_params.activation = udl_atype_leakyrelu;
            upt_conv_params.leaky_pos_rshift = eff_shift;
            upt_conv_params.leaky_neg_rshift = eff_shift + 3;
            break;
        default:
            upt_conv_params.activation = udl_atype_linear;
            upt_conv_params.leaky_pos_rshift = eff_shift;
            upt_conv_params.leaky_neg_rshift = eff_shift; 
            break;
    }
    
    const uint32_t DEFAULT_OUTPUT_ADDR = 0x50000000;
    
    // Process each channel separately using hardware convolution
    for (uint32_t ch = 0; ch < input_channels; ch++) {
        
        // Copy single channel data to aligned temporary buffers
        // Single channel input (CHW format)
        const int16_t *src_input = padded_input + ch * padded_height * padded_width;
        memcpy(temp_input_buffer, src_input, single_channel_input_size_bytes);
        
        // Single channel weight (depthwise format)
        // Depthwise weights layout: weights[ch * kernel_height * kernel_width + ky * kernel_width + kx]
        const int16_t *src_weight = weights + ch * kernel_height * kernel_width;
        memcpy(temp_weight_buffer, src_weight, single_channel_weight_size_bytes);
        
        // Single bias value
        temp_bias_buffer[0] = bias ? bias[ch] : 0;
        
        // Bounds checking on source pointers
        if (src_input < padded_input || 
            src_input >= (padded_input + input_channels * padded_height * padded_width)) {
            updl_Error("Invalid input pointer for channel %u\n", ch);
            if (padded_input != input) free(padded_input);
            aligned_free(temp_input_buffer);
            aligned_free(temp_weight_buffer);
            aligned_free(temp_bias_buffer);
            return 1;
        }
        
        // Reset hardware state between channels to avoid conflicts
        if (ch > 0) {
            metal_udl_reset(udl_device);
        }
        
        // Call hardware convolution for this single channel using aligned buffers
        enum_udl_retcode_Type result = metal_udl_conv_2d_16x16_upt_api(
            udl_device,
            &upt_conv_params,
            &upt_input_dims,
            (uint32_t *)temp_input_buffer,
            &upt_weight_dims,
            (uint32_t *)temp_weight_buffer,
            (uint32_t *)temp_bias_buffer,
            &upt_output_dims,
            (uint32_t *)DEFAULT_OUTPUT_ADDR);
        
        if (result != E_UDL_SUCCESS) {
            updl_Error("Hardware API call failed for channel %u with result: %d\n", ch, result);
            // Cleanup and return error
            if (padded_input != input) free(padded_input);
            aligned_free(temp_input_buffer);
            aligned_free(temp_weight_buffer);
            aligned_free(temp_bias_buffer);
            return 1;
        }
        
        
        // Wait for hardware completion with timeout
        while (!metal_udl_read_pending(udl_device));

        // Handle memory rearrangement (single channel output)
        metal_udl_rearrange(udl_device, (uint32_t *)DEFAULT_OUTPUT_ADDR, 
                           hw_output_width * hw_output_height * 1); // 1 channel
        
        // Copy result to final output location with stride handling
        const int16_t *hw_result = (const int16_t *)DEFAULT_OUTPUT_ADDR;
        
        if (need_subsampling) {
            // Subsample and copy to final output (CHW format)
            for (uint32_t h = 0; h < output_height; h++) {
                for (uint32_t w = 0; w < output_width; w++) {
                    // Source index from hardware output (stride subsampling)
                    uint32_t src_idx = (h * stride_y) * hw_output_width + (w * stride_x);
                    
                    // Destination index in final output (CHW format)
                    uint32_t dst_idx = ch * output_height * output_width + h * output_width + w;
                    
                    output[dst_idx] = hw_result[src_idx];
                }
            }
        } else {
            // Direct copy to final output (CHW format)
            int16_t *dst_channel = output + ch * output_height * output_width;
            memcpy(dst_channel, hw_result, single_channel_output_size);
        }
    }
    
    // Cleanup allocated buffers
    if (padded_input != input) {
        free(padded_input);
    }
    aligned_free(temp_input_buffer);
    aligned_free(temp_weight_buffer);
    aligned_free(temp_bias_buffer);
    
    return 0; // Success
    
#else
    // Hardware not available, return error to trigger software fallback
    return 1;
#endif
}