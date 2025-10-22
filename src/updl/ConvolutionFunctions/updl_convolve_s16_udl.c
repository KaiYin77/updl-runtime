/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file updl_conv2d_s16_udl.c
 * @brief Hardware dispatch functions for Conv2D operations
 * 
 * This file provides hardware acceleration hooks for Conv2D operations,
 * interfacing with the UDL (Upbeat Deep Learning) hardware accelerator
 * based on the reference test_ai_accelerator.c implementation.
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
 * @brief Hardware dispatch function for Conv2D operations
 * 
 * @param input Input tensor data
 * @param output Output tensor data  
 * @param weights Weight data
 * @param bias Bias data
 * @param input_height Input height dimension
 * @param input_width Input width dimension
 * @param input_channels Input channel dimension
 * @param output_height Output height dimension
 * @param output_width Output width dimension
 * @param output_channels Output channel dimension
 * @param kernel_height Kernel height
 * @param kernel_width Kernel width
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
uint8_t updl_conv2d_s16_udl(int16_t *input, int16_t *output, int16_t *weights, int16_t *bias,
                                uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                                uint32_t output_height, uint32_t output_width, uint32_t output_channels,
                                uint32_t kernel_height, uint32_t kernel_width,
                                uint32_t stride_x, uint32_t stride_y, ptype_t padding,
                                atype_t activation,
                                int32_t eff_multiplier, int16_t eff_shift,
                                int16_t input_zp, int16_t weight_zp, int16_t output_zp,
                                int32_t eff_bias_multiplier, int16_t eff_bias_shift)
{
#ifdef USE_UPDL_UDL_KERNEL
    // updl_Info("Dispatch to %s\n", __func__);
    // Get UDL hardware device (assuming UDL1_1 is available)
    metal_udl_Type *udl_device = upt_udl_get_device(0); // NAON_UDL_1_1
    if (!udl_device) {
        updl_Error("udl_device not found");
        return 1; // Hardware not available, fallback to software
    }
    
    // Reset UDL module
    metal_udl_reset(udl_device);
    
    // WORKAROUND 1: Handle padding in software (hardware doesn't support padding)
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
    
    // WORKAROUND 2: Handle stride_y > 1 by using stride=1 in hardware then subsampling
    uint32_t hw_output_height = output_height;
    uint32_t hw_output_width = output_width;
    bool need_subsampling = false;
    
    if (stride_y > 1) {
        // Calculate intermediate output size with stride=1
        hw_output_height = padded_height - kernel_height + 1;
        need_subsampling = true;        
    }
    
    // Setup UPT-based convolution parameters for hardware
    upt_nn_conv_params_Type upt_conv_params;
    upt_nn_dims_Type upt_input_dims;
    upt_nn_dims_Type upt_weight_dims;
    upt_nn_dims_Type upt_output_dims;
    
    // Configure dimensions for hardware (with workarounds applied)
    upt_input_dims.w = padded_width;
    upt_input_dims.h = padded_height;
    upt_input_dims.c = input_channels;
    
    // Configure weight dimensions
    upt_weight_dims.w = kernel_width;
    upt_weight_dims.h = kernel_height;
    
    upt_output_dims.w = hw_output_width;
    upt_output_dims.h = hw_output_height;
    upt_output_dims.c = output_channels;
    
    // Configure convolution parameters
    upt_conv_params.stride.w = stride_x;
    upt_conv_params.stride.h = 1;  // Always use stride=1 for hardware
    
    // Map quantization parameters to hardware format
    int8_t calib_acc_shift = 0;
    upt_conv_params.acc_rshift = calib_acc_shift; // Currently Supposed that acc will not overflow 40bits
    
    // Convert signed eff_bias_shift to positive left shift value for hardware
    // eff_bias_shift convention: negative = left shift, positive = right shift
    // Hardware expects: positive uint8_t = left shift amount
    if (eff_bias_shift < 0) {
        // Negative eff_bias_shift means left shift, convert to positive value
        upt_conv_params.bias_lshift = (uint8_t)(-eff_bias_shift);
    } else {
        // Positive eff_bias_shift means right shift, hardware doesn't support this directly
        // Set to 0 and handle in software if needed
        upt_conv_params.bias_lshift = 0;
        updl_Warning("Positive eff_bias_shift (%d) not directly supported by hardware, using bias_lshift=0\n", eff_bias_shift);
    }
    
    upt_conv_params.leaky_pos_rshift = 0; // Default for non-leaky activations
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
    
    // User defined output address (typically in L2 SRAM)
    const uint32_t *upt_output_ref = (uint32_t *)output;
    
    // Call hardware convolution function
    enum_udl_retcode_Type result = metal_udl_conv_2d_16x16_upt_api(
        udl_device,
        &upt_conv_params,
        &upt_input_dims,
        (uint32_t *)padded_input,
        &upt_weight_dims,
        (uint32_t *)weights,
        (uint32_t *)bias,
        &upt_output_dims,
        upt_output_ref);
    
    if (result != E_UDL_SUCCESS) {
        updl_Error("Hardware API call failed with result: %d\n", result);
        // Cleanup allocated buffers
        if (padded_input != input) free(padded_input);
        return 1; // Hardware execution failed
    }
    
    // Wait for hardware completion (polling mode)
    while (!metal_udl_read_pending(udl_device));
    
    // Handle memory rearrangement if needed (workaround for memory wiring issue)
    const uint32_t DEFAULT_OUTPUT_ADDR = 0x50000000;
    metal_udl_rearrange(udl_device, (uint32_t *)DEFAULT_OUTPUT_ADDR, 
                       hw_output_width * hw_output_height * output_channels);
    
    // POST-PROCESSING: Handle stride subsampling or direct copy
    if (need_subsampling) {
        // Subsample directly from DEFAULT_OUTPUT_ADDR to final output
        const int16_t *hw_result = (const int16_t *)DEFAULT_OUTPUT_ADDR;
        for (uint32_t c = 0; c < output_channels; c++) {
            for (uint32_t h = 0; h < output_height; h++) {
                for (uint32_t w = 0; w < output_width; w++) {
                    // Source index from full-resolution hardware output at DEFAULT_OUTPUT_ADDR
                    uint32_t src_idx = c * hw_output_height * hw_output_width + 
                                      (h * stride_y) * hw_output_width + w;
                    
                    // Destination index in final output
                    uint32_t dst_idx = c * output_height * output_width + 
                                      h * output_width + w;
                    
                    output[dst_idx] = hw_result[src_idx];
                }
            }
        }
    } else {
        // No subsampling needed - direct copy from DEFAULT_OUTPUT_ADDR to output
        memcpy(output, (void *)DEFAULT_OUTPUT_ADDR, 
               hw_output_width * hw_output_height * output_channels * sizeof(int16_t));
    }
    
    // Cleanup allocated buffers
    if (padded_input != input) {
        free(padded_input);
    }
    return 0; // Success
    
#else
    // Hardware not available, return error to trigger software fallback
    return 1;
#endif
}