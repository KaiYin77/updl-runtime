/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file updl_fully_connected_s16_udl.c
 * @brief Hardware dispatch functions for Dense (Fully Connected) operations
 * 
 * This file provides hardware acceleration hooks for Dense/FC operations,
 * interfacing with the UDL (Upbeat Deep Learning) hardware accelerator
 * based on the reference test_ai_accelerator.c implementation.
 */

#include "updl/updl_kernels.h"
#include "updl/updl_kernels_support.h"
#include "updl/updl_operator.h"

// Hardware accelerator includes (conditional compilation)
#ifdef USE_UPDL_UDL_KERNEL
#include "metal.h"
#include "metal-platform.h"
#include "metal/udl.h"
#include "metal/upt_isr.h"
#endif

/**
 * @brief Hardware dispatch function for Dense operations
 * 
 * @param input Input tensor data
 * @param output Output tensor data  
 * @param weights Weight data
 * @param bias Bias data
 * @param input_features Number of input features
 * @param output_features Number of output features (units)
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
uint8_t updl_fully_connected_s16_udl(int16_t *input, int16_t *output, int16_t *weights, int16_t *bias,
                               uint32_t input_features, uint32_t output_features,
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
        return 1; // Hardware not available, fallback to software
    }
    
    // Reset UDL module
    metal_udl_reset(udl_device);
    
    // Setup UPT-based convolution parameters (reused for dense operations)
    upt_nn_conv_params_Type upt_conv_params;
    
    // Configure convolution parameters for dense layer
    // Dense layer uses 1x1 convolution with appropriate stride settings
    upt_conv_params.stride.w = 1;
    upt_conv_params.stride.h = 1;
    
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
    
    // Call hardware fully connected function
    enum_udl_retcode_Type result = metal_udl_fully_connect_upt_api(
        udl_device,
        &upt_conv_params,
        input_features,
        (uint32_t *)input,
        (uint32_t *)weights,
        (uint32_t *)bias,
        output_features,
        upt_output_ref);
    
    if (result != E_UDL_SUCCESS) {
        return 1; // Hardware execution failed
    }
    
    // Wait for hardware completion (polling mode)
    while (!metal_udl_read_pending(udl_device));
    
    // Handle memory rearrangement if needed (workaround for memory wiring issue)
    const uint32_t DEFAULT_OUTPUT_ADDR = 0x50000000;
    metal_udl_rearrange(udl_device, (uint32_t *)DEFAULT_OUTPUT_ADDR, output_features);
    
    // Copy results back to output buffer
    memcpy(output, (void *)DEFAULT_OUTPUT_ADDR, output_features * sizeof(int16_t));
    
    return 0; // Success
    
#else
    // Hardware not available, return error to trigger software fallback
    return 1;
#endif
}