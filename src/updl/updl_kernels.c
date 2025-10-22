/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file updl_kernels_modular.c
 * @brief Modular UPDL kernels following CMSIS-NN structure
 * 
 * This file provides the main interface functions that use the modular
 * kernel implementations organized by function type, similar to CMSIS-NN.
 * 
 * The implementation is optimized for RISC-V without SIMD, following
 * CMSIS-NN algorithmic strategies:
 * - Im2col + GEMM for Conv2D
 * - Direct convolution for DepthwiseConv2D  
 * - Vector-matrix multiplication for Dense
 * - Register optimization for all operations
 */

#include "updl/updl_kernels.h"
#include "updl/updl_kernels_support.h"
#include "updl/updl_operator.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef MAX
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

// ============================================================================
// LAYER INTERFACE FUNCTIONS (Following original updl_kernels.c pattern)
// ============================================================================

/**
 * @brief Conv1D layer implementation using Conv2D modular functions
 */
uint8_t updl_conv_1d(updl_executor_t *executor, updl_layer_t *layer, updl_exec_layer_t *exec_layer) {
    if (layer->type != Ltype_conv_1d)
        return 1;

    int16_t *weights = layer->weights.weight;
    int16_t *bias = layer->bias.weight;

    // Conv1D: treat as Conv2D with height=1
    return updl_convolve_s16(
                (int16_t *)exec_layer->input_ptr,
                (int16_t *)exec_layer->output_ptr,
                weights, bias,
                1, layer->input_shape[2], layer->input_shape[3],  // height=1, width, channels
                1, layer->output_shape[2], layer->output_shape[3], // height=1, width, channels
                1, layer->kernel_size[1],  // kernel: height=1, width
                layer->strides[1], layer->strides[1], layer->padding,  // stride_x, stride_y, padding
                layer->activation,
                layer->effective_multiplier, layer->effective_shift,
                layer->input_zp, layer->weight_zp, layer->output_zp,
                layer->effective_bias_multiplier, layer->effective_bias_shift);
}

/**
 * @brief Conv2D layer implementation with hardware dispatch
 */
uint8_t updl_conv_2d(updl_executor_t *executor, updl_layer_t *layer, updl_exec_layer_t *exec_layer) {
    if (layer->type != Ltype_conv_2d)
        return 1;

    int16_t *weights = layer->weights.weight;
    int16_t *bias = layer->bias.weight;

    // Try hardware accelerator first
    uint8_t hw_result = updl_conv2d_s16_udl(
        (int16_t *)exec_layer->input_ptr,
        (int16_t *)exec_layer->output_ptr,
        weights, bias,
        layer->input_shape[1], layer->input_shape[2], layer->input_shape[3],
        layer->output_shape[1], layer->output_shape[2], layer->output_shape[3],
        layer->kernel_size[0], layer->kernel_size[1],
        layer->strides[0], layer->strides[1], layer->padding,
        layer->activation,
        layer->effective_multiplier, layer->effective_shift,
        layer->input_zp, layer->weight_zp, layer->output_zp,
        layer->effective_bias_multiplier, layer->effective_bias_shift);
        
    // If hardware succeeded, return success
    if (hw_result == 0) {
        return 0;
    }
    // If hardware failed, continue to software fallback
    updl_Warning("Hardware accelerator unavailable, falling back to software\n");
#ifndef USE_UPDL_UDL_KERNEL

    // Check for optimized kernel 1: 64x1x1x64 -> 64x25x5 (1x1 convolution)
    if (layer->input_shape[1] == 25 && layer->input_shape[2] == 5 && layer->input_shape[3] == 64 &&
        layer->output_shape[1] == 25 && layer->output_shape[2] == 5 && layer->output_shape[3] == 64 &&
        layer->kernel_size[0] == 1 && layer->kernel_size[1] == 1 &&
        layer->strides[0] == 1 && layer->strides[1] == 1) {
        
        // updl_Info("Using optimized kernel: updl_convolve_64x1x1x64_s16_64x25x5\n");
        return updl_convolve_64x1x1x64_s16_64x25x5(
                    (int16_t *)exec_layer->input_ptr,
                    (int16_t *)exec_layer->output_ptr,
                    weights, bias,
                    layer->padding, layer->activation,
                    layer->effective_multiplier, layer->effective_shift,
                    layer->input_zp, layer->weight_zp, layer->output_zp,
                    layer->effective_bias_multiplier, layer->effective_bias_shift);
    }

    // Check for optimized kernel 2: 64x10x4 -> 1x49x10 (large convolution)
    if (layer->input_shape[1] == 49 && layer->input_shape[2] == 10 && layer->input_shape[3] == 1 &&
        layer->output_shape[3] == 64 &&
        layer->kernel_size[0] == 10 && layer->kernel_size[1] == 4) {
        
        // updl_Info("Using optimized kernel: updl_convolve_64x1x10x4_s16_1x49x10\n");
        return updl_convolve_64x1x10x4_s16_1x49x10(
                    (int16_t *)exec_layer->input_ptr,
                    (int16_t *)exec_layer->output_ptr,
                    weights, bias,
                    layer->input_shape[1], layer->input_shape[2], layer->input_shape[3],
                    layer->output_shape[1], layer->output_shape[2], layer->output_shape[3],
                    layer->kernel_size[0], layer->kernel_size[1],
                    layer->strides[0], layer->strides[1], layer->padding,
                    layer->activation,
                    layer->effective_multiplier, layer->effective_shift,
                    layer->input_zp, layer->weight_zp, layer->output_zp,
                    layer->effective_bias_multiplier, layer->effective_bias_shift);
    }
#endif
    // Fallback to general software implementation
    return updl_convolve_s16(
                (int16_t *)exec_layer->input_ptr,
                (int16_t *)exec_layer->output_ptr,
                weights, bias,
                layer->input_shape[1], layer->input_shape[2], layer->input_shape[3],
                layer->output_shape[1], layer->output_shape[2], layer->output_shape[3],
                layer->kernel_size[0], layer->kernel_size[1],
                layer->strides[0], layer->strides[1], layer->padding,
                layer->activation,
                layer->effective_multiplier, layer->effective_shift,
                layer->input_zp, layer->weight_zp, layer->output_zp,
                layer->effective_bias_multiplier, layer->effective_bias_shift);
}


/**
 * @brief DepthwiseConv2D layer implementation with hardware dispatch and optimized kernel selection
 */
uint8_t updl_depthwise_conv_2d(updl_executor_t *executor, updl_layer_t *layer,
                               updl_exec_layer_t *exec_layer) {
    if (layer->type != Ltype_depthwise_conv_2d)
        return 1;

    int16_t *weights = layer->weights.weight;
    int16_t *bias = layer->bias.weight;

    // Try hardware accelerator first
    uint8_t hw_result = updl_depthwise_conv_s16_udl(
        (int16_t *)exec_layer->input_ptr,
        (int16_t *)exec_layer->output_ptr,
        weights, bias,
        layer->input_shape[1], layer->input_shape[2], layer->input_shape[3],
        layer->output_shape[1], layer->output_shape[2], layer->output_shape[3],
        layer->kernel_size[0], layer->kernel_size[1], layer->depth_multiplier,
        layer->strides[0], layer->strides[1], layer->padding,
        layer->activation,
        layer->effective_multiplier, layer->effective_shift,
        layer->input_zp, layer->weight_zp, layer->output_zp,
        layer->effective_bias_multiplier, layer->effective_bias_shift);
        
    // If hardware succeeded, return success
    if (hw_result == 0) {
        return 0;
    }
    // If hardware failed, continue to software fallback
    updl_Warning("Hardware accelerator unavailable, falling back to software\n");

#ifndef USE_UPDL_UDL_KERNEL
    // Check for optimized kernel: 64x25x5 input, 64x3x3 weights, depth_multiplier=1, stride=1
    if (layer->input_shape[1] == 25 && layer->input_shape[2] == 5 && layer->input_shape[3] == 64 &&
        layer->kernel_size[0] == 3 && layer->kernel_size[1] == 3 && 
        layer->depth_multiplier == 1 && layer->strides[0] == 1 && layer->strides[1] == 1) {
        
        updl_Debug("Using optimized kernel: updl_depthwise_conv_64x3x3_s16_64x25x5\n");
        return updl_depthwise_conv_64x3x3_s16_64x25x5(
                    (int16_t *)exec_layer->input_ptr,
                    (int16_t *)exec_layer->output_ptr,
                    weights, bias,
                    layer->padding, layer->activation,
                    layer->effective_multiplier, layer->effective_shift,
                    layer->input_zp, layer->weight_zp, layer->output_zp,
                    layer->effective_bias_multiplier, layer->effective_bias_shift);
    }
#endif

    // Fallback to general depthwise convolution
    updl_Debug("Using general kernel: updl_depthwise_conv_s16\n");
    return updl_depthwise_conv_s16(
                (int16_t *)exec_layer->input_ptr,
                (int16_t *)exec_layer->output_ptr,
                weights, bias,
                layer->input_shape[1], layer->input_shape[2], layer->input_shape[3],
                layer->output_shape[1], layer->output_shape[2], layer->output_shape[3],
                layer->kernel_size[0], layer->kernel_size[1], layer->depth_multiplier,
                layer->strides[0], layer->strides[1], layer->padding,
                layer->activation,
                layer->effective_multiplier, layer->effective_shift,
                layer->input_zp, layer->weight_zp, layer->output_zp,
                layer->effective_bias_multiplier, layer->effective_bias_shift);
}

/**
 * @brief Dense layer implementation with hardware dispatch
 */
uint8_t updl_dense(updl_executor_t *executor, updl_layer_t *layer, updl_exec_layer_t *exec_layer) {
    if (layer->type != Ltype_dense)
        return 1;

    int16_t *weights = layer->weights.weight;
    int16_t *bias = layer->bias.weight;

    // Attempt hardware dispatch
    uint8_t hw_result = updl_fully_connected_s16_udl(
        (int16_t *)exec_layer->input_ptr,
        (int16_t *)exec_layer->output_ptr,
        weights, bias,
        layer->input_shape[1], layer->units,
        layer->activation,
        layer->effective_multiplier, layer->effective_shift,
        layer->input_zp, layer->weight_zp, layer->output_zp,
        layer->effective_bias_multiplier, layer->effective_bias_shift);
        
    // If hardware succeeded, return success
    if (hw_result == 0) {
        return 0;
    }
    // If hardware failed, continue to software fallback
    updl_Warning("Hardware accelerator unavailable, falling back to software\n");

    // Fallback to software implementation
    return updl_fully_connected_s16(
                (int16_t *)exec_layer->input_ptr,
                (int16_t *)exec_layer->output_ptr,
                weights, bias,
                layer->input_shape[1], layer->units,
                layer->activation,
                layer->effective_multiplier, layer->effective_shift,
                layer->input_zp, layer->weight_zp, layer->output_zp,
                layer->effective_bias_multiplier, layer->effective_bias_shift);
}


/**
 * @brief MaxPooling2D layer implementation using modular functions
 */
uint8_t updl_max_pooling_2d(updl_layer_t *layer, updl_exec_layer_t *exec_layer) {
    if (layer->type != Ltype_max_pooling_2d)
        return 1;

    return updl_max_pool_s16((int16_t *)exec_layer->input_ptr,
                            (int16_t *)exec_layer->output_ptr,
                            layer->input_shape[1], layer->input_shape[2], layer->input_shape[3],
                            layer->output_shape[1], layer->output_shape[2], layer->output_shape[3],
                            layer->pool_size[0], layer->pool_size[1],
                            layer->strides[0], layer->strides[1], layer->padding);
}

/**
 * @brief AveragePooling2D layer implementation using modular functions
 */
uint8_t updl_average_pooling_2d(updl_layer_t *layer, updl_exec_layer_t *exec_layer) {
    if (layer->type != Ltype_average_pooling_2d)
        return 1;

    // Use general average pooling implementation
    return updl_avg_pool_s16((int16_t *)exec_layer->input_ptr,
                            (int16_t *)exec_layer->output_ptr,
                            layer->input_shape[1], layer->input_shape[2], layer->input_shape[3],
                            layer->output_shape[1], layer->output_shape[2], layer->output_shape[3],
                            layer->pool_size[0], layer->pool_size[1],
                            layer->strides[0], layer->strides[1], layer->padding,
                            layer->effective_multiplier, layer->effective_shift,
                            layer->input_zp, layer->output_zp);
}

/**
 * @brief L2 normalization layer implementation
 */
uint8_t updl_l2_norm(updl_layer_t *layer, updl_exec_layer_t *exec_layer) {
    if (layer->type != Ltype_lambda)
        return 1;

    // Use global average pooling for L2 norm approximation
    return updl_global_avg_pool_s16((int16_t *)exec_layer->input_ptr,
                                   (int16_t *)exec_layer->output_ptr,
                                   1, layer->input_shape[1], 1);
}

/**
 * @brief Softmax layer implementation using modular functions
 */
uint8_t updl_softmax(updl_executor_t *executor, updl_layer_t *layer, updl_exec_layer_t *exec_layer) {
    if (layer->type != Ltype_softmax)
        return 1;

    // Calculate total number of elements
    uint32_t size = layer->input_shape[1];
    for (uint16_t i = 2; i < 4; i++) {
        if (layer->input_shape[i] > 1) {
            size *= layer->input_shape[i];
        }
    }

    return updl_softmax_s16((int16_t *)exec_layer->input_ptr,
                           (int16_t *)exec_layer->output_ptr,
                           size,
                           layer->effective_multiplier, layer->effective_shift,
                           layer->input_zp, layer->output_zp);
}