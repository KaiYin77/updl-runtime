/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#ifndef UPDL_KERNELS_H
#define UPDL_KERNELS_H

// ============================================================================
// UPDL KERNELS - MODULAR IMPLEMENTATION
// ============================================================================
// This module provides the main interface for neural network layer execution.
// 
// Architecture follows CMSIS-NN structure with modular organization:
// - ConvolutionFunctions: Conv2D, DepthwiseConv2D operations
// - FullyConnectedFunctions: Dense layer operations  
// - PoolingFunctions: MaxPool, AvgPool operations
// - ActivationFunctions: All activation functions
// - NNSupportFunctions: Core computational kernels
//
// Optimizations for RISC-V without SIMD:
// - Im2col + GEMM for Conv2D operations
// - Vector-matrix multiplication for Dense operations  
// - Register optimization and loop unrolling
// - Memory-efficient tiled processing
// ============================================================================

#include <updl/updl_interpreter.h>
#include <updl/updl_utility.h>
#include <updl/updl_kernels_support.h>

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// FORWARD DECLARATIONS
// ============================================================================

typedef struct updl_executor_t updl_executor_t;
typedef struct updl_exec_layer_t updl_exec_layer_t;

// ============================================================================
// HIGH-LEVEL LAYER EXECUTION INTERFACE
// ============================================================================

/**
 * @brief Execute Conv1D layer
 * @param[in] executor   Model executor context
 * @param[in] layer      Layer configuration
 * @param[in] exec_layer Layer execution context
 * @return Status code (0 = success)
 */
uint8_t updl_conv_1d(updl_executor_t *executor, updl_layer_t *layer, 
                     updl_exec_layer_t *exec_layer);

/**
 * @brief Execute Conv2D layer
 * @param[in] executor   Model executor context
 * @param[in] layer      Layer configuration
 * @param[in] exec_layer Layer execution context
 * @return Status code (0 = success)
 */
uint8_t updl_conv_2d(updl_executor_t *executor, updl_layer_t *layer, 
                     updl_exec_layer_t *exec_layer);

/**
 * @brief Execute DepthwiseConv2D layer
 * @param[in] executor   Model executor context
 * @param[in] layer      Layer configuration
 * @param[in] exec_layer Layer execution context
 * @return Status code (0 = success)
 */
uint8_t updl_depthwise_conv_2d(updl_executor_t *executor, updl_layer_t *layer,
                               updl_exec_layer_t *exec_layer);

/**
 * @brief Execute Dense (Fully Connected) layer
 * @param[in] executor   Model executor context
 * @param[in] layer      Layer configuration
 * @param[in] exec_layer Layer execution context
 * @return Status code (0 = success)
 */
uint8_t updl_dense(updl_executor_t *executor, updl_layer_t *layer, 
                   updl_exec_layer_t *exec_layer);

/**
 * @brief Execute MaxPooling2D layer
 * @param[in] layer      Layer configuration
 * @param[in] exec_layer Layer execution context
 * @return Status code (0 = success)
 */
uint8_t updl_max_pooling_2d(updl_layer_t *layer, updl_exec_layer_t *exec_layer);

/**
 * @brief Execute AveragePooling2D layer
 * @param[in] layer      Layer configuration
 * @param[in] exec_layer Layer execution context
 * @return Status code (0 = success)
 */
uint8_t updl_average_pooling_2d(updl_layer_t *layer, updl_exec_layer_t *exec_layer);

/**
 * @brief Execute L2 normalization layer
 * @param[in] layer      Layer configuration
 * @param[in] exec_layer Layer execution context
 * @return Status code (0 = success)
 */
uint8_t updl_l2_norm(updl_layer_t *layer, updl_exec_layer_t *exec_layer);

/**
 * @brief Execute Softmax layer
 * @param[in] executor   Model executor context
 * @param[in] layer      Layer configuration
 * @param[in] exec_layer Layer execution context
 * @return Status code (0 = success)
 */
uint8_t updl_softmax(updl_executor_t *executor, updl_layer_t *layer, 
                     updl_exec_layer_t *exec_layer);

// ============================================================================
// LAYER EXECUTION DISPATCHER
// ============================================================================

/**
 * @brief Generic layer execution dispatcher
 * 
 * Automatically selects the appropriate layer function based on layer type
 * 
 * @param[in] executor   Model executor context
 * @param[in] layer      Layer configuration
 * @param[in] exec_layer Layer execution context
 * @return Status code (0 = success)
 */
uint8_t updl_run_layer(updl_executor_t *executor, updl_layer_t *layer,
                       updl_exec_layer_t *exec_layer);

// ============================================================================
// HARDWARE ACCELERATION INTERFACE
// ============================================================================

/**
 * @brief Hardware dispatch function for Conv2D operations
 * @param[in] input Input tensor data
 * @param[out] output Output tensor data
 * @param[in] weights Weight data
 * @param[in] bias Bias data
 * @param[in] input_height Input height dimension
 * @param[in] input_width Input width dimension
 * @param[in] input_channels Input channel dimension
 * @param[in] output_height Output height dimension
 * @param[in] output_width Output width dimension
 * @param[in] output_channels Output channel dimension
 * @param[in] kernel_height Kernel height
 * @param[in] kernel_width Kernel width
 * @param[in] stride_x Stride in x direction
 * @param[in] stride_y Stride in y direction
 * @param[in] padding Padding type
 * @param[in] activation Activation function
 * @param[in] eff_multiplier Effective multiplier for quantization
 * @param[in] eff_shift Effective shift for quantization
 * @param[in] input_zp Input zero point
 * @param[in] weight_zp Weight zero point
 * @param[in] output_zp Output zero point
 * @param[in] eff_bias_multiplier Effective bias multiplier
 * @param[in] eff_bias_shift Effective bias shift
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
                                int32_t eff_bias_multiplier, int16_t eff_bias_shift);

/**
 * @brief Hardware dispatch function for DepthwiseConv2D operations
 * @param[in] input Input tensor data
 * @param[out] output Output tensor data
 * @param[in] weights Weight data (depthwise format)
 * @param[in] bias Bias data
 * @param[in] input_height Input height dimension
 * @param[in] input_width Input width dimension
 * @param[in] input_channels Input channel dimension
 * @param[in] output_height Output height dimension
 * @param[in] output_width Output width dimension
 * @param[in] output_channels Output channel dimension
 * @param[in] kernel_height Kernel height
 * @param[in] kernel_width Kernel width
 * @param[in] depth_multiplier Depth multiplier (typically 1)
 * @param[in] stride_x Stride in x direction
 * @param[in] stride_y Stride in y direction
 * @param[in] padding Padding type
 * @param[in] activation Activation function
 * @param[in] eff_multiplier Effective multiplier for quantization
 * @param[in] eff_shift Effective shift for quantization
 * @param[in] input_zp Input zero point
 * @param[in] weight_zp Weight zero point
 * @param[in] output_zp Output zero point
 * @param[in] eff_bias_multiplier Effective bias multiplier
 * @param[in] eff_bias_shift Effective bias shift
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
                                    int32_t eff_bias_multiplier, int16_t eff_bias_shift);

/**
 * @brief Hardware dispatch function for Dense operations
 * @param[in] input Input tensor data
 * @param[out] output Output tensor data
 * @param[in] weights Weight data
 * @param[in] bias Bias data
 * @param[in] input_features Number of input features
 * @param[in] output_features Number of output features (units)
 * @param[in] activation Activation function
 * @param[in] eff_multiplier Effective multiplier for quantization
 * @param[in] eff_shift Effective shift for quantization
 * @param[in] input_zp Input zero point
 * @param[in] weight_zp Weight zero point
 * @param[in] output_zp Output zero point
 * @param[in] eff_bias_multiplier Effective bias multiplier
 * @param[in] eff_bias_shift Effective bias shift
 * @return 0 on success, non-zero on error
 */
uint8_t updl_fully_connected_s16_udl(int16_t *input, int16_t *output, int16_t *weights, int16_t *bias,
                               uint32_t input_features, uint32_t output_features,
                               atype_t activation,
                               int32_t eff_multiplier, int16_t eff_shift,
                               int16_t input_zp, int16_t weight_zp, int16_t output_zp,
                               int32_t eff_bias_multiplier, int16_t eff_bias_shift);


#ifdef __cplusplus
}
#endif

#endif // UPDL_KERNELS_H