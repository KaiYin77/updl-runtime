/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#ifndef UPDL_KERNELS_SUPPORT_H
#define UPDL_KERNELS_SUPPORT_H

/*
 * UPDL Quantization Mode Configuration
 * ====================================
 * 
 * This library supports two quantization modes via compile-time macros:
 * 
 * 1. UDL Hardware Mode (UPDL_UDL_SHIFT_ONLY_MODE defined):
 *    - Forces all multipliers to 1
 *    - Uses shift-only quantization for UDL hardware compatibility  
 *    - Optimal for embedded UDL accelerators
 *    
 * 2. TensorFlow Lite Mode (default, UPDL_UDL_SHIFT_ONLY_MODE not defined):
 *    - Uses full multiplier + shift quantization
 *    - Compatible with TensorFlow Lite quantization schemes
 *    - Higher precision but requires multiplier support
 * 
 * Usage:
 *   // For UDL hardware mode:
 *   #define UPDL_UDL_SHIFT_ONLY_MODE
 *   
 *   // For TensorFlow Lite mode (default):
 *   // (no macro definition needed)
 */

#include "updl/updl_operator.h"
#include <math.h>
#define UPDL_UDL_SHIFT_ONLY_MODE 1
#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// ACTIVATION FUNCTIONS
// ============================================================================

/**
 * @brief Apply activation function to input value
 * @param[in] input     Input value
 * @param[in] activation Activation type
 * @return Activated value
 */
int32_t updl_activation(int32_t input, atype_t activation);

// ============================================================================
// CONVOLUTION FUNCTION DECLARATIONS
// ============================================================================

// General convolution
uint8_t updl_convolve_s16(
    int16_t *input, int16_t *output, int16_t *weights, int16_t *bias,
    uint32_t input_height, uint32_t input_width, uint32_t input_channels,
    uint32_t output_height, uint32_t output_width, uint32_t output_channels,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t stride_x,
    uint32_t stride_y, ptype_t padding, atype_t activation,
    int32_t eff_multiplier, int16_t eff_shift, int16_t input_zp,
    int16_t weight_zp, int16_t output_zp, int32_t eff_bias_multiplier, int16_t eff_bias_shift);

// Optimized 1x1 convolution for 64x25x5 input (CHW) with 64x64x1x1 weights (OIHW)
uint8_t updl_convolve_64x1x1x64_s16_64x25x5(int16_t *input, int16_t *output,
                                      int16_t *weights, int16_t *bias,
                                      ptype_t padding, atype_t activation,
                                      int32_t eff_multiplier, int16_t eff_shift,
                                      int16_t input_zp, int16_t weight_zp,
                                      int16_t output_zp, int32_t eff_bias_multiplier, int16_t eff_bias_shift);

// Optimized convolution for 1x49x10 input (CHW) with 64x1x10x4 weights (OIHW)
uint8_t updl_convolve_64x1x10x4_s16_1x49x10(int16_t *input, int16_t *output,
                                  int16_t *weights, int16_t *bias,
                                  uint32_t input_height, uint32_t input_width, uint32_t input_channels,
                                  uint32_t output_height, uint32_t output_width, uint32_t output_channels,
                                  uint32_t kernel_height, uint32_t kernel_width,
                                  uint32_t stride_x, uint32_t stride_y, ptype_t padding,
                                  atype_t activation, int32_t eff_multiplier,
                                  int16_t eff_shift, int16_t input_zp,
                                  int16_t weight_zp, int16_t output_zp,
                                  int32_t eff_bias_multiplier, int16_t eff_bias_shift);
// General depthwise convolution
uint8_t updl_depthwise_conv_s16(
    int16_t *input, int16_t *output, int16_t *weights, int16_t *bias,
    uint32_t input_height, uint32_t input_width, uint32_t input_channels,
    uint32_t output_height, uint32_t output_width, uint32_t output_channels,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t depth_multiplier,
    uint32_t stride_x, uint32_t stride_y, ptype_t padding, atype_t activation,
    int32_t eff_multiplier, int16_t eff_shift, int16_t input_zp,
    int16_t weight_zp, int16_t output_zp, int32_t eff_bias_multiplier, int16_t eff_bias_shift);
// Optimized depthwise convolution for 64x25x5 input (CHW) with 64x3x3 weights (depthwise format)
uint8_t updl_depthwise_conv_64x3x3_s16_64x25x5(int16_t *input, int16_t *output,
                                        int16_t *weights, int16_t *bias,
                                        ptype_t padding, atype_t activation,
                                        int32_t eff_multiplier,
                                        int16_t eff_shift, int16_t input_zp,
                                        int16_t weight_zp, int16_t output_zp,
                                        int32_t eff_bias_multiplier, int16_t eff_bias_shift);

// ============================================================================
// FULLY CONNECTED FUNCTION DECLARATIONS
// ============================================================================

uint8_t updl_fully_connected_s16(int16_t *input, int16_t *output,
                                 int16_t *weights, int16_t *bias,
                                 uint32_t input_features, uint32_t output_features,
                                 atype_t activation,
                                 int32_t eff_multiplier, int16_t eff_shift,
                                 int16_t input_zp, int16_t weight_zp,
                                 int16_t output_zp, int32_t eff_bias_multiplier, int16_t eff_bias_shift);

// ============================================================================
// POOLING FUNCTION DECLARATIONS
// ============================================================================

uint8_t updl_max_pool_s16(int16_t *input, int16_t *output,
                          uint32_t input_height, uint32_t input_width,
                          uint32_t input_channels, uint32_t output_height,
                          uint32_t output_width, uint32_t output_channels,
                          uint32_t kernel_height, uint32_t kernel_width,
                          uint32_t stride_x, uint32_t stride_y,
                          ptype_t padding);

uint8_t updl_avg_pool_s16(int16_t *input, int16_t *output,
                          uint32_t input_height, uint32_t input_width,
                          uint32_t input_channels, uint32_t output_height,
                          uint32_t output_width, uint32_t output_channels,
                          uint32_t kernel_height, uint32_t kernel_width,
                          uint32_t stride_x, uint32_t stride_y, ptype_t padding,
                          int32_t eff_multiplier, int16_t eff_shift,
                          int16_t input_zp, int16_t output_zp);

uint8_t updl_global_avg_pool_s16(int16_t *input, int16_t *output,
                                 uint32_t input_height, uint32_t input_width,
                                 uint32_t input_channels);

// ============================================================================
// SOFTMAX FUNCTION DECLARATIONS
// ============================================================================

uint8_t updl_softmax_s16(int16_t *input, int16_t *output, uint32_t size,
                         int32_t eff_multiplier, int16_t eff_shift,
                         int16_t input_zp, int16_t output_zp);

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Clamp int32 value to int16 range
 * @param[in] input Input value
 * @return Clamped int16 value
 */
static inline int16_t updl_clamp_s16(int32_t input) {
  if (input > INT16_MAX)
    return INT16_MAX;
  if (input < INT16_MIN)
    return INT16_MIN;
  return (int16_t)input;
}

/**
 * @brief Clamp int64 value to int32 range
 * @param[in] input Input value
 * @return Clamped int32 value
 */
static inline int32_t updl_clamp_s32(int64_t input) {
  if (input > INT32_MAX)
    return INT32_MAX;
  if (input < INT32_MIN)
    return INT32_MIN;
  return (int32_t)input;
}

// Requantization function supporting both UDL and TensorFlow Lite modes
static inline int32_t updl_requantize(int32_t value, int32_t multiplier,
                                      int16_t shift) {
  int64_t result = (int64_t)value;
  
#ifdef UPDL_UDL_SHIFT_ONLY_MODE
  // UDL hardware mode: multiplier should always be 1, use shift-only
  // This simplifies to: output = input >> shift (for positive shift)
  //                     output = input << (-shift) (for negative shift)
  
  // In UDL mode, multiplier must always be 1 for pure shift-only operation
  // Any non-1 multiplier indicates a configuration error that would cause inference errors
  (void)multiplier;  // Explicitly ignore multiplier parameter in UDL mode
  
#else
  // TensorFlow Lite mode: use full multiplier + shift quantization
  
  // Multiply in 64-bit to avoid overflow
  result *= (int64_t)multiplier;
  
  // Normalize from Q30 back to int32 (matches 30-bit multiplier precision)
  result >>= 30;
  
#endif

  // Shift handling: positive = right shift, negative = left shift
  // (Same for both modes)
  if (shift > 0) {
    // Right shift with rounding
    int64_t rounding = (int64_t)1 << (shift - 1);
    result += rounding;
    result >>= shift;
  } else if (shift < 0) {
    // Left shift
    int16_t left_shift = -shift;
    if (left_shift < 32) {
      result <<= left_shift;
    }
  }

  // Clamp to int32 range
  if (result > INT32_MAX)
    return INT32_MAX;
  if (result < INT32_MIN)
    return INT32_MIN;

  return (int32_t)result;
}

/**
 * @brief Clamp int64 to int32 range with dynamic scaling and compensation
 * @param[in] sum Input int64 value
 * @param[out] dynamic_shift Number of right shifts applied  
 * @return Clamped int32 value
 */
static inline int32_t updl_clamp_s32_with_scaling(int64_t sum, int16_t *dynamic_shift) {
  *dynamic_shift = 0;
  
  if (sum > INT32_MAX) {
    int64_t temp = sum;
    while (temp > INT32_MAX) {
      temp >>= 1;
      (*dynamic_shift)++;
    }
    return (int32_t)temp;
  } else if (sum < INT32_MIN) {
    int64_t temp = sum;
    while (temp < INT32_MIN) {
      temp >>= 1;
      (*dynamic_shift)++;
    }
    return (int32_t)temp;
  }
  
  return (int32_t)sum;
}

/**
 * @brief Scale bias value from bias_scale to accumulator scale (input_scale * weight_scale)
 * Supports both UDL (shift-only) and TensorFlow Lite (multiplier+shift) modes
 * @param[in] bias_value    Input bias value (int16)
 * @param[in] multiplier    Scaling multiplier (1 for UDL, 16-bit precision for TFLite)
 * @param[in] shift         Scaling shift
 * @return Scaled bias value for accumulator
 */
static inline int64_t updl_scale_bias(int16_t bias_value, int32_t multiplier, int16_t shift) {
    int64_t scaled_bias = (int64_t)bias_value;
    
#ifdef UPDL_UDL_SHIFT_ONLY_MODE
    // UDL mode: multiplier must always be 1, use pure shift-only scaling
    // Any non-1 multiplier indicates configuration error that causes inference errors
    (void)multiplier;  // Explicitly ignore multiplier parameter in UDL mode
    
#else
    // TensorFlow Lite mode: use 16-bit precision multiplier
    
    // Scale bias from bias_scale to accumulator scale (input_scale * weight_scale)
    scaled_bias *= (int64_t)multiplier;
    
    // Normalize from 16-bit precision (divide by 65536)
    scaled_bias >>= 16;
    
#endif
    
    // Apply shift: positive = right shift, negative = left shift  
    // (Same for both modes)
    if (shift > 0) {
        // Right shift with rounding
        int64_t rounding = (int64_t)1 << (shift - 1);
        scaled_bias += rounding;
        scaled_bias >>= shift;
    } else if (shift < 0) {
        // Left shift
        int16_t left_shift = -shift;
        if (left_shift < 32) {
            scaled_bias <<= left_shift;
        }
    }
    
    return scaled_bias;
}

// Structure to hold multiplier and shift for precise quantization
typedef struct {
    int32_t multiplier;
    int16_t shift;
} requant_params_t;

// Convert TensorFlow Lite scale to multiplier + shift for precise quantization
// Supports both UDL (shift-only) and TFLite (multiplier+shift) modes via compile-time macro
static inline requant_params_t updl_scale_to_multiplier_shift(float scale) {
    requant_params_t params = {0, 0};
    
    if (scale <= 0.0f) {
        return params; // Invalid scale
    }
    
#ifdef UPDL_UDL_SHIFT_ONLY_MODE
    // UDL-optimized: Force multiplier=1, handle all scaling through shift only
    params.multiplier = 1; // Force multiplier = 1 for UDL compatibility
    
    // UDL hardware only supports shift-based requantization
    // Convert scale to shift-only representation: output = input >> shift
    // For scale S, we want: output = input * S = input >> (-log2(S))
    
    // Improved power-of-2 approximation with error minimization
    float log2_scale = log2f(scale);
    
    // Try both floor and ceil to find better approximation
    int16_t shift_floor = (int16_t)floorf(-log2_scale);
    int16_t shift_ceil = (int16_t)ceilf(-log2_scale);
    
    // Calculate errors for both options
    float scale_floor = (shift_floor >= 0) ? (1.0f / (1 << shift_floor)) : (1 << (-shift_floor));
    float scale_ceil = (shift_ceil >= 0) ? (1.0f / (1 << shift_ceil)) : (1 << (-shift_ceil));
    
    float error_floor = fabsf(scale_floor - scale) / scale;
    float error_ceil = fabsf(scale_ceil - scale) / scale;
    
    // Choose the option with smaller error
    int16_t calculated_shift = (error_floor < error_ceil) ? shift_floor : shift_ceil;
    
    // Clamp shift to UDL hardware limits (5-bit: 0x1F = 31)
    const int16_t UDL_MIN_SHIFT = -31;
    const int16_t UDL_MAX_SHIFT = 31;
    
    if (calculated_shift < UDL_MIN_SHIFT) {
        params.shift = UDL_MIN_SHIFT;
    } else if (calculated_shift > UDL_MAX_SHIFT) {
        params.shift = UDL_MAX_SHIFT;
    } else {
        params.shift = calculated_shift;
    }
    
#else
    // TensorFlow Lite compatible mode: Use multiplier + shift approach
    
    // Handle scales >= 1.0 (left shift case)
    if (scale >= 1.0f) {
        // For scales >= 1.0, we use left shift (negative shift value)
        int16_t left_shift = 0;
        float temp_scale = scale;
        while (temp_scale >= 2.0f && left_shift < 15) {
            temp_scale /= 2.0f;
            left_shift++;
        }
        params.multiplier = (int32_t)(temp_scale * (1 << 30) + 0.5f); // 30-bit precision
        params.shift = -left_shift; // Negative means left shift
        return params;
    }
    
    // Handle scales < 1.0 (right shift case) - TensorFlow Lite method
    int16_t shift = 0;
    float normalized_scale = scale;
    
    // Normalize scale to be in range [0.5, 1.0)
    while (normalized_scale < 0.5f && shift < 31) {
        normalized_scale *= 2.0f;
        shift++;
    }
    
    // Convert to 30-bit integer multiplier
    // Using 30 bits to avoid overflow in (input * multiplier)
    params.multiplier = (int32_t)(normalized_scale * (1LL << 30) + 0.5f);
    params.shift = shift;
    
    // Clamp multiplier to valid range
    if (params.multiplier >= (1LL << 31)) {
        params.multiplier = (1LL << 31) - 1;
    }
#endif
    
    return params;
}

// Convert bias scale to multiplier + shift for bias scaling (without Q30 normalization)
// Supports both UDL (shift-only) and TFLite (multiplier+shift) modes via compile-time macro
static inline requant_params_t updl_bias_scale_to_multiplier_shift(float scale) {
    requant_params_t params = {0, 0};
    
    if (scale <= 0.0f) {
        return params; // Invalid scale
    }
    
#ifdef UPDL_UDL_SHIFT_ONLY_MODE
    // UDL-optimized: Force multiplier=1, handle all scaling through shift only
    params.multiplier = 1; // Force multiplier = 1 for UDL compatibility
    
    // UDL hardware only supports shift-based bias scaling
    // Convert scale to shift-only representation for bias scaling
    // For bias scale S, we want: scaled_bias = bias * S = bias >> (-log2(S))
    
    // Improved power-of-2 approximation with error minimization (same as above)
    float log2_scale = log2f(scale);
    
    // Try both floor and ceil to find better approximation
    int16_t shift_floor = (int16_t)floorf(-log2_scale);
    int16_t shift_ceil = (int16_t)ceilf(-log2_scale);
    
    // Calculate errors for both options
    float scale_floor = (shift_floor >= 0) ? (1.0f / (1 << shift_floor)) : (1 << (-shift_floor));
    float scale_ceil = (shift_ceil >= 0) ? (1.0f / (1 << shift_ceil)) : (1 << (-shift_ceil));
    
    float error_floor = fabsf(scale_floor - scale) / scale;
    float error_ceil = fabsf(scale_ceil - scale) / scale;
    
    // Choose the option with smaller error
    int16_t calculated_shift = (error_floor < error_ceil) ? shift_floor : shift_ceil;
    
    // Clamp shift to UDL hardware limits for bias operations
    const int16_t UDL_MIN_BIAS_SHIFT = -15;
    const int16_t UDL_MAX_BIAS_SHIFT = 15;
    
    if (calculated_shift < UDL_MIN_BIAS_SHIFT) {
        params.shift = UDL_MIN_BIAS_SHIFT;
    } else if (calculated_shift > UDL_MAX_BIAS_SHIFT) {
        params.shift = UDL_MAX_BIAS_SHIFT;
    } else {
        params.shift = calculated_shift;
    }
    
#else
    // TensorFlow Lite compatible mode: Use multiplier + shift approach for bias scaling
    
    // For bias scaling, we want direct scaling without Q30 normalization
    // Find the best representation as: scaled_value = (value * multiplier) >> shift
    
    if (scale >= 1.0f) {
        // Large scale factors: use smaller multiplier with negative shift (left shift)
        int16_t left_shift = 0;
        float temp_scale = scale;
        
        // Normalize scale to reasonable range [1.0, 2.0) for 16-bit precision
        while (temp_scale >= 2.0f && left_shift < 15) {
            temp_scale /= 2.0f;
            left_shift++;
        }
        
        // Use 16-bit precision multiplier (not Q30)
        params.multiplier = (int32_t)(temp_scale * 65536.0f + 0.5f); // 16-bit precision
        params.shift = -left_shift; // Negative means left shift in updl_scale_bias
        
    } else {
        // Small scale factors: use larger multiplier with positive shift (right shift)
        int16_t right_shift = 0;
        float temp_scale = scale;
        
        // Normalize scale to range [0.5, 1.0) 
        while (temp_scale < 0.5f && right_shift < 15) {
            temp_scale *= 2.0f;
            right_shift++;
        }
        
        // Use 16-bit precision multiplier
        params.multiplier = (int32_t)(temp_scale * 65536.0f + 0.5f);
        params.shift = right_shift; // Positive means right shift
    }
#endif
    
    return params;
}

/**
 * @brief Optimized pipeline combining 5 frequently-called functions
 * 
 * Combines: updl_clamp_s32 -> updl_activation -> updl_requantize -> output_zp -> updl_clamp_s16
 * This replaces 5 function calls with a single optimized inline operation.
 * 
 * @param[in] sum              64-bit accumulator value
 * @param[in] activation       Activation type
 * @param[in] eff_multiplier   Quantization multiplier
 * @param[in] eff_shift        Quantization shift
 * @param[in] output_zp        Output zero-point
 * @return Final int16 output value
 */
static inline int16_t updl_quantize_pipeline(int64_t sum, atype_t activation,
                                            int32_t eff_multiplier, int16_t eff_shift,
                                            int16_t output_zp) {
    // Step 1: Clamp int64 to int32 (updl_clamp_s32)
    int32_t clamped;
    if (sum > INT32_MAX) {
        clamped = INT32_MAX;
    } else if (sum < INT32_MIN) {
        clamped = INT32_MIN;
    } else {
        clamped = (int32_t)sum;
    }
    
    // Step 2: Apply activation (updl_activation) - inline common cases
    int32_t activated;
    switch (activation) {
        case Atype_none:
        case Atype_linear:
            activated = clamped;
            break;
        case Atype_relu:
            activated = (clamped > 0) ? clamped : 0;
            break;
        case Atype_leakyrelu:
            activated = (clamped > 0) ? clamped : (clamped >> 3); // leaky factor ~0.125
            break;
        default:
            // For complex activations, fall back to function call
            activated = updl_activation(clamped, activation);
            break;
    }
    
    // Step 3: Requantize (updl_requantize) - inline for common cases
    int32_t requantized;
    int64_t result = (int64_t)activated;
    
#ifdef UPDL_UDL_SHIFT_ONLY_MODE
    // UDL mode: multiplier must always be 1 for pure shift-only operation
    // Any non-1 multiplier indicates configuration error that causes inference errors
    (void)eff_multiplier;  // Explicitly ignore multiplier parameter in UDL mode
#else
    // TensorFlow Lite mode: multiply and normalize
    result *= (int64_t)eff_multiplier;
    result >>= 30;
#endif
    
    // Apply shift with rounding
    if (eff_shift > 0) {
        int64_t rounding = (int64_t)1 << (eff_shift - 1);
        result += rounding;
        result >>= eff_shift;
    } else if (eff_shift < 0) {
        int16_t left_shift = -eff_shift;
        if (left_shift < 32) {
            result <<= left_shift;
        }
    }
    
    // Clamp requantized result to int32
    if (result > INT32_MAX) {
        requantized = INT32_MAX;
    } else if (result < INT32_MIN) {
        requantized = INT32_MIN;
    } else {
        requantized = (int32_t)result;
    }
    
    // Step 4: Add output zero-point
    int32_t with_zp = requantized + (int32_t)output_zp;
    
    // Step 5: Final clamp to int16 (updl_clamp_s16)
    if (with_zp > INT16_MAX) {
        return INT16_MAX;
    } else if (with_zp < INT16_MIN) {
        return INT16_MIN;
    } else {
        return (int16_t)with_zp;
    }
}

/**
 * @brief Calculate padding for SAME padding mode
 * @param[in]  input_size    Input dimension size
 * @param[in]  kernel_size   Kernel dimension size
 * @param[in]  stride        Stride
 * @param[out] pad_before    Padding before
 * @param[out] pad_after     Padding after
 */
void updl_calculate_padding(const uint32_t input_size,
                            const uint32_t kernel_size, const uint32_t stride,
                            uint32_t *pad_before, uint32_t *pad_after);

#ifdef __cplusplus
}
#endif

#endif // UPDL_KERNELS_SUPPORT_H