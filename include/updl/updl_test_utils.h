/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#ifndef UPDL_TEST_UTILS_H
#define UPDL_TEST_UTILS_H

// ============================================================================
// INCLUDES
// ============================================================================

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// ============================================================================
// QUANTIZATION UTILITIES (Model-agnostic)
// ============================================================================

/**
 * Quantize a single fp32 value to int16
 *
 * Formula: quantized = round(value / scale) + zero_point
 * Clamps result to INT16_MIN..INT16_MAX range
 *
 * @param value        Float32 value to quantize
 * @param scale        Quantization scale factor
 * @param zero_point   Zero point offset
 * @return             Quantized int16 value
 */
int16_t updl_quantize_fp32_to_int16(float value, float scale,
                                    int16_t zero_point);

/**
 * Quantize an array of fp32 values to int16
 *
 * @param fp32_array   Source fp32 array
 * @param int16_array  Destination int16 array
 * @param size         Number of elements
 * @param scale        Quantization scale factor
 * @param zero_point   Zero point offset
 */
void updl_quantize_fp32_array(const float *fp32_array, int16_t *int16_array,
                              size_t size, float scale, int16_t zero_point);

/**
 * Dequantize a single int16 value to fp32
 *
 * Formula: dequantized = (value - zero_point) * scale
 *
 * @param value        Int16 value to dequantize
 * @param scale        Quantization scale factor
 * @param zero_point   Zero point offset
 * @return             Dequantized float32 value
 */
float updl_dequantize_int16_to_fp32(int16_t value, float scale,
                                    int16_t zero_point);

/**
 * Dequantize an array of int16 values to fp32
 *
 * @param int16_array  Source int16 array
 * @param fp32_array   Destination fp32 array
 * @param size         Number of elements
 * @param scale        Quantization scale factor
 * @param zero_point   Zero point offset
 */
void updl_dequantize_int16_array(const int16_t *int16_array, float *fp32_array,
                                 size_t size, float scale, int16_t zero_point);

// ============================================================================
// COMPARISON UTILITIES (Model-agnostic)
// ============================================================================

/**
 * Test metrics structure for comparing arrays
 */
typedef struct {
  float mean_error_rate; // Mean of (test_value - golden_value) / golden_value
  float max_error_rate;  // Maximum absolute error rate
  size_t num_samples;    // Number of samples compared
} updl_test_metrics_t;

/**
 * Compare two fp32 arrays and calculate error rate
 *
 * Calculates error rate as: (test_value - golden_value) / golden_value
 *
 * @param golden_array  Golden (reference) fp32 array
 * @param test_array    Test (actual) fp32 array
 * @param size          Number of elements
 * @return              Calculated error rate metrics
 */
updl_test_metrics_t updl_compare_fp32_arrays(const float *golden_array,
                                             const float *test_array,
                                             size_t size);

#endif // UPDL_TEST_UTILS_H
