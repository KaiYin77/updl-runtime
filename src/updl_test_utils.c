/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

// ============================================================================
// INCLUDES
// ============================================================================

#include <updl/updl_test_utils.h>

#include <math.h>
#include <stdio.h>

// ============================================================================
// QUANTIZATION FUNCTIONS
// ============================================================================

int16_t updl_quantize_fp32_to_int16(float value, float scale,
                                    int16_t zero_point) {
  // Formula: quantized = round(value / scale) + zero_point
  float scaled = roundf(value / scale) + (float)zero_point;

  // Clamp to int16 range
  if (scaled > (float)INT16_MAX) {
    return INT16_MAX;
  } else if (scaled < (float)INT16_MIN) {
    return INT16_MIN;
  }

  return (int16_t)scaled;
}

void updl_quantize_fp32_array(const float *fp32_array, int16_t *int16_array,
                              size_t size, float scale, int16_t zero_point) {
  if (!fp32_array || !int16_array || size == 0) {
    return;
  }

  for (size_t i = 0; i < size; i++) {
    int16_array[i] =
        updl_quantize_fp32_to_int16(fp32_array[i], scale, zero_point);
  }
}

// ============================================================================
// DEQUANTIZATION FUNCTIONS
// ============================================================================

float updl_dequantize_int16_to_fp32(int16_t value, float scale,
                                    int16_t zero_point) {
  return (float)(value - zero_point) * scale;
}

void updl_dequantize_int16_array(const int16_t *int16_array, float *fp32_array,
                                 size_t size, float scale, int16_t zero_point) {
  if (!int16_array || !fp32_array || size == 0) {
    return;
  }

  for (size_t i = 0; i < size; i++) {
    fp32_array[i] =
        updl_dequantize_int16_to_fp32(int16_array[i], scale, zero_point);
  }
}

// ============================================================================
// COMPARISON FUNCTIONS
// ============================================================================

updl_test_metrics_t updl_compare_fp32_arrays(const float *golden_array,
                                             const float *test_array,
                                             size_t size) {
  updl_test_metrics_t metrics = {0};

  if (!golden_array || !test_array || size == 0) {
    return metrics;
  }

  metrics.num_samples = size;
  float sum_error_rate = 0.0f;
  float max_error_rate = 0.0f;

  for (size_t i = 0; i < size; i++) {
    float golden = golden_array[i];
    float test = test_array[i];

    // Skip if golden value is zero (avoid division by zero)
    if (fabsf(golden) < 1e-10f) {
      continue;
    }

    // Calculate error rate: (test - golden) / golden
    float error_rate = (test - golden) / golden;
    float abs_error_rate = fabsf(error_rate);

    sum_error_rate += error_rate;

    if (abs_error_rate > max_error_rate) {
      max_error_rate = abs_error_rate;
    }
  }

  metrics.mean_error_rate = sum_error_rate / (float)size;
  metrics.max_error_rate = max_error_rate;

  return metrics;
}