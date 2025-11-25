/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#ifndef UPDL_TEST_RUNNER_UTILS_H
#define UPDL_TEST_RUNNER_UTILS_H

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
 * Reports distribution of error bits across samples
 */
typedef struct {
  size_t num_samples;       // Total number of samples compared
  size_t matched;           // Count of exact matches (0 bits difference)
  size_t diff_1bit;         // Count of 1-bit differences
  size_t diff_2plus_bits;   // Count of 2+ bits differences
  int16_t max_error_bits;   // Maximum absolute error in bits
} updl_test_metrics_t;

#include <updl/updl_operator.h>

// ============================================================================
// EXECUTION UTILITIES
// ============================================================================

/**
 * Get string representation of layer type
 */
const char *updl_get_layer_type_name(ltype_t type);

/**
 * Execute a single layer in isolation
 *
 * @param executor     UPDL executor
 * @param layer_index  Index of layer to execute
 * @param input_int16  Int16 input buffer
 * @param output_int16 Int16 output buffer
 * @return 0 on success, -1 on failure
 */
int updl_execute_single_layer(updl_executor_t *executor, uint16_t layer_idx,
                              const int16_t *input_int16,
                              int16_t *output_int16);

// ============================================================================
// VALIDATION UTILITIES
// ============================================================================

/**
 * Validate layer output against golden reference (FP32)
 *
 * Compares actual int16 output with golden FP32 values.
 * Simulates int16 quantization on golden values for fair comparison.
 * Reports distribution: matched, 1-bit diff, 2+ bits diff
 *
 * @param actual         Actual int16 output from layer
 * @param golden_fp32    Expected FP32 output
 * @param size           Number of elements
 * @param scale          Layer output scale
 * @param verbose        Print detailed errors
 * @param layer_name     Layer name for logging
 * @param metrics        Optional: Output metrics (distribution counts), can be NULL
 * @return true if all features are exact match or 1-bit difference
 */
bool updl_validate_layer_output(const int16_t *actual, const float *golden_fp32,
                                size_t size, float scale,
                                bool verbose, const char *layer_name,
                                updl_test_metrics_t *metrics);

/**
 * Compare two int16 buffers for exact match (or close match)
 *
 * Used for comparing HW vs SW implementations.
 *
 * @param layer_name     Layer name for logging
 * @param layer_idx      Layer index
 * @param buffer1        First buffer (e.g., HW output)
 * @param buffer2        Second buffer (e.g., SW output)
 * @param size           Number of elements
 * @param matched        Output: incremented if buffers match
 * @param mismatched     Output: incremented if buffers mismatch
 */
void updl_compare_int16_buffers(const char *layer_name, uint16_t layer_idx,
                                const int16_t *buffer1, const int16_t *buffer2,
                                size_t size, uint32_t *matched,
                                uint32_t *mismatched);



// ============================================================================
// UNIFIED TEST STRUCTURES
// ============================================================================

/**
 * Layer golden reference structure (Unified)
 * Used for both isolation and propagation tests
 */
typedef struct {
  const char *layer_name;   // Layer name (e.g., "activation", "dense")
  uint16_t layer_index;     // Layer index in model
  
  // Input golden reference (FP32) - Optional, used for isolation tests
  const float *input_golden_fp32; 
  size_t input_size;

  // Output golden reference (FP32)
  const float *output_golden_fp32; // Expected FP32 output
  size_t output_size;              // Output size
} updl_test_layer_golden_t;

/**
 * Test sample structure (Unified)
 */
typedef struct {
  const float *input_fp32; // FP32 input data (Model input)
  size_t input_size;       // Input size
  
  const updl_test_layer_golden_t *layers; // Array of layers to validate
  size_t num_layers;                      // Number of layers to validate
} updl_test_sample_t;

/**
 * Unified Test configuration
 */
typedef struct {
  const updl_test_sample_t *samples; // Array of test samples
  size_t num_samples;                // Number of test samples

  updl_model_t *model;       // UPDL model to test
  updl_executor_t *executor; // UPDL executor to use

  bool verbose; // Print detailed metrics

  // Buffer configuration
  // For Quant/Isolation tests: need input, output, and golden buffers (int16)
  // For Prop/Inference tests: need dequant buffer (float)
  
  int16_t *int16_input_buffer;     // Reusable input buffer
  size_t int16_input_buffer_size;
  
  int16_t *int16_output_buffer;    // Reusable output buffer
  size_t int16_output_buffer_size;
  
  float *dequant_buffer;           // Reusable dequantization buffer
  size_t dequant_buffer_size;
} updl_test_config_t;

/**
 * Per-layer test result (Unified)
 */
typedef struct {
  const char *layer_name;
  uint16_t layer_index;
  bool passed;
  
  updl_test_metrics_t metrics;

  // Feature-level statistics
  size_t total_features;
  size_t features_passed;
  size_t features_failed;
} updl_test_layer_result_t;

/**
 * Per-sample test result (Unified)
 */
typedef struct {
  uint32_t sample_index;
  updl_test_layer_result_t *layer_results; // Array of layer results
  size_t num_layers;
  uint32_t layers_passed;
  uint32_t layers_failed;
} updl_test_sample_result_t;

/**
 * Overall test report (Unified)
 */
typedef struct {
  updl_test_sample_result_t *sample_results; // Array of sample results
  size_t num_samples;
} updl_test_report_t;

// ============================================================================
// SHARED RUNNER FUNCTIONS
// ============================================================================

/**
 * Run a single layer isolation test
 * 
 * @param config       Test configuration
 * @param layer_golden Golden reference for this layer
 * @param input_fp32   Input data (FP32) - from model input or layer input
 * @return             Layer test result
 */
updl_test_layer_result_t updl_test_run_layer_isolation(
    const updl_test_config_t *config,
    const updl_test_layer_golden_t *layer_golden,
    const float *input_fp32);

/**
 * Run a full inference test with layer capture
 * 
 * @param config       Test configuration
 * @param sample       Test sample
 * @param sample_idx   Sample index
 * @return             Sample test result
 */
updl_test_sample_result_t updl_test_run_inference_with_capture(
    const updl_test_config_t *config,
    const updl_test_sample_t *sample,
    uint32_t sample_idx);

/**
 * Free test report memory
 */
void updl_test_free_report(updl_test_report_t *report);

#endif // UPDL_TEST_RUNNER_UTILS_H

