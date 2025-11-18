/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

/*
 * UPDL Propagation Test Runner - Full Pipeline Validation with Error Propagation
 *
 * This test runner validates the full inference pipeline by running layers sequentially.
 * Errors propagate from layer to layer, testing the complete end-to-end accuracy.
 *
 * Key Characteristics:
 * - Runs FULL inference (all layers sequentially)
 * - Error PROPAGATES from layer N to layer N+1
 * - Compares dequantized FP32 output with FP32 golden reference
 * - Tests real-world inference behavior
 *
 * Contrast with Quantization Test Runner (updl_quant_test_runner.h):
 * - Quant runner tests layers IN ISOLATION (no error propagation)
 * - Quant runner uses golden FP32 input for each layer
 * - Quant runner compares in INT16 domain
 */

#ifndef UPDL_PROP_TEST_RUNNER_H
#define UPDL_PROP_TEST_RUNNER_H

// ============================================================================
// INCLUDES
// ============================================================================

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <updl/updl_interpreter.h>
#include <updl/updl_operator.h>
#include <updl/updl_test_utils.h>

// ============================================================================
// PROPAGATION TEST DATA INTERFACE (Model-agnostic)
// ============================================================================

/**
 * Layer golden reference structure
 * Model-specific code provides these for layers to validate
 */
typedef struct {
  const char *layer_name;   // Layer name (e.g., "activation", "dense")
  uint16_t layer_index;     // Layer index in model
  const float *golden_fp32; // Expected FP32 output
  size_t output_size;       // Output size
  float error_threshold;    // Acceptable error rate (e.g., 0.05 = 5%)
} updl_layer_golden_t;

/**
 * Test sample structure (redesigned to support per-sample golden references)
 * Model-specific code provides arrays of these
 */
typedef struct {
  const float *input_fp32; // FP32 input data
  size_t input_size;       // Input size
  const updl_layer_golden_t
      *layers;       // Array of layers to validate for this sample
  size_t num_layers; // Number of layers to validate
} updl_test_sample_t;

/**
 * Test configuration (Model-agnostic)
 */
typedef struct {
  const updl_test_sample_t
      *samples;       // Array of test samples (each with its own golden refs)
  size_t num_samples; // Number of test samples

  updl_model_t *model;       // UPDL model to test
  updl_executor_t *executor; // UPDL executor to use

  bool verbose; // Print detailed metrics

  // REQUIRED: Reusable buffer for dequantization (eliminates malloc per layer)
  // Must be sized for the largest layer output in the model
  float *dequant_buffer;      // Reusable buffer (MUST NOT be NULL)
  size_t dequant_buffer_size; // Buffer size in elements
} updl_test_config_t;

// ============================================================================
// TEST RESULTS (Model-agnostic)
// ============================================================================

/**
 * Per-layer test result
 */
typedef struct {
  const char *layer_name;
  uint16_t layer_index;
  updl_test_metrics_t metrics;
  bool passed;

  // Feature-level statistics
  size_t total_features;    // Total features in this layer
  size_t features_passed;   // Features within threshold
  size_t features_failed;   // Features exceeding threshold
} updl_layer_result_t;

/**
 * Per-sample test result
 */
typedef struct {
  uint32_t sample_index;
  updl_layer_result_t *layer_results; // Array of layer results
  size_t num_layers;
  uint32_t layers_passed;
  uint32_t layers_failed;
} updl_sample_result_t;

/**
 * Overall test report
 */
typedef struct {
  updl_sample_result_t *sample_results; // Array of sample results
  size_t num_samples;
} updl_test_report_t;

// ============================================================================
// TEST RUNNER FUNCTIONS (Model-agnostic)
// ============================================================================

/**
 * Run validation tests on UPDL model
 *
 * Generic test runner that works with any model. Model-specific code
 * provides test samples and golden references.
 *
 * @param config  Test configuration (samples, layers, model, executor)
 * @return        Test report with detailed results
 */
updl_test_report_t *updl_run_validation_tests(const updl_test_config_t *config);

/**
 * Free test report memory
 *
 * @param report  Test report to free
 */
void updl_free_test_report(updl_test_report_t *report);

/**
 * Print test report summary
 *
 * @param report  Test report to print
 */
void updl_print_test_report(const updl_test_report_t *report);

#endif // UPDL_PROP_TEST_RUNNER_H
