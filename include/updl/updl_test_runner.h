/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#ifndef UPDL_TEST_RUNNER_H
#define UPDL_TEST_RUNNER_H

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
// TEST DATA INTERFACE (Model-agnostic)
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

  uint32_t total_tests; // Total layer tests (samples * layers)
  uint32_t tests_passed;
  uint32_t tests_failed;
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

#endif // UPDL_TEST_RUNNER_H
