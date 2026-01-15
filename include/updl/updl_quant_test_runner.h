/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#ifndef UPDL_QUANT_TEST_RUNNER_H
#define UPDL_QUANT_TEST_RUNNER_H

// ============================================================================
// INCLUDES
// ============================================================================

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <updl/updl_interpreter.h>
#include <updl/updl_operator.h>
#include <updl/updl_test_runner_utils.h>

// ============================================================================
// QUANTIZATION TEST DATA INTERFACE (Model-agnostic)
// ============================================================================

// ============================================================================
// QUANTIZATION TEST DATA INTERFACE (Model-agnostic)
// ============================================================================

/**
 * Layer quantization configuration structure
 *
 * Used to define static layer test configuration (before per-sample expansion).
 * Model-specific code provides an array of these.
 */
typedef struct {
  const char *layer_name;         // Layer name (e.g., "activation", "flatten")
  uint16_t layer_index;           // Layer index in model

  // Primary input (required)
  const void *input_golden_data;  // Pointer to 2D array [num_samples][input_size]
  size_t input_size;              // Input size per sample

  // Optional secondary input (for Add/concat layers)
  const void *second_input_golden_data; // Pointer to [num_samples][second_input_size]
  size_t second_input_size;             // Size per sample for second input
  uint16_t num_inputs;                  // Total number of inputs (1 or 2)

  const void *output_golden_data; // Pointer to 2D array [num_samples][output_size]
  size_t output_size;             // Output size per sample
} updl_layer_quant_config_t;

// Use unified types from updl_test_runner_utils.h
typedef updl_test_layer_golden_t updl_layer_quant_golden_t;

/**
 * Quantization test sample structure
 * Model-specific code provides arrays of these
 */
typedef struct {
  const float *model_input_fp32; // FP32 model input (for layer 0)
  size_t model_input_size;       // Model input size

  const updl_layer_quant_golden_t *layers; // Array of layers to validate
  size_t num_layers;                       // Number of layers to validate
} updl_quant_test_sample_t;

/**
 * Quantization test configuration (Model-agnostic)
 */
typedef struct {
  const updl_quant_test_sample_t *samples; // Array of test samples
  size_t num_samples;                      // Number of test samples

  updl_model_t *model;       // UPDL model to test
  updl_executor_t *executor; // UPDL executor to use

  bool verbose; // Print detailed metrics

  // REQUIRED: Reusable buffers to avoid malloc per layer
  int16_t *int16_input_buffer; // Buffer for quantized inputs
  size_t int16_input_buffer_size;

  int16_t *int16_output_buffer; // Buffer for layer outputs
  size_t int16_output_buffer_size;

  int16_t *int16_golden_buffer; // Buffer for quantized golden outputs
  size_t int16_golden_buffer_size;
} updl_quant_test_config_t;

// ============================================================================
// TEST RESULTS (Model-agnostic)
// ============================================================================

// Use unified types from updl_test_runner_utils.h
typedef updl_test_layer_result_t updl_layer_quant_result_t;
typedef updl_test_sample_result_t updl_sample_quant_result_t;

/**
 * Overall quantization test report
 */
typedef struct {
  updl_sample_quant_result_t *sample_results; // Array of sample results
  size_t num_samples;
} updl_quant_test_report_t;

// ============================================================================
// TEST RUNNER FUNCTIONS (Model-agnostic)
// ============================================================================

/**
 * Run quantization tests on UPDL model (layer-by-layer isolation)
 *
 * This test runner validates each layer's quantization in ISOLATION:
 * - Uses FP32 golden input (not previous layer's int16 output)
 * - Compares in int16 domain (using quantization scales)
 * - Prevents error propagation
 *
 * @param config  Test configuration (samples, layers, model, executor)
 * @return        Test report with detailed results
 */
updl_quant_test_report_t *
updl_run_quantization_tests(const updl_quant_test_config_t *config);

/**
 * Free quantization test report memory
 *
 * @param report  Test report to free
 */
void updl_free_quant_test_report(updl_quant_test_report_t *report);

/**
 * Initialize per-sample test data from layer configuration array
 *
 * Helper function to reduce boilerplate in model-specific test runners.
 * Expands layer configs into per-sample golden references.
 *
 * @param layer_configs      Array of layer configurations
 * @param num_layers         Number of layers to test
 * @param num_samples        Number of test samples
 * @param model_inputs_fp32  Model input data [num_samples][input_size]
 * @param model_input_size   Model input size
 * @param quant_layers       Output: per-sample layer golden refs [num_samples][num_layers]
 * @param quant_samples      Output: test samples array [num_samples]
 */
void updl_init_quant_test_data(const updl_layer_quant_config_t *layer_configs,
                               size_t num_layers, size_t num_samples,
                               const float *model_inputs_fp32,
                               size_t model_input_size,
                               updl_layer_quant_golden_t *quant_layers,
                               updl_quant_test_sample_t *quant_samples);

/**
 * Initialize quant test configuration
 */
void updl_init_quant_test_config(
    updl_quant_test_config_t *config, const updl_quant_test_sample_t *samples,
    size_t num_samples, int16_t *input_buffer, size_t input_buffer_size,
    int16_t *output_buffer, size_t output_buffer_size,
    int16_t *golden_buffer, size_t golden_buffer_size, updl_model_t *model,
    updl_executor_t *executor, bool verbose);

/**
 * Free quantization test configuration
 *
 * @param config  Configuration to free
 */
void updl_free_quant_test_config(updl_quant_test_config_t *config);

#endif // UPDL_QUANT_TEST_RUNNER_H
