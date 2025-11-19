/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#ifndef UPDL_IMPL_TEST_RUNNER_H
#define UPDL_IMPL_TEST_RUNNER_H

/**
 * @file updl_impl_test_runner.h
 * @brief Generic UPDL Implementation Test Runner
 *
 * This module provides generic utilities for comparing hardware (UDL) and
 * software-only implementations layer by layer. Model-specific test runners
 * can use these utilities to implement their own A/B testing.
 */

// ============================================================================
// INCLUDES
// ============================================================================

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <updl/updl_interpreter.h>

// ============================================================================
// IMPLEMENTATION TESTING CONFIGURATION
// ============================================================================

/**
 * Test input sample for implementation testing
 */
typedef struct {
  const float *input_fp32; // FP32 input data
  size_t input_size;       // Input size in elements
} updl_impl_test_input_t;

/**
 * Implementation test configuration (Model-agnostic)
 */
typedef struct {
  updl_model_t *model;       // UPDL model to test
  updl_executor_t *executor; // UPDL executor to use

  const updl_impl_test_input_t *test_inputs; // Array of test inputs
  size_t num_test_inputs;                    // Number of test inputs

  bool verbose; // Print detailed comparison for each layer
} updl_impl_test_config_t;

/**
 * Implementation test statistics
 */
typedef struct {
  uint32_t total_layers_compared; // Total number of layers compared
  uint32_t layers_matched;        // Number of layers that matched perfectly
  uint32_t layers_mismatched;     // Number of layers with mismatches
} updl_impl_test_stats_t;

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Compare two int16 arrays element-by-element and report statistics
 *
 * Prints first 5 values of each buffer, counts mismatches, finds max
 * difference, and reports the result. Updates the matched/mismatched counters.
 *
 * @param layer_name  Name of the layer being compared
 * @param layer_idx   Index of the layer
 * @param hw_output   Hardware/UDL output buffer
 * @param sw_output   Software output buffer
 * @param size        Number of elements to compare
 * @param matched     Pointer to counter for matched layers (incremented if all
 * match)
 * @param mismatched  Pointer to counter for mismatched layers (incremented if
 * any mismatch)
 */
void updl_compare_int16_arrays(const char *layer_name, uint16_t layer_idx,
                               const int16_t *hw_output,
                               const int16_t *sw_output, size_t size,
                               uint32_t *matched, uint32_t *mismatched);

/**
 * Get layer type name as string
 *
 * @param type  Layer type enum
 * @return      String representation of layer type
 */
const char *updl_get_layer_type_name(ltype_t type);

/**
 * Execute a single layer with specified mode (UDL or SW)
 *
 * The executor state (rstate_running_hard or rstate_running_soft) must be set
 * by the caller before calling this function.
 *
 * @param executor     UPDL executor (state must be set by caller)
 * @param layer_idx    Layer index to execute
 * @param layer_input  Input buffer for the layer
 * @param layer_output Output buffer for the layer
 * @return             0 on success, negative on error
 */
int updl_execute_single_layer(updl_executor_t *executor, uint16_t layer_idx,
                               const int16_t *layer_input,
                               int16_t *layer_output);

// ============================================================================
// HIGH-LEVEL TEST RUNNER
// ============================================================================

/**
 * Run implementation tests comparing hardware and software execution
 *
 * This generic function performs layer-by-layer comparison between:
 * - Hardware (UDL) implementation (rstate_running_hard)
 * - Software-only implementation (rstate_running_soft)
 *
 * For each test input, it:
 * 1. Quantizes the input from FP32 to int16
 * 2. Runs inference with hardware acceleration (UDL kernels)
 * 3. Runs inference with software-only kernels
 * 4. Compares layer outputs element-by-element
 * 5. Reports any discrepancies
 *
 * Memory efficiency: Uses 3 buffers (HW state, SW state, temp output)
 * sized to the maximum layer output size.
 *
 * @param config  Test configuration with model, executor, and test inputs
 * @return        Test statistics with match/mismatch counts
 */
updl_impl_test_stats_t updl_run_implementation_tests(const updl_impl_test_config_t *config);

#endif // UPDL_IMPL_TEST_RUNNER_H
