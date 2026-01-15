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

  // Optional reusable buffers (provide to avoid malloc)
  int16_t *buffer_hw;
  int16_t *buffer_sw;
  int16_t *buffer_temp;
  size_t buffer_size; // in int16 elements

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
 * Initialize implementation test configuration
 */
void updl_init_impl_test_config(
    updl_impl_test_config_t *config, updl_model_t *model,
    updl_executor_t *executor, const updl_impl_test_input_t *test_inputs,
    size_t num_test_inputs, int16_t *buffer_hw, int16_t *buffer_sw,
    int16_t *buffer_temp, size_t buffer_size, bool verbose);

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
