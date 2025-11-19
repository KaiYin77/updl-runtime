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
#include <updl/updl_test_runner_utils.h>

// ============================================================================
// PROPAGATION TEST DATA INTERFACE (Model-agnostic)
// ============================================================================

// Use unified types from updl_test_runner_utils.h
typedef updl_test_layer_golden_t updl_layer_golden_t;
// updl_test_sample_t is already defined in utils
// updl_test_config_t is already defined in utils

// ============================================================================
// TEST RESULTS (Model-agnostic)
// ============================================================================

// Use unified types from updl_test_runner_utils.h
typedef updl_test_layer_result_t updl_layer_result_t;
typedef updl_test_sample_result_t updl_sample_result_t;
// updl_test_report_t is already defined in utils

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
updl_test_report_t *updl_run_propagation_tests(const updl_test_config_t *config);

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
