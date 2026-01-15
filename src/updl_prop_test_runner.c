/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

// ============================================================================
// INCLUDES
// ============================================================================

#include <updl/updl_prop_test_runner.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// PUBLIC FUNCTIONS
// ============================================================================

updl_test_report_t *
updl_run_propagation_tests(const updl_test_config_t *config) {

  if (!config || !config->samples || !config->model || !config->executor) {
    updl_Error("%s", "ERROR: Invalid test configuration\n");
    return NULL;
  }

  // Allocate report
  updl_test_report_t *report =
      (updl_test_report_t *)calloc(1, sizeof(updl_test_report_t));
  if (!report) {
    updl_Error("%s", "ERROR: Memory allocation failed for test report\n");
    return NULL;
  }

  report->num_samples = config->num_samples;
  report->sample_results = (updl_sample_result_t *)calloc(
      config->num_samples, sizeof(updl_sample_result_t));

  if (!report->sample_results) {
    updl_Error("%s", "ERROR: Memory allocation failed for sample results\n");
    free(report);
    return NULL;
  }

  updl_Info("%s", "\n");
  updl_Info("%s", "========================================\n");
  updl_Info("%s", "  UPDL Runtime Validation Tests\n");
  updl_Info("%s", "========================================\n");
  updl_Info("Model: %s\n", config->model->model_name);
  updl_Info("Test samples: %d\n", (int)config->num_samples);
  // Run tests for each sample (final output only)
  for (size_t i = 0; i < config->num_samples; i++) {
      if (config->verbose) {
          updl_Info("Sample %d/%d:\n", (int)(i + 1), (int)config->num_samples);
      }
      
      const updl_test_sample_t *sample = &config->samples[i];
      updl_test_sample_result_t sample_result =
          updl_test_run_final_output(config, sample, (uint32_t)i);

      report->sample_results[i] = sample_result;
  }
  
  return report;
}

void updl_free_test_report(updl_test_report_t *report) {
    updl_test_free_report(report);
}

void updl_print_test_report(const updl_test_report_t *report) {
  if (!report) {
    return;
  }
  // No summary needed - all details already shown during layer-by-layer testing
  (void)report;
}
