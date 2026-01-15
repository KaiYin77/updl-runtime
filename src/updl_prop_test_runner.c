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

      if (sample->num_layers == 0) {
        updl_Error("ERROR: No layers configured for sample %d\n", (int)i);
        continue;
      }

      const updl_test_layer_golden_t *golden =
          &sample->layers[sample->num_layers - 1];

      // Allocate result container for final layer only
      updl_sample_result_t *sample_result = &report->sample_results[i];
      sample_result->sample_index = (uint32_t)i;
      sample_result->num_layers = 1;
      sample_result->layer_results = (updl_layer_result_t *)calloc(
          1, sizeof(updl_layer_result_t));
      if (!sample_result->layer_results) {
        updl_Error("ERROR: Memory allocation failed for sample %d\n", (int)i);
        continue;
      }

      // Prepare input buffer
      int16_t *input_int16 = config->int16_input_buffer;
      bool free_input = false;
      if (!input_int16 ||
          (config->int16_input_buffer_size > 0 &&
           sample->input_size > config->int16_input_buffer_size)) {
        input_int16 = (int16_t *)malloc(sample->input_size * sizeof(int16_t));
        free_input = true;
      }

      if (!input_int16) {
        updl_Error("ERROR: Memory allocation failed for sample %d input\n",
                   (int)i);
        free(sample_result->layer_results);
        sample_result->layer_results = NULL;
        continue;
      }

      updl_quantize_fp32_array(sample->input_fp32, input_int16,
                               sample->input_size, config->model->input_scale,
                               0);

      // Prepare output buffer
      size_t output_size = 0;
      if (config->executor && config->executor->exec_layers) {
        output_size = config->executor->exec_layers[config->model->num_layers - 1]
                          .output_size;
      } else {
        const updl_layer_t *last_layer =
            &config->model->layers[config->model->num_layers - 1];
        size_t size = 1;
        for (int k = 0; k < 4; k++) {
          if (last_layer->output_shape[k] > 0) {
            size *= last_layer->output_shape[k];
          }
        }
        output_size = size;
      }

      int16_t *output_int16 = config->int16_output_buffer;
      bool free_output = false;
      if (!output_int16 ||
          (config->int16_output_buffer_size > 0 &&
           output_size > config->int16_output_buffer_size)) {
        output_int16 = (int16_t *)malloc(output_size * sizeof(int16_t));
        free_output = true;
      }

      if (!output_int16) {
        updl_Error("ERROR: Memory allocation failed for sample %d output\n",
                   (int)i);
        if (free_input) {
          free(input_int16);
        }
        free(sample_result->layer_results);
        sample_result->layer_results = NULL;
        continue;
      }

      int exec_result = updl_execute(config->executor, input_int16, output_int16);
      if (exec_result != 0) {
        updl_Error("ERROR: updl_execute failed for sample %d\n", (int)i);
        if (free_input) {
          free(input_int16);
        }
        if (free_output) {
          free(output_int16);
        }
        free(sample_result->layer_results);
        sample_result->layer_results = NULL;
        continue;
      }

      updl_layer_result_t *layer_result = &sample_result->layer_results[0];
      layer_result->layer_name = golden->layer_name;
      layer_result->layer_index = golden->layer_index;

      float output_scale = config->model->layers[golden->layer_index].act_scale;
      bool passed = updl_validate_layer_output(
          output_int16, golden->output_golden_fp32, golden->output_size,
          output_scale, config->verbose, golden->layer_name,
          &layer_result->metrics);

      layer_result->passed = passed;
      layer_result->total_features = golden->output_size;
      layer_result->features_passed =
          layer_result->metrics.matched + layer_result->metrics.diff_1bit;
      layer_result->features_failed = layer_result->metrics.diff_2plus_bits;

      if (passed) {
        sample_result->layers_passed = 1;
      } else {
        sample_result->layers_failed = 1;
      }

      if (free_input) {
        free(input_int16);
      }
      if (free_output) {
        free(output_int16);
      }
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
