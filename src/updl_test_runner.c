/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

// ============================================================================
// INCLUDES
// ============================================================================

#include <updl/updl_test_runner.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Layer capture context for callback
 */
typedef struct {
  const updl_test_sample_t *sample;   // Current sample being tested
  updl_layer_result_t *layer_results; // Array to store results
  bool verbose;                       // Verbose output flag
  const updl_model_t *model;          // Model reference for quantization params
  float *dequant_buffer;              // Reusable buffer (NULL = use malloc)
  size_t dequant_buffer_size;         // Buffer size in elements
} layer_capture_context_t;

/**
 * Callback function to capture layer outputs during inference
 * Called by updl_execute() after each layer completes
 */
static void layer_capture_callback(uint16_t layer_idx, const int16_t *output,
                                   size_t output_size, void *user_data) {
  layer_capture_context_t *ctx = (layer_capture_context_t *)user_data;
  if (!ctx || !ctx->sample || !ctx->layer_results) {
    return;
  }

  // Find if this layer is one we want to test
  for (size_t i = 0; i < ctx->sample->num_layers; i++) {
    const updl_layer_golden_t *golden = &ctx->sample->layers[i];

    if (golden->layer_index == layer_idx) {
      // This is a layer we want to test - capture and compare
      updl_layer_result_t *result = &ctx->layer_results[i];
      result->layer_name = golden->layer_name;
      result->layer_index = golden->layer_index;
      result->passed = false;

      // Use provided reusable buffer (no dynamic allocation)
      if (!ctx->dequant_buffer || output_size > ctx->dequant_buffer_size) {
        updl_Error(
            "ERROR: Dequant buffer too small for layer %s (need %zu, have %zu)\n",
            golden->layer_name, output_size, ctx->dequant_buffer_size);
        return;
      }
      float *fp32_output = ctx->dequant_buffer;

      // Get quantization parameters for this layer
      const updl_layer_t *layer = &ctx->model->layers[layer_idx];
      float output_scale = layer->act_scale;
      int16_t output_zp = layer->act_zp;

      // Dequantize int16 to fp32
      updl_dequantize_int16_array(output, fp32_output, output_size,
                                  output_scale, output_zp);

      // Compare with golden reference
      result->metrics = updl_compare_fp32_arrays(
          golden->golden_fp32, fp32_output, golden->output_size);

      // Check if passed
      float abs_mean_error = result->metrics.mean_error_rate < 0
                                 ? -result->metrics.mean_error_rate
                                 : result->metrics.mean_error_rate;
      result->passed = (abs_mean_error <= golden->error_threshold);

      if (ctx->verbose) {
        updl_Info("  Layer: %s (sample_idx=%d, layer_idx=%d)\n",
                  golden->layer_name, i, golden->layer_index);
        updl_Info("    Mean Error: %.4f%%\n", abs_mean_error * 100.0f);
        updl_Info("    Max Error:  %.4f%%\n",
                  result->metrics.max_error_rate * 100.0f);
        if (result->passed == false) {
          updl_Error("    Result: %s\n", "FAIL");
        } else {
          updl_Info("    Result: %s\n", "PASS");
        }
      }

      // Buffer is reused, no free needed
      break; // Found and processed this layer
    }
  }
}

/**
 * Test a single sample (run inference with callback to validate layers)
 */
static updl_sample_result_t test_single_sample(const updl_test_config_t *config,
                                               const updl_test_sample_t *sample,
                                               uint32_t sample_idx) {

  updl_sample_result_t sample_result = {0};
  sample_result.sample_index = sample_idx;
  sample_result.num_layers = sample->num_layers;

  // Allocate layer results
  sample_result.layer_results = (updl_layer_result_t *)calloc(
      sample->num_layers, sizeof(updl_layer_result_t));
  if (!sample_result.layer_results) {
    updl_Error("ERROR: Memory allocation failed for layer results\n");
    return sample_result;
  }

  if (config->verbose) {
    updl_Info("\n--- Sample %u ---\n", sample_idx);
  }

  // Quantize input from fp32 to int16
  int16_t *input_int16 =
      (int16_t *)malloc(sample->input_size * sizeof(int16_t));
  if (!input_int16) {
    updl_Error("ERROR: Memory allocation failed for input quantization\n");
    free(sample_result.layer_results);
    sample_result.layer_results = NULL;
    return sample_result;
  }

  // Get input quantization parameters from model
  float input_scale = config->model->input_scale;
  int16_t input_zp = 0; // Typically 0 for inputs, but could be configurable

  updl_quantize_fp32_array(sample->input_fp32, input_int16, sample->input_size,
                           input_scale, input_zp);

  // Allocate output buffer
  // Get output size from last layer
  size_t output_size = config->model->layers[config->model->num_layers - 1]
                           .output_shape[1]; // Assuming [batch, classes]
  int16_t *output_int16 = (int16_t *)malloc(output_size * sizeof(int16_t));
  if (!output_int16) {
    updl_Error("ERROR: Memory allocation failed for output buffer\n");
    free(input_int16);
    free(sample_result.layer_results);
    sample_result.layer_results = NULL;
    return sample_result;
  }

  // Set up callback context for layer capture
  layer_capture_context_t capture_ctx = {
      .sample = sample,
      .layer_results = sample_result.layer_results,
      .verbose = config->verbose,
      .model = config->model,
      .dequant_buffer = config->dequant_buffer,
      .dequant_buffer_size = config->dequant_buffer_size};

  // Register callback to capture layer outputs during inference
  updl_set_layer_callback(config->executor, layer_capture_callback,
                          &capture_ctx);

  // Run inference (callback will be called for each layer)
  int result = updl_execute(config->executor, input_int16, output_int16);

  // Unregister callback
  updl_set_layer_callback(config->executor, NULL, NULL);

  if (result != 0) {
    updl_Error("ERROR: updl_execute failed with error %d for sample %u\n",
               result, sample_idx);
    free(input_int16);
    free(output_int16);
    free(sample_result.layer_results);
    sample_result.layer_results = NULL;
    return sample_result;
  }

  // Count passed/failed layers (callback already populated results)
  for (size_t i = 0; i < sample->num_layers; i++) {
    if (sample_result.layer_results[i].passed) {
      sample_result.layers_passed++;
    } else {
      sample_result.layers_failed++;
    }
  }

  free(input_int16);
  free(output_int16);

  return sample_result;
}

// ============================================================================
// PUBLIC FUNCTIONS
// ============================================================================

updl_test_report_t *
updl_run_validation_tests(const updl_test_config_t *config) {

  if (!config || !config->samples || !config->model || !config->executor) {
    updl_Error("ERROR: Invalid test configuration\n");
    return NULL;
  }

  // Allocate report
  updl_test_report_t *report =
      (updl_test_report_t *)calloc(1, sizeof(updl_test_report_t));
  if (!report) {
    updl_Error("ERROR: Memory allocation failed for test report\n");
    return NULL;
  }

  report->num_samples = config->num_samples;
  report->sample_results = (updl_sample_result_t *)calloc(
      config->num_samples, sizeof(updl_sample_result_t));

  if (!report->sample_results) {
    updl_Error("ERROR: Memory allocation failed for sample results\n");
    free(report);
    return NULL;
  }

  updl_Info("\n");
  updl_Info("========================================\n");
  updl_Info("  UPDL Runtime Validation Tests\n");
  updl_Info("========================================\n");
  updl_Info("Model: %s\n", config->model->model_name);
  updl_Info("Test samples: %d\n", config->num_samples);

  // Print layers info from first sample (assuming all samples test same layers)
  if (config->num_samples > 0 && config->samples[0].num_layers > 0) {
    updl_Info("Layers to validate per sample: %d\n",
              config->samples[0].num_layers);
    for (size_t i = 0; i < config->samples[0].num_layers; i++) {
      updl_Info("  - %s (idx=%d, threshold=%.2f%%)\n",
                config->samples[0].layers[i].layer_name,
                config->samples[0].layers[i].layer_index,
                config->samples[0].layers[i].error_threshold * 100.0f);
    }
  }
  updl_Info("========================================\n");

  // Run tests for each sample
  for (size_t i = 0; i < config->num_samples; i++) {
    report->sample_results[i] =
        test_single_sample(config, &config->samples[i], (uint32_t)i);

    report->total_tests += config->samples[i].num_layers;
    report->tests_passed += report->sample_results[i].layers_passed;
    report->tests_failed += report->sample_results[i].layers_failed;
  }

  return report;
}

void updl_free_test_report(updl_test_report_t *report) {
  if (!report) {
    return;
  }

  if (report->sample_results) {
    for (size_t i = 0; i < report->num_samples; i++) {
      free(report->sample_results[i].layer_results);
    }
    free(report->sample_results);
  }

  free(report);
}

void updl_print_test_report(const updl_test_report_t *report) {
  if (!report) {
    return;
  }

  updl_Info("\n");
  updl_Info("========================================\n");
  updl_Info("  Test Report Summary\n");
  updl_Info("========================================\n");
  updl_Info("Total tests:  %u (samples: %d, layers: %d)\n", report->total_tests,
            report->num_samples,
            report->total_tests > 0 ? report->total_tests / report->num_samples
                                    : 0);
  updl_Info("Passed:       %u\n", report->tests_passed);
  updl_Error("Failed:       %u\n", report->tests_failed);
  updl_Info("Success rate: %.1f%%\n",
            report->total_tests > 0
                ? (float)report->tests_passed / report->total_tests * 100.0f
                : 0.0f);
  updl_Info("========================================\n\n");
}