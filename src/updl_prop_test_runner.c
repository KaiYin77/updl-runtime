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
        updl_Error("ERROR: Dequant buffer too small for layer %s (need %d, "
                   "have %d)\n",
                   golden->layer_name, output_size, ctx->dequant_buffer_size);
        return;
      }
      float *fp32_output = ctx->dequant_buffer;

      // Get quantization parameters for this layer
      const updl_layer_t *layer = &ctx->model->layers[layer_idx];
      float output_scale = layer->act_scale;
      int16_t output_zp = layer->act_zp;

      // Dequantize int16 to fp32 for comparison
      updl_dequantize_int16_array(output, fp32_output, output_size,
                                  output_scale, output_zp);

      // Compare with golden reference and count threshold violations
      size_t pass_count = 0;
      size_t fail_count = 0;

      if (ctx->verbose) {
        updl_Info("  Testing Layer %d: %s\n", golden->layer_index,
                  golden->layer_name);
      }

      // Check each sample against threshold
      for (size_t idx = 0; idx < golden->output_size; idx++) {
        // Get int16 values for comparison
        int16_t actual_int16 = output[idx];

        // Quantize golden fp32 to int16 for comparison
        int16_t golden_int16 = (int16_t)((golden->golden_fp32[idx] / output_scale) + 0.5f);

        // Get fp32 values for display
        float actual_fp32 = fp32_output[idx];
        float golden_fp32 = golden->golden_fp32[idx];

        // Calculate error rate based on fp32 values (using scale)
        float error_rate = 0.0f;
        if (golden_fp32 != 0.0f) {
          error_rate = (actual_fp32 - golden_fp32) / golden_fp32;
          if (error_rate < 0) error_rate = -error_rate;
        } else if (actual_fp32 != 0.0f) {
          error_rate = 1.0f; // 100% error if golden is 0 but actual is not
        }

        if (error_rate <= golden->error_threshold) {
          pass_count++;
        } else {
          fail_count++;
          // Log samples that exceed threshold (limit to first 10)
          if (fail_count <= 10 && ctx->verbose) {
            updl_Error("  [%s] output[%d] = int16(actual=0x%04x, golden=0x%04x), fp32(actual=%.6f, golden=%.6f), error=%.4f%%\n",
                       golden->layer_name, (int)idx, (uint16_t)actual_int16, (uint16_t)golden_int16,
                       actual_fp32, golden_fp32, error_rate * 100.0f);
          }
        }
      }

      // Store pass/fail status and feature counts
      result->passed = (fail_count == 0);
      result->total_features = golden->output_size;
      result->features_passed = pass_count;
      result->features_failed = fail_count;

      if (ctx->verbose) {
        if (fail_count > 10) {
          updl_Info("  (showing first 10 of %d failed features)\n", (int)fail_count);
        }
        updl_Info("  [%s] Result: %d/%d (%.2f%%) features pass\n",
                  golden->layer_name, (int)pass_count, (int)golden->output_size,
                  (100.0f * pass_count) / golden->output_size);
      }

      // Still compute metrics for backward compatibility
      result->metrics = updl_compare_fp32_arrays(
          golden->golden_fp32, fp32_output, golden->output_size);

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
    updl_Error("%s", "ERROR: Memory allocation failed for layer results\n");
    return sample_result;
  }

  if (config->verbose) {
    updl_Info("=== Sample %u ===\n", sample_idx);
  }

  // Quantize input from fp32 to int16
  int16_t *input_int16 =
      (int16_t *)malloc(sample->input_size * sizeof(int16_t));
  if (!input_int16) {
    updl_Error("%s",
               "ERROR: Memory allocation failed for input quantization\n");
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
    updl_Error("%s", "ERROR: Memory allocation failed for output buffer\n");
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
  updl_Info("%s", "========================================\n");

  // Run tests for each sample
  for (size_t i = 0; i < config->num_samples; i++) {
    report->sample_results[i] =
        test_single_sample(config, &config->samples[i], (uint32_t)i);
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

  // No summary needed - all details already shown during layer-by-layer testing
  (void)report;
}