/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

// ============================================================================
// INCLUDES
// ============================================================================

#include <updl/updl_quant_test_runner.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Execute a single layer in isolation
 *
 * This function executes ONLY the specified layer, using the provided
 * int16 input buffer and writing to the int16 output buffer.
 *
 * @param executor     UPDL executor
 * @param layer_index  Index of layer to execute
 * @param input_int16  Int16 input buffer
 * @param output_int16 Int16 output buffer
 * @return 0 on success, -1 on failure
 */
static int execute_single_layer(updl_executor_t *executor, uint16_t layer_index,
                                const int16_t *input_int16,
                                int16_t *output_int16) {
  if (!executor || !executor->model || !input_int16 || !output_int16) {
    updl_Error("%s", "ERROR: NULL pointer in execute_single_layer\n");
    return -1;
  }

  const updl_model_t *model = executor->model;
  if (layer_index >= model->num_layers) {
    updl_Error("ERROR: Layer index %d out of range (model has %d layers)\n",
               layer_index, model->num_layers);
    return -1;
  }

  const updl_layer_t *layer = &model->layers[layer_index];
  updl_exec_layer_t *exec_layer = &executor->exec_layers[layer_index];

  // Set up input/output pointers for this layer
  exec_layer->input_ptr = (uint16_t *)input_int16;
  exec_layer->output_ptr = (uint16_t *)output_int16;

  // Execute layer based on type (same logic as updl_execute)
  int32_t result = 0;
  switch (layer->type) {
  case Ltype_conv_1d:
    result = updl_conv_1d(executor, layer, exec_layer);
    break;
  case Ltype_depthwise_conv_2d:
    result = updl_depthwise_conv_2d(executor, layer, exec_layer);
    break;
  case Ltype_conv_2d:
    result = updl_conv_2d(executor, layer, exec_layer);
    break;
  case Ltype_dense:
    result = updl_dense(executor, layer, exec_layer);
    break;
  case Ltype_max_pooling_2d:
    result = updl_max_pooling_2d(layer, exec_layer);
    break;
  case Ltype_average_pooling_2d:
    result = updl_average_pooling_2d(layer, exec_layer);
    break;
  case Ltype_flatten:
    // Flatten is just a copy operation
    if (exec_layer->input_size > 0) {
      memcpy(exec_layer->output_ptr, exec_layer->input_ptr,
             exec_layer->input_size * sizeof(int16_t));
    }
    result = 0;
    break;
  case Ltype_lambda:
    result = updl_l2_norm(layer, exec_layer);
    break;
  case Ltype_add:
    result = updl_add(layer, exec_layer);
    break;
  case Ltype_softmax:
    result = updl_softmax(executor, layer, exec_layer);
    break;
  default:
    updl_Error("Unsupported layer type: %d\n", layer->type);
    result = -1;
    break;
  }

  if (result != 0) {
    updl_Error("Layer %d execution failed with error %d\n", layer_index,
               result);
    return -1;
  }

  return 0;
}

/**
 * Compare int16 arrays in int16 domain using quantization scale
 *
 * Compares actual int16 output with golden int16 values, using the layer's
 * scale to compute meaningful error rates.
 *
 * @param actual         Actual int16 output from layer
 * @param golden         Expected int16 output (quantized from FP32)
 * @param size           Number of elements
 * @param scale          Layer output scale (for error calculation)
 * @param threshold      Error threshold (e.g., 0.05 = 5%)
 * @param verbose        Print detailed errors
 * @param layer_name     Layer name for logging
 * @param pass_count     Output: number of features that passed
 * @param fail_count     Output: number of features that failed
 * @return true if all features pass threshold
 */
static bool compare_int16_with_scale(const int16_t *actual,
                                     const int16_t *golden, size_t size,
                                     float scale, float threshold, bool verbose,
                                     const char *layer_name, size_t *pass_count,
                                     size_t *fail_count) {
  *pass_count = 0;
  *fail_count = 0;

  for (size_t i = 0; i < size; i++) {
    // Get int16 values for comparison (both already quantized)
    int16_t actual_val = actual[i];
    int16_t golden_val = golden[i];

    // Convert int16 values to fp32 using scale (int16 domain baseline)
    float actual_fp32 = (float)actual_val * scale;
    float golden_fp32 = (float)golden_val * scale;

    // Calculate error rate based on int16 domain (converted to fp32 using scale)
    float error_rate = 0.0f;
    if (golden_fp32 != 0.0f) {
      error_rate = (actual_fp32 - golden_fp32) / golden_fp32;
      if (error_rate < 0)
        error_rate = -error_rate;
    } else if (actual_fp32 != 0.0f) {
      error_rate = 1.0f; // 100% error if golden is 0 but actual is not
    }

    if (error_rate <= threshold) {
      (*pass_count)++;
    } else {
      (*fail_count)++;
      // Log samples that exceed threshold (limit to first 10)
      if (*fail_count <= 10 && verbose) {
        updl_Error("  [%s] output[%d] = int16(actual=0x%04x, golden=0x%04x), fp32(actual=%.6f, golden=%.6f), error=%.4f%%\n",
                   layer_name, (int)i, actual_val, golden_val, actual_fp32, golden_fp32, error_rate * 100.0f);
      }
    }
  }

  if (verbose && *fail_count > 10) {
    updl_Info("  (showing first 10 of %d failed features)\n", (int)*fail_count);
  }

  return (*fail_count == 0);
}

/**
 * Test a single layer in isolation for quantization
 */
static updl_layer_quant_result_t
test_single_layer_quant(const updl_quant_test_config_t *config,
                        const updl_layer_quant_golden_t *layer_golden,
                        const float *layer_input_fp32, bool verbose) {
  updl_layer_quant_result_t result = {0};
  result.layer_name = layer_golden->layer_name;
  result.layer_index = layer_golden->layer_index;
  result.total_features = layer_golden->output_size;
  result.passed = false;

  if (verbose) {
    updl_Info("  Testing Layer %d: %s\n", layer_golden->layer_index,
              layer_golden->layer_name);
  }

  // Get layer quantization parameters
  const updl_layer_t *layer = &config->model->layers[layer_golden->layer_index];
  updl_exec_layer_t *exec_layer =
      &config->executor->exec_layers[layer_golden->layer_index];

  // Determine input scale (from model input or previous layer output)
  float input_scale;
  if (layer_golden->layer_index == 0) {
    input_scale = config->model->input_scale;
  } else {
    const updl_layer_t *prev_layer =
        &config->model->layers[layer_golden->layer_index - 1];
    input_scale = prev_layer->act_scale;
  }

  float output_scale = layer->act_scale;

  // Step 1: Quantize FP32 golden input to int16
  if (layer_golden->input_size > config->int16_input_buffer_size) {
    updl_Error(
        "ERROR: Input buffer too small for layer %s (need %d, have %d)\n",
        layer_golden->layer_name, (int)layer_golden->input_size,
        (int)config->int16_input_buffer_size);
    return result;
  }

  updl_quantize_fp32_array(layer_input_fp32, config->int16_input_buffer,
                           layer_golden->input_size, input_scale, 0);

  // Step 2: Quantize FP32 golden output to int16 for comparison
  if (layer_golden->output_size > config->int16_golden_buffer_size) {
    updl_Error(
        "ERROR: Golden buffer too small for layer %s (need %d, have %d)\n",
        layer_golden->layer_name, (int)layer_golden->output_size,
        (int)config->int16_golden_buffer_size);
    return result;
  }

  updl_quantize_fp32_array(layer_golden->output_golden_fp32,
                           config->int16_golden_buffer,
                           layer_golden->output_size, output_scale, 0);

  // Step 3: Execute ONLY this layer
  if (layer_golden->output_size > config->int16_output_buffer_size) {
    updl_Error(
        "ERROR: Output buffer too small for layer %s (need %d, have %d)\n",
        layer_golden->layer_name, (int)layer_golden->output_size,
        (int)config->int16_output_buffer_size);
    return result;
  }

  int exec_result = execute_single_layer(
      config->executor, layer_golden->layer_index, config->int16_input_buffer,
      config->int16_output_buffer);

  if (exec_result != 0) {
    updl_Error("ERROR: Layer %d (%s) execution failed\n",
               layer_golden->layer_index, layer_golden->layer_name);
    return result;
  }

  // Step 4: Compare int16 output with int16 golden (in int16 domain)
  size_t pass_count = 0;
  size_t fail_count = 0;

  bool passed = compare_int16_with_scale(
      config->int16_output_buffer, config->int16_golden_buffer,
      layer_golden->output_size, output_scale, layer_golden->error_threshold,
      verbose, layer_golden->layer_name, &pass_count, &fail_count);

  result.passed = passed;
  result.features_passed = pass_count;
  result.features_failed = fail_count;

  if (verbose) {
    updl_Info("  [%s] Result: %d/%d (%.2f%%) features pass\n",
              layer_golden->layer_name, (int)pass_count,
              (int)layer_golden->output_size,
              (100.0f * pass_count) / layer_golden->output_size);
  }

  return result;
}

/**
 * Test a single sample (all layers in isolation)
 */
static updl_sample_quant_result_t
test_single_sample_quant(const updl_quant_test_config_t *config,
                         const updl_quant_test_sample_t *sample,
                         uint32_t sample_idx) {
  updl_sample_quant_result_t sample_result = {0};
  sample_result.sample_index = sample_idx;
  sample_result.num_layers = sample->num_layers;

  // Allocate layer results
  sample_result.layer_results = (updl_layer_quant_result_t *)calloc(
      sample->num_layers, sizeof(updl_layer_quant_result_t));
  if (!sample_result.layer_results) {
    updl_Error("%s", "ERROR: Memory allocation failed for layer results\n");
    return sample_result;
  }

  if (config->verbose) {
    updl_Info("=== Sample %u ===\n", sample_idx);
  }

  // Test each layer in isolation
  for (size_t i = 0; i < sample->num_layers; i++) {
    const updl_layer_quant_golden_t *layer_golden = &sample->layers[i];

    // Determine input for this layer
    const float *layer_input_fp32;
    if (layer_golden->layer_index == 0) {
      // First layer uses model input
      layer_input_fp32 = sample->model_input_fp32;
    } else {
      // Other layers use their golden input (from config)
      layer_input_fp32 = layer_golden->input_golden_fp32;
    }

    // Test this layer in isolation
    sample_result.layer_results[i] = test_single_layer_quant(
        config, layer_golden, layer_input_fp32, config->verbose);

    if (sample_result.layer_results[i].passed) {
      sample_result.layers_passed++;
    } else {
      sample_result.layers_failed++;
    }
  }

  return sample_result;
}

// ============================================================================
// PUBLIC FUNCTIONS
// ============================================================================

void updl_init_quant_test_data(const updl_layer_quant_config_t *layer_configs,
                               size_t num_layers, size_t num_samples,
                               const float *model_inputs_fp32,
                               size_t model_input_size,
                               updl_layer_quant_golden_t *quant_layers,
                               updl_quant_test_sample_t *quant_samples) {
  if (!layer_configs || !model_inputs_fp32 || !quant_layers || !quant_samples) {
    updl_Error("%s", "ERROR: NULL pointer in updl_init_quant_test_data\n");
    return;
  }

  // Initialize per-sample layer golden references and samples
  for (size_t sample_idx = 0; sample_idx < num_samples; sample_idx++) {
    // Initialize each layer for this sample
    for (size_t layer_idx = 0; layer_idx < num_layers; layer_idx++) {
      const updl_layer_quant_config_t *config = &layer_configs[layer_idx];

      // Calculate array index for this sample's layers
      // quant_layers is treated as a 2D array [num_samples][num_layers]
      updl_layer_quant_golden_t *layer_golden =
          &quant_layers[sample_idx * num_layers + layer_idx];

      layer_golden->layer_name = config->layer_name;
      layer_golden->layer_index = config->layer_index;

      // Set input golden reference
      if (config->input_golden_data == NULL) {
        // No input golden data configured (will use model input from sample)
        layer_golden->input_golden_fp32 = NULL;
      } else {
        // Use configured input golden data for this sample
        layer_golden->input_golden_fp32 =
            (const float *)config->input_golden_data +
            (sample_idx * config->input_size);
      }
      layer_golden->input_size = config->input_size;

      // Set output golden reference
      layer_golden->output_golden_fp32 =
          (const float *)config->output_golden_data +
          (sample_idx * config->output_size);
      layer_golden->output_size = config->output_size;
      layer_golden->error_threshold = config->error_threshold;
    }

    // Initialize sample
    quant_samples[sample_idx].model_input_fp32 =
        model_inputs_fp32 + (sample_idx * model_input_size);
    quant_samples[sample_idx].model_input_size = model_input_size;
    quant_samples[sample_idx].layers = &quant_layers[sample_idx * num_layers];
    quant_samples[sample_idx].num_layers = num_layers;
  }
}

updl_quant_test_report_t *
updl_run_quantization_tests(const updl_quant_test_config_t *config) {
  if (!config || !config->samples || !config->model || !config->executor) {
    updl_Error("%s", "ERROR: Invalid test configuration\n");
    return NULL;
  }

  if (!config->int16_input_buffer || !config->int16_output_buffer ||
      !config->int16_golden_buffer) {
    updl_Error("%s", "ERROR: Buffers not provided in config\n");
    return NULL;
  }

  // Allocate report
  updl_quant_test_report_t *report =
      (updl_quant_test_report_t *)calloc(1, sizeof(updl_quant_test_report_t));
  if (!report) {
    updl_Error("%s", "ERROR: Memory allocation failed for test report\n");
    return NULL;
  }

  report->num_samples = config->num_samples;
  report->sample_results = (updl_sample_quant_result_t *)calloc(
      config->num_samples, sizeof(updl_sample_quant_result_t));

  if (!report->sample_results) {
    updl_Error("%s", "ERROR: Memory allocation failed for sample results\n");
    free(report);
    return NULL;
  }

  updl_Info("%s", "\n");
  updl_Info("%s", "========================================\n");
  updl_Info("%s", "  UPDL Quantization Tests\n");
  updl_Info("%s", "  (Layer-by-Layer Isolation)\n");
  updl_Info("%s", "========================================\n");
  updl_Info("Model: %s\n", config->model->model_name);
  updl_Info("Test samples: %d\n", (int)config->num_samples);

  // Print layers info from first sample
  if (config->num_samples > 0 && config->samples[0].num_layers > 0) {
    updl_Info("Layers to validate per sample: %d\n",
              (int)config->samples[0].num_layers);
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
        test_single_sample_quant(config, &config->samples[i], (uint32_t)i);
  }

  return report;
}

void updl_free_quant_test_report(updl_quant_test_report_t *report) {
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

void updl_free_quant_test_config(updl_quant_test_config_t *config) {
  if (!config) {
    return;
  }

  // Only free the config struct itself
  // Samples, layers, and buffers are expected to be static or managed by caller
  free(config);
}
