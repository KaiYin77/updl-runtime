/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

// ============================================================================
// INCLUDES
// ============================================================================

#include <updl/updl_test_runner_utils.h>

#include <math.h>
#include <string.h>
#include <stdlib.h>

#include <updl/updl_kernels.h>
#include <updl/updl_operator.h>
#include <updl/updl_utility.h>

// ============================================================================
// QUANTIZATION FUNCTIONS
// ============================================================================

int16_t updl_quantize_fp32_to_int16(float value, float scale,
                                    int16_t zero_point) {
  // Formula: quantized = round(value / scale) + zero_point
  float scaled = roundf(value / scale) + (float)zero_point;

  // Clamp to int16 range
  if (scaled > (float)INT16_MAX) {
    return INT16_MAX;
  } else if (scaled < (float)INT16_MIN) {
    return INT16_MIN;
  }

  return (int16_t)scaled;
}

void updl_quantize_fp32_array(const float *fp32_array, int16_t *int16_array,
                              size_t size, float scale, int16_t zero_point) {
  if (!fp32_array || !int16_array || size == 0) {
    return;
  }

  for (size_t i = 0; i < size; i++) {
    int16_array[i] =
        updl_quantize_fp32_to_int16(fp32_array[i], scale, zero_point);
  }
}

// ============================================================================
// DEQUANTIZATION FUNCTIONS
// ============================================================================

float updl_dequantize_int16_to_fp32(int16_t value, float scale,
                                    int16_t zero_point) {
  return (float)(value - zero_point) * scale;
}

void updl_dequantize_int16_array(const int16_t *int16_array, float *fp32_array,
                                 size_t size, float scale, int16_t zero_point) {
  if (!int16_array || !fp32_array || size == 0) {
    return;
  }

  for (size_t i = 0; i < size; i++) {
    fp32_array[i] =
        updl_dequantize_int16_to_fp32(int16_array[i], scale, zero_point);
  }
}

// ============================================================================
// COMPARISON FUNCTIONS
// ============================================================================

updl_test_metrics_t updl_compare_fp32_arrays(const float *golden_array,
                                             const float *test_array,
                                             size_t size) {
  updl_test_metrics_t metrics = {0};

  if (!golden_array || !test_array || size == 0) {
    return metrics;
  }

  metrics.num_samples = size;
  float sum_error_rate = 0.0f;
  float max_error_rate = 0.0f;

  for (size_t i = 0; i < size; i++) {
    float golden = golden_array[i];
    float test = test_array[i];

    // Skip if golden value is zero (avoid division by zero)
    if (fabsf(golden) < 1e-10f) {
      continue;
    }

    // Calculate error rate: (test - golden) / golden
    float error_rate = (test - golden) / golden;
    float abs_error_rate = fabsf(error_rate);

    sum_error_rate += error_rate;

    if (abs_error_rate > max_error_rate) {
      max_error_rate = abs_error_rate;
    }
  }

  metrics.mean_error_rate = sum_error_rate / (float)size;
  metrics.max_error_rate = max_error_rate;

  return metrics;
}

// ============================================================================
// EXECUTION UTILITIES
// ============================================================================

const char *updl_get_layer_type_name(ltype_t type) {
  switch (type) {
  case Ltype_conv_1d:
    return "Conv1D";
  case Ltype_conv_2d:
    return "Conv2D";
  case Ltype_depthwise_conv_2d:
    return "DepthwiseConv2D";
  case Ltype_dense:
    return "Dense";
  case Ltype_max_pooling_2d:
    return "MaxPool2D";
  case Ltype_average_pooling_2d:
    return "AvgPool2D";
  case Ltype_flatten:
    return "Flatten";
  case Ltype_lambda:
    return "Lambda/L2Norm";
  case Ltype_add:
    return "Add";
  case Ltype_softmax:
    return "Softmax";
  default:
    return "Unknown";
  }
}

int updl_execute_single_layer(updl_executor_t *executor, uint16_t layer_idx,
                              const int16_t *input_int16,
                              int16_t *output_int16) {
  if (!executor || !executor->model || !input_int16 || !output_int16) {
    updl_Error("%s", "ERROR: NULL pointer in updl_execute_single_layer\n");
    return -1;
  }

  const updl_model_t *model = executor->model;
  if (layer_idx >= model->num_layers) {
    updl_Error("ERROR: Layer index %d out of range (model has %d layers)\n",
               layer_idx, model->num_layers);
    return -1;
  }

  const updl_layer_t *layer = &model->layers[layer_idx];
  updl_exec_layer_t *exec_layer = &executor->exec_layers[layer_idx];

  // Set executor state for proper execution
  executor->current_layer = layer_idx;

  // Set up input/output pointers for this layer
  exec_layer->input_ptr = (uint16_t *)input_int16;
  exec_layer->output_ptr = (uint16_t *)output_int16;

  // Execute layer based on type
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
    updl_Error("Layer %d execution failed with error %d\n", layer_idx, result);
    return -1;
  }

  return 0;
}

// ============================================================================
// VALIDATION UTILITIES
// ============================================================================

bool updl_validate_layer_output(const int16_t *actual, const float *golden_fp32,
                                size_t size, float scale, float threshold,
                                bool verbose, const char *layer_name,
                                size_t *pass_count, size_t *fail_count,
                                updl_test_metrics_t *metrics) {
  *pass_count = 0;
  *fail_count = 0;

  if (metrics) {
    memset(metrics, 0, sizeof(updl_test_metrics_t));
    metrics->num_samples = size;
  }

  float sum_error_rate = 0.0f;
  float max_error_rate = 0.0f;

  for (size_t i = 0; i < size; i++) {
    // Get int16 values for comparison
    int16_t actual_val = actual[i];
    float golden_val_fp32 = golden_fp32[i];

    // Quantize golden fp32 to int16 for comparison (simulate int16 domain)
    // Note: We use simple rounding here as per standard quantization
    int16_t golden_val_int16 = (int16_t)((golden_val_fp32 / scale) + 0.5f);

    // Convert int16 values to fp32 using scale (int16 domain baseline)
    float actual_fp32 = (float)actual_val * scale;
    float golden_fp32_from_int16 = (float)golden_val_int16 * scale;

    // Calculate error rate based on int16 domain (converted to fp32 using scale)
    float error_rate = 0.0f;
    if (golden_fp32_from_int16 != 0.0f) {
      error_rate =
          (actual_fp32 - golden_fp32_from_int16) / golden_fp32_from_int16;
      if (error_rate < 0)
        error_rate = -error_rate;
    } else if (actual_fp32 != 0.0f) {
      error_rate = 1.0f; // 100% error if golden is 0 but actual is not
    }

    // Update metrics if requested
    if (metrics) {
      sum_error_rate += (actual_fp32 - golden_fp32_from_int16) /
                        (golden_fp32_from_int16 != 0.0f ? golden_fp32_from_int16 : 1.0f); // Approximate signed error for mean
      if (error_rate > max_error_rate) {
        max_error_rate = error_rate;
      }
    }

    if (error_rate <= threshold) {
      (*pass_count)++;
    } else {
      (*fail_count)++;
      // Log samples that exceed threshold (limit to first 10)
      if (*fail_count <= 10 && verbose) {
        updl_Error("  [%s] output[%d] = int16(actual=0x%04x, golden=0x%04x), "
                   "fp32(actual=%.6f, golden=%.6f), error=%.4f%%\n",
                   layer_name, (int)i, actual_val, golden_val_int16,
                   actual_fp32, golden_val_fp32, error_rate * 100.0f);
      }
    }
  }

  if (metrics) {
    metrics->mean_error_rate = sum_error_rate / (float)size;
    metrics->max_error_rate = max_error_rate;
  }

  if (verbose && *fail_count > 10) {
    updl_Info("  (showing first 10 of %d mismatched features)\n", (int)*fail_count);
  }

  return (*fail_count == 0);
}

void updl_compare_int16_buffers(const char *layer_name, uint16_t layer_idx,
                                const int16_t *buffer1, const int16_t *buffer2,
                                size_t size, uint32_t *matched,
                                uint32_t *mismatched) {
  // Debug output for first few elements
  updl_Info("    - output_1[:5]: 0x%04x 0x%04x 0x%04x 0x%04x 0x%04x\n",
            (uint16_t)buffer1[0], (uint16_t)buffer1[1], (uint16_t)buffer1[2],
            (uint16_t)buffer1[3], (uint16_t)buffer1[4]);
  updl_Info("    - output_2[:5]: 0x%04x 0x%04x 0x%04x 0x%04x 0x%04x\n",
            (uint16_t)buffer2[0], (uint16_t)buffer2[1], (uint16_t)buffer2[2],
            (uint16_t)buffer2[3], (uint16_t)buffer2[4]);

  uint32_t mismatch_count = 0;
  int32_t max_diff = 0;
  size_t first_mismatch_idx = 0;
  bool found_first_mismatch = false;

  for (size_t i = 0; i < size; i++) {
    if (buffer1[i] != buffer2[i]) {
      int32_t diff = abs(buffer1[i] - buffer2[i]);
      if (diff > max_diff) {
        max_diff = diff;
      }
      if (!found_first_mismatch) {
        first_mismatch_idx = i;
        found_first_mismatch = true;
      }
      mismatch_count++;
    }
  }

  if (mismatch_count == 0) {
    updl_Info("  Layer %2d (%15s): PASS - All %d values match\n", layer_idx,
              layer_name, (int)size);
    (*matched)++;
  } else {
    updl_Error("  Layer %2d (%15s): FAIL - %u/%d mismatches (max_diff=%d, "
               "first@%d: output_1=%d "
               "output_2=%d)\n",
               layer_idx, layer_name, mismatch_count, (int)size, max_diff,
               (int)first_mismatch_idx, buffer1[first_mismatch_idx],
               buffer2[first_mismatch_idx]);
    (*mismatched)++;
  }
}

// ============================================================================
// SHARED RUNNER FUNCTIONS
// ============================================================================

updl_test_layer_result_t updl_test_run_layer_isolation(
    const updl_test_config_t *config,
    const updl_test_layer_golden_t *layer_golden,
    const float *input_fp32) {
    
  updl_test_layer_result_t result = {0};
  result.layer_name = layer_golden->layer_name;
  result.layer_index = layer_golden->layer_index;
  result.total_features = layer_golden->output_size;
  result.passed = false;

  if (config->verbose) {
    updl_Info("  Testing Layer %d: %s\n", layer_golden->layer_index,
              layer_golden->layer_name);
  }

  // Get layer quantization parameters
  const updl_layer_t *layer = &config->model->layers[layer_golden->layer_index];
  
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

  updl_quantize_fp32_array(input_fp32, config->int16_input_buffer,
                           layer_golden->input_size, input_scale, 0);

  // Step 2: Execute ONLY this layer
  if (layer_golden->output_size > config->int16_output_buffer_size) {
    updl_Error(
        "ERROR: Output buffer too small for layer %s (need %d, have %d)\n",
        layer_golden->layer_name, (int)layer_golden->output_size,
        (int)config->int16_output_buffer_size);
    return result;
  }

  int exec_result = updl_execute_single_layer(
      config->executor, layer_golden->layer_index, config->int16_input_buffer,
      config->int16_output_buffer);

  if (exec_result != 0) {
    updl_Error("ERROR: Layer %d (%s) execution failed\n",
               layer_golden->layer_index, layer_golden->layer_name);
    return result;
  }

  // Step 3: Compare int16 output with golden FP32 (using shared validation)
  size_t pass_count = 0;
  size_t fail_count = 0;

  bool passed = updl_validate_layer_output(
      config->int16_output_buffer, layer_golden->output_golden_fp32,
      layer_golden->output_size, output_scale, layer_golden->error_threshold,
      config->verbose, layer_golden->layer_name, &pass_count, &fail_count, NULL);

  result.passed = passed;
  result.features_passed = pass_count;
  result.features_failed = fail_count;

  if (config->verbose) {
    updl_Info("  [%s] Result: %d/%d (%.2f%%) features pass\n",
              layer_golden->layer_name, (int)pass_count,
              (int)layer_golden->output_size,
              (100.0f * pass_count) / layer_golden->output_size);
  }

  return result;
}

// Helper for capture callback
typedef struct {
  const updl_test_sample_t *sample;
  updl_test_layer_result_t *layer_results;
  bool verbose;
  const updl_model_t *model;
} layer_capture_context_t;

static void layer_capture_callback(uint16_t layer_idx, const int16_t *output,
                                   size_t output_size, void *user_data) {
  layer_capture_context_t *ctx = (layer_capture_context_t *)user_data;
  if (!ctx || !ctx->sample || !ctx->layer_results) {
    return;
  }

  // Find if this layer is one we want to test
  for (size_t i = 0; i < ctx->sample->num_layers; i++) {
    const updl_test_layer_golden_t *golden = &ctx->sample->layers[i];

    if (golden->layer_index == layer_idx) {
      // This is a layer we want to test - capture and compare
      updl_test_layer_result_t *result = &ctx->layer_results[i];
      result->layer_name = golden->layer_name;
      result->layer_index = golden->layer_index;
      result->passed = false;

      if (ctx->verbose) {
        updl_Info("  Testing Layer %d: %s\n", golden->layer_index,
                  golden->layer_name);
      }

      // Get quantization parameters for this layer
      const updl_layer_t *layer = &ctx->model->layers[layer_idx];
      float output_scale = layer->act_scale;

      // Compare with golden reference using shared validation utility
      size_t pass_count = 0;
      size_t fail_count = 0;

      updl_validate_layer_output(output, golden->output_golden_fp32,
                                 golden->output_size, output_scale,
                                 golden->error_threshold, ctx->verbose,
                                 golden->layer_name, &pass_count, &fail_count,
                                 &result->metrics);

      // Store pass/fail status and feature counts
      result->passed = (fail_count == 0);
      result->total_features = golden->output_size;
      result->features_passed = pass_count;
      result->features_failed = fail_count;

      if (ctx->verbose) {
        updl_Info("  [%s] Result: %d/%d (%.2f%%) features pass\n",
                  golden->layer_name, (int)pass_count, (int)golden->output_size,
                  (100.0f * pass_count) / golden->output_size);
      }

      break; // Found and processed this layer
    }
  }
}

updl_test_sample_result_t updl_test_run_inference_with_capture(
    const updl_test_config_t *config,
    const updl_test_sample_t *sample,
    uint32_t sample_idx) {
    
  updl_test_sample_result_t sample_result = {0};
  sample_result.sample_index = sample_idx;
  sample_result.num_layers = sample->num_layers;

  // Allocate layer results
  sample_result.layer_results = (updl_test_layer_result_t *)calloc(
      sample->num_layers, sizeof(updl_test_layer_result_t));
  if (!sample_result.layer_results) {
    updl_Error("%s", "ERROR: Memory allocation failed for layer results\n");
    return sample_result;
  }



  // Quantize input from fp32 to int16
  // Use the shared input buffer from config if available, otherwise malloc
  int16_t *input_int16 = config->int16_input_buffer;
  bool free_input = false;
  
  if (!input_int16) {
      input_int16 = (int16_t *)malloc(sample->input_size * sizeof(int16_t));
      free_input = true;
  }
  
  if (!input_int16) {
    updl_Error("%s",
               "ERROR: Memory allocation failed for input quantization\n");
    free(sample_result.layer_results);
    sample_result.layer_results = NULL;
    return sample_result;
  }

  // Get input quantization parameters from model
  float input_scale = config->model->input_scale;
  int16_t input_zp = 0; // Typically 0 for inputs

  updl_quantize_fp32_array(sample->input_fp32, input_int16, sample->input_size,
                           input_scale, input_zp);

  // Allocate output buffer
  // Use shared output buffer if available
  int16_t *output_int16 = config->int16_output_buffer;
  bool free_output = false;
  
  // Get output size from last layer
  size_t output_size = config->model->layers[config->model->num_layers - 1]
                           .output_shape[1]; // Assuming [batch, classes]
                           
  if (!output_int16) {
      output_int16 = (int16_t *)malloc(output_size * sizeof(int16_t));
      free_output = true;
  }
  
  if (!output_int16) {
    updl_Error("%s", "ERROR: Memory allocation failed for output buffer\n");
    if (free_input) free(input_int16);
    free(sample_result.layer_results);
    sample_result.layer_results = NULL;
    return sample_result;
  }

  // Set up callback context for layer capture
  layer_capture_context_t capture_ctx = {
      .sample = sample,
      .layer_results = sample_result.layer_results,
      .verbose = config->verbose,
      .model = config->model
  };

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
    if (free_input) free(input_int16);
    if (free_output) free(output_int16);
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

  if (free_input) free(input_int16);
  if (free_output) free(output_int16);

  return sample_result;
}

void updl_test_free_report(updl_test_report_t *report) {
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