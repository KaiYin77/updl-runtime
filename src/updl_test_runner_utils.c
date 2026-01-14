/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

// ============================================================================
// INCLUDES
// ============================================================================

#include <updl/updl_test_runner_utils.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

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
    result = updl_add(executor, layer, exec_layer);
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
                                size_t size, float scale, bool verbose,
                                const char *layer_name,
                                updl_test_metrics_t *metrics) {
  if (metrics) {
    memset(metrics, 0, sizeof(updl_test_metrics_t));
    metrics->num_samples = size;
  }

  size_t matched = 0;
  size_t diff_1 = 0;
  size_t diff_2plus = 0;
  int16_t max_error = 0;
  int16_t max_error_actual = 0;
  int16_t max_error_golden = 0;
  size_t logged_errors = 0;

  for (size_t i = 0; i < size; i++) {
    // Get int16 values for comparison
    int16_t actual_val = actual[i];
    float golden_val_fp32 = golden_fp32[i];

    // Quantize golden fp32 to int16 for comparison (simulate int16 domain)
    int16_t golden_val_int16 = updl_quantize_fp32_to_int16(golden_val_fp32, scale, 0);

    // Calculate absolute error in int16 domain
    int16_t abs_error = (int16_t)(actual_val - golden_val_int16);
    if (abs_error < 0) {
      abs_error = -abs_error;
    }

    // Categorize into buckets
    if (abs_error == 0) {
      matched++;
    } else if (abs_error == 1) {
      diff_1++;
    } else {
      diff_2plus++;
      // Log samples with 2+ errors (limit to first 10)
      if (logged_errors < 10 && verbose) {
        float actual_fp32 = (float)actual_val * scale;
        float fp32_error = actual_fp32 - golden_val_fp32;
        float error_rate =
            (golden_val_fp32 != 0.0f) ? (fp32_error / golden_val_fp32) : 0.0f;

        updl_Warning(
            "    out[%d] = i16(err=%d, compute=0x%04x, golden=0x%04x), "
            "fp32(err=%.4f, err_ratio=%.4f%%, compute=%.4f, golden=%.4f)\n",
            (int)i, abs_error, actual_val, golden_val_int16, fp32_error,
            error_rate * 100.0f, actual_fp32, golden_val_fp32);
        logged_errors++;
      }
    }

    // Track max error and its values
    if (abs_error > max_error) {
      max_error = abs_error;
      max_error_actual = actual_val;
      max_error_golden = golden_val_int16;
    }
  }

  // Store metrics
  if (metrics) {
    metrics->matched = matched;
    metrics->diff_1bit = diff_1;
    metrics->diff_2plus_bits = diff_2plus;
    metrics->max_error_bits = max_error;
  }

  // Print distribution with percentages
  if (verbose) {
    float matched_pct = (size > 0) ? (100.0f * matched / size) : 0.0f;
    float diff_1_pct = (size > 0) ? (100.0f * diff_1 / size) : 0.0f;
    float diff_2plus_pct = (size > 0) ? (100.0f * diff_2plus / size) : 0.0f;

    updl_Info("    Report: matched=%d/%d(%.1f%%), abs(1)=%d/%d(%.1f%%), "
              "abs(2+)=%d/%d(%.1f%%), abs(max_err)=%d(0x%04x, 0x%04x)\n",
              (int)matched, (int)size, matched_pct, (int)diff_1, (int)size,
              diff_1_pct, (int)diff_2plus, (int)size, diff_2plus_pct, max_error,
              (uint16_t)max_error_actual, (uint16_t)max_error_golden);
  }

  if (verbose && logged_errors > 0 && diff_2plus > 10) {
    updl_Info("    Note: showing first 10 of %d features with abs(2+) errors\n",
              (int)diff_2plus);
  }

  // Consider pass if all are exact match or abs(1) difference
  return (diff_2plus == 0);
}

void updl_compare_int16_buffers(const char *layer_name, uint16_t layer_idx,
                                const int16_t *buffer1, const int16_t *buffer2,
                                size_t size, uint32_t *matched,
                                uint32_t *mismatched) {
  // Debug output for first few elements
  updl_Info("      output_1[:5]: 0x%04x 0x%04x 0x%04x 0x%04x 0x%04x\n",
            (uint16_t)buffer1[0], (uint16_t)buffer1[1], (uint16_t)buffer1[2],
            (uint16_t)buffer1[3], (uint16_t)buffer1[4]);
  updl_Info("      output_2[:5]: 0x%04x 0x%04x 0x%04x 0x%04x 0x%04x\n",
            (uint16_t)buffer2[0], (uint16_t)buffer2[1], (uint16_t)buffer2[2],
            (uint16_t)buffer2[3], (uint16_t)buffer2[4]);

  // Categorize differences into 3 buckets
  size_t exact_match = 0;
  size_t diff_1 = 0;
  size_t diff_2plus = 0;
  int32_t max_diff = 0;
  int16_t max_diff_val1 = 0;
  int16_t max_diff_val2 = 0;
  size_t first_mismatch_idx = 0;
  bool found_first_mismatch = false;

  for (size_t i = 0; i < size; i++) {
    int32_t diff = abs(buffer1[i] - buffer2[i]);

    // Categorize into buckets
    if (diff == 0) {
      exact_match++;
    } else if (diff == 1) {
      diff_1++;
    } else {
      diff_2plus++;
      if (!found_first_mismatch) {
        first_mismatch_idx = i;
        found_first_mismatch = true;
      }
    }

    if (diff > max_diff) {
      max_diff = diff;
      max_diff_val1 = buffer1[i];
      max_diff_val2 = buffer2[i];
    }
  }

  // Calculate percentages
  float matched_pct = (size > 0) ? (100.0f * exact_match / size) : 0.0f;
  float diff_1_pct = (size > 0) ? (100.0f * diff_1 / size) : 0.0f;
  float diff_2plus_pct = (size > 0) ? (100.0f * diff_2plus / size) : 0.0f;

  // Report distribution
  if (diff_2plus == 0) {
    updl_Info("  Layer %2d (%15s): PASS - matched=%d/%d(%.1f%%), "
              "abs(1)=%d/%d(%.1f%%), abs(2+)=%d/%d(%.1f%%), "
              "abs(max_error)=%d(0x%04x, 0x%04x)\n",
              layer_idx, layer_name, (int)exact_match, (int)size, matched_pct,
              (int)diff_1, (int)size, diff_1_pct, (int)diff_2plus, (int)size,
              diff_2plus_pct, max_diff, (uint16_t)max_diff_val1,
              (uint16_t)max_diff_val2);
    (*matched)++;
  } else {
    updl_Warning("  Layer %2d (%15s): FAIL - matched=%d/%d(%.1f%%), "
                 "abs(1)=%d/%d(%.1f%%), abs(2+)=%d/%d(%.1f%%), "
                 "abs(max_error)=%d(0x%04x, 0x%04x), "
                 "first@%d\n",
                 layer_idx, layer_name, (int)exact_match, (int)size,
                 matched_pct, (int)diff_1, (int)size, diff_1_pct,
                 (int)diff_2plus, (int)size, diff_2plus_pct, max_diff,
                 (uint16_t)max_diff_val1, (uint16_t)max_diff_val2,
                 (int)first_mismatch_idx);
    (*mismatched)++;
  }
}

// ============================================================================
// SHARED RUNNER FUNCTIONS
// ============================================================================

updl_test_layer_result_t
updl_test_run_layer_isolation(const updl_test_config_t *config,
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
  updl_test_metrics_t metrics = {0};

  bool passed = updl_validate_layer_output(
      config->int16_output_buffer, layer_golden->output_golden_fp32,
      layer_golden->output_size, output_scale, config->verbose,
      layer_golden->layer_name, &metrics);

  result.passed = passed;
  result.metrics = metrics;
  result.features_passed = metrics.matched + metrics.diff_1bit;
  result.features_failed = metrics.diff_2plus_bits;

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
      updl_validate_layer_output(
          output, golden->output_golden_fp32, golden->output_size, output_scale,
          ctx->verbose, golden->layer_name, &result->metrics);

      // Store pass/fail status and feature counts
      result->passed = (result->metrics.diff_2plus_bits == 0);
      result->total_features = golden->output_size;
      result->features_passed =
          result->metrics.matched + result->metrics.diff_1bit;
      result->features_failed = result->metrics.diff_2plus_bits;

      break; // Found and processed this layer
    }
  }
}

updl_test_sample_result_t
updl_test_run_inference_with_capture(const updl_test_config_t *config,
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
    if (free_input)
      free(input_int16);
    free(sample_result.layer_results);
    sample_result.layer_results = NULL;
    return sample_result;
  }

  // Set up callback context for layer capture
  layer_capture_context_t capture_ctx = {.sample = sample,
                                         .layer_results =
                                             sample_result.layer_results,
                                         .verbose = config->verbose,
                                         .model = config->model};

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
    if (free_input)
      free(input_int16);
    if (free_output)
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

  if (free_input)
    free(input_int16);
  if (free_output)
    free(output_int16);

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