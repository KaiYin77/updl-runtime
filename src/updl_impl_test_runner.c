/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file updl_impl_test_runner.c
 * @brief Generic UPDL Implementation Test Runner
 *
 * This module provides model-agnostic utilities for comparing hardware (UDL)
 * and software-only implementations layer by layer.
 */

// ============================================================================
// INCLUDES
// ============================================================================

#include <updl/updl_impl_test_runner.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <updl/updl_kernels.h>
#include <updl/updl_operator.h>
#include <updl/updl_test_runner_utils.h>
#include <updl/updl_utility.h>

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Check if a layer type has hardware acceleration support
 */
static bool updl_layer_has_hw_support(ltype_t type) {
  switch (type) {
  case Ltype_conv_2d:
  case Ltype_depthwise_conv_2d:
  case Ltype_dense:
    return true;
  default:
    return false;
  }
}

// ============================================================================
// HIGH-LEVEL TEST RUNNER
// ============================================================================

updl_impl_test_stats_t
updl_run_implementation_tests(const updl_impl_test_config_t *config) {
  updl_impl_test_stats_t stats = {0};

  if (!config || !config->model || !config->executor || !config->test_inputs ||
      config->num_test_inputs == 0) {
    updl_Error("ERROR: Invalid implementation test configuration\n");
    return stats;
  }

  updl_model_t *model = config->model;
  updl_executor_t *executor = config->executor;

  updl_Info("\n");
  updl_Info("========================================\n");
  updl_Info("UPDL Implementation Test: UDL-accelerated vs Pure-software\n");
  updl_Info("Layer-by-Layer Comparison\n");
  updl_Info("========================================\n");
  updl_Info("Model: %s\n", model->model_name);
  updl_Info("Total layers: %d\n", model->num_layers);
  updl_Info("Test samples: %d\n", (int)config->num_test_inputs);

  // Find maximum layer output size for buffer allocation
  size_t max_layer_size = 0;
  for (uint16_t i = 0; i < model->num_layers; i++) {
    updl_exec_layer_t *exec_layer = &executor->exec_layers[i];
    if (exec_layer->output_size > max_layer_size) {
      max_layer_size = exec_layer->output_size;
    }
  }

  updl_Info("Max layer size: %d int16 elements (%.1f KB)\n", (int)max_layer_size,
            (max_layer_size * sizeof(int16_t)) / 1024.0f);
  updl_Info("Memory usage: 3 buffers x %.1f KB = %.1f KB total\n",
            (max_layer_size * sizeof(int16_t)) / 1024.0f,
            (3 * max_layer_size * sizeof(int16_t)) / 1024.0f);
  updl_Info("\n");

  // Allocate 3 buffers: UDL state, SW state, and temporary output
  int16_t *buffer_hw = (int16_t *)malloc(max_layer_size * sizeof(int16_t));
  int16_t *buffer_sw = (int16_t *)malloc(max_layer_size * sizeof(int16_t));
  int16_t *buffer_temp = (int16_t *)malloc(max_layer_size * sizeof(int16_t));

  if (!buffer_hw || !buffer_sw || !buffer_temp) {
    updl_Error("ERROR: Failed to allocate buffers (need %.1f KB)\n",
               (3 * max_layer_size * sizeof(int16_t)) / 1024.0f);
    free(buffer_hw);
    free(buffer_sw);
    free(buffer_temp);
    return stats;
  }

  // Test each input sample
  for (size_t sample_idx = 0; sample_idx < config->num_test_inputs;
       sample_idx++) {
    const updl_impl_test_input_t *test_input = &config->test_inputs[sample_idx];
    updl_Info("Sample %d/%d:\n", (int)(sample_idx + 1),
              (int)config->num_test_inputs);

    // Quantize input to int16 using model's quantization parameters
    size_t input_size = test_input->input_size;
    int16_t *input_int16 = (int16_t *)malloc(input_size * sizeof(int16_t));
    if (!input_int16) {
      updl_Error("ERROR: Failed to allocate input buffer\n");
      continue;
    }

    // Get input quantization parameters from model
    float input_scale = model->input_scale;
    int16_t input_zp = 0; // Typically 0 for inputs

    updl_quantize_fp32_array(test_input->input_fp32, input_int16, input_size,
                             input_scale, input_zp);

    // Initialize both UDL and SW buffers with the same input
    memcpy(buffer_hw, input_int16, input_size * sizeof(int16_t));
    memcpy(buffer_sw, input_int16, input_size * sizeof(int16_t));

    // Debug: Verify buffer_hw and buffer_sw contain same data (first sample only)
    if (sample_idx == 0 && config->verbose) {
      updl_Info("     udl_input[:5]:   0x%04x 0x%04x 0x%04x 0x%04x 0x%04x\n",
                (uint16_t)buffer_hw[0], (uint16_t)buffer_hw[1],
                (uint16_t)buffer_hw[2], (uint16_t)buffer_hw[3],
                (uint16_t)buffer_hw[4]);
      
      // Verify they're identical
      bool buffers_match =
          (memcmp(buffer_hw, buffer_sw, input_size * sizeof(int16_t)) == 0);
      updl_Info("  Buffers match before Layer 0: %s\n",
                buffers_match ? "YES" : "NO");
    }

    bool sample_passed = true;

    // ====================================================================
    // Process each layer: UDL and SW independently using shared temp buffer
    // ====================================================================
    for (uint16_t layer_idx = 0; layer_idx < model->num_layers; layer_idx++) {
      const updl_layer_t *layer = &model->layers[layer_idx];
      updl_exec_layer_t *exec_layer = &executor->exec_layers[layer_idx];
      bool has_hw_support = updl_layer_has_hw_support(layer->type);

      // Execute with HARDWARE (or SW if no HW support): buffer_hw (input) -> buffer_temp (output)
      executor->state = rstate_running_hard;

      int hw_result = updl_execute_single_layer(executor, layer_idx, buffer_hw,
                                                buffer_temp);

      if (hw_result != 0) {
        const char *mode = has_hw_support ? "HW" : "SW";
        updl_Error("  Layer %2d: ERROR - %s execution failed (result=%d)\n",
                   layer_idx, mode, hw_result);
        sample_passed = false;
        stats.layers_mismatched++;
        break;
      }

      // Copy result back to buffer_hw for next layer
      size_t output_bytes = exec_layer->output_size * sizeof(int16_t);
      memcpy(buffer_hw, buffer_temp, output_bytes);

      // Execute with SOFTWARE: buffer_sw (input) -> buffer_temp (output)
      executor->state = rstate_running_soft;

      int sw_result = updl_execute_single_layer(executor, layer_idx, buffer_sw,
                                                buffer_temp);

      if (sw_result != 0) {
        updl_Error("  Layer %2d: ERROR - SW execution failed (result=%d)\n",
                   layer_idx, sw_result);
        sample_passed = false;
        stats.layers_mismatched++;
        break;
      }

      // Print comparison labels based on HW support
      const char *layer_name = updl_get_layer_type_name(layer->type);
      if (has_hw_support) {
        updl_Info("  Comparing HW vs SW for layer %2d (%15s):\n", layer_idx,
                  layer_name);
      } else {
        updl_Info("  SW-only layer %2d (%15s):\n", layer_idx, layer_name);
      }

      // Compare immediately: buffer_hw vs buffer_temp (SW result)
      updl_compare_int16_buffers(layer_name, layer_idx, buffer_hw, buffer_temp,
                                 exec_layer->output_size, &stats.layers_matched,
                                 &stats.layers_mismatched);

      stats.total_layers_compared++;

      if (memcmp(buffer_hw, buffer_temp, output_bytes) != 0) {
        sample_passed = false;
      }

      // Copy SW result back to buffer_sw for next layer
      memcpy(buffer_sw, buffer_temp, output_bytes);
    }

    if (sample_passed) {
      updl_Info("  -> Sample PASSED: All layers match\n");
    } else {
      updl_Error("  -> Sample FAILED: Found mismatches\n");
    }
    updl_Info("\n");

    free(input_int16);
  }

  // ====================================================================
  // Print summary
  // ====================================================================
  updl_Info("========================================\n");
  updl_Info("Implementation Test Summary\n");
  updl_Info("========================================\n");
  updl_Info("Total layers compared: %u\n", stats.total_layers_compared);

  if (stats.layers_mismatched == 0) {
    updl_Info("Layers matched:        %u (%.1f%%)\n", stats.layers_matched,
              100.0f * stats.layers_matched /
                  (float)stats.total_layers_compared);

  } else {
    updl_Error("Layers mismatched:     %u (%.1f%%)\n", stats.layers_mismatched,
               100.0f * stats.layers_mismatched /
                   (float)stats.total_layers_compared);
  }
  updl_Info("========================================\n");

  // Cleanup
  free(buffer_hw);
  free(buffer_sw);
  free(buffer_temp);

  // Reset executor to default state
  executor->state = rstate_idle;
  executor->current_layer = 0;

  return stats;
}
