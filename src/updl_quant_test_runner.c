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

      // Determine number of inputs (default 1)
      uint16_t num_inputs = config->num_inputs;
      if (num_inputs == 0) {
        num_inputs = 1;
      }
      layer_golden->num_inputs = num_inputs > 2 ? 2 : num_inputs;

      // Set input golden reference (primary)
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

      // Optional second input
      if (layer_golden->num_inputs > 1 &&
          config->second_input_golden_data != NULL) {
        layer_golden->second_input_golden_fp32 =
            (const float *)config->second_input_golden_data +
            (sample_idx * config->second_input_size);
        layer_golden->second_input_size = config->second_input_size;
      } else {
        layer_golden->second_input_golden_fp32 = NULL;
        layer_golden->second_input_size = 0;
      }

      // Set output golden reference
      layer_golden->output_golden_fp32 =
          (const float *)config->output_golden_data +
          (sample_idx * config->output_size);
      layer_golden->output_size = config->output_size;
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

  if (!config->int16_input_buffer || !config->int16_output_buffer) {
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

  // Create a unified config wrapper to pass to shared functions
  updl_test_config_t unified_config = {
      .model = config->model,
      .executor = config->executor,
      .verbose = config->verbose,
      .int16_input_buffer = config->int16_input_buffer,
      .int16_input_buffer_size = config->int16_input_buffer_size,
      .int16_output_buffer = config->int16_output_buffer,
      .int16_output_buffer_size = config->int16_output_buffer_size
      // dequant buffer not needed for isolation tests
  };

  // Run tests for each sample
  for (size_t i = 0; i < config->num_samples; i++) {
    const updl_quant_test_sample_t *sample = &config->samples[i];
    updl_sample_quant_result_t *sample_result = &report->sample_results[i];
    
    sample_result->sample_index = (uint32_t)i;
    sample_result->num_layers = sample->num_layers;
    sample_result->layer_results = (updl_layer_quant_result_t *)calloc(
        sample->num_layers, sizeof(updl_layer_quant_result_t));
        
    if (config->verbose) {
      updl_Info("Sample %d/%d:\n", (int)(i + 1), (int)config->num_samples);
    }

    for (size_t j = 0; j < sample->num_layers; j++) {
        const updl_layer_quant_golden_t *layer_golden = &sample->layers[j];
        
        // Map to unified layer golden struct
        updl_test_layer_golden_t unified_golden = {
            .layer_name = layer_golden->layer_name,
            .layer_index = layer_golden->layer_index,
            .input_golden_fp32 = layer_golden->input_golden_fp32,
            .input_size = layer_golden->input_size,
            .second_input_golden_fp32 = layer_golden->second_input_golden_fp32,
            .second_input_size = layer_golden->second_input_size,
            .num_inputs = layer_golden->num_inputs,
            .output_golden_fp32 = layer_golden->output_golden_fp32,
            .output_size = layer_golden->output_size
        };
        
        // Determine input for this layer
        const float *layer_input_fp32;
        if (layer_golden->layer_index == 0) {
          // First layer uses model input
          layer_input_fp32 = sample->model_input_fp32;
        } else {
          // Other layers use their golden input
          layer_input_fp32 = layer_golden->input_golden_fp32;
        }
        
        // Run shared isolation test
        updl_test_layer_result_t result = updl_test_run_layer_isolation(
            &unified_config, &unified_golden, layer_input_fp32);
            
        // Map result back
        sample_result->layer_results[j].layer_name = result.layer_name;
        sample_result->layer_results[j].layer_index = result.layer_index;
        sample_result->layer_results[j].passed = result.passed;
        sample_result->layer_results[j].total_features = result.total_features;
        sample_result->layer_results[j].features_passed = result.features_passed;
        sample_result->layer_results[j].features_failed = result.features_failed;
        
        if (result.passed) {
            sample_result->layers_passed++;
        } else {
            sample_result->layers_failed++;
        }
    }
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
  free(config);
}
