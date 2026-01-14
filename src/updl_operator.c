/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#include "updl/updl_operator.h"
#include "updl/updl_debug.h"
#include "updl/updl_kernels.h"
#include "updl/updl_utility.h"
#include "updl/updl_utility_tanh.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

updl_model_t *updl_load_model(updl_context_t *ctx, uint8_t *model_data) {
  if (!ctx || !model_data) {
    updl_Error("%s", "Invalid parameters for model loading\n");
    return NULL;
  }

  // Allocate model structure
  updl_model_t *model = (updl_model_t *)updl_alloc(ctx, sizeof(updl_model_t));
  if (!model)
    return NULL;

  model->context = ctx;
  uint8_t *fp = model_data;

  updl_Debug("Loading UPH5 Model from %p...\n", (void *)fp);

  // Parse model description
  strncpy(model->description, (const char *)fp, DESCRIPTION_LENGTH - 1);
  model->description[DESCRIPTION_LENGTH - 1] = '\0';
  fp += DESCRIPTION_LENGTH;
  updl_Debug("Model description: %s\n", model->description);

  // Parse model metadata
  updl_load_data(&model->model_name, &fp, Dtype_char, 1, "model_name",
                 TAG_FIELD, TAG_CHECK);
  updl_load_data(&model->num_layers, &fp, Dtype_uint16_t, 1, "num_layers",
                 TAG_FIELD, TAG_CHECK);
  updl_load_data(&model->batch_input_shape, &fp, Dtype_uint16_t, 4,
                 "batch_inputshape", TAG_FIELD, TAG_CHECK);
  updl_load_data(&model->dtype, &fp, Dtype_dtype_t, 1, "dtype", TAG_FIELD,
                 TAG_CHECK);
  updl_load_data(&model->input_scale, &fp, Dtype_float32_t, 1, "input_scale",
                 TAG_FIELD, TAG_CHECK);

  // Load buffer allocation metadata
  updl_load_data(&model->buffer_count, &fp, Dtype_uint16_t, 1, "buffer_count",
                 TAG_FIELD, TAG_CHECK);
  updl_load_data(&model->total_memory_bytes, &fp, Dtype_uint32_t, 1, "total_memory",
                 TAG_FIELD, TAG_CHECK);

  // Allocate and load buffer sizes
  model->buffer_sizes = (uint32_t *)updl_alloc(ctx, sizeof(uint32_t) * model->buffer_count);
  if (!model->buffer_sizes) {
    updl_Error("%s", "Failed to allocate buffer sizes array!\n");
    return NULL;
  }

  for (size_t i = 0; i < model->buffer_count; i++) {
    updl_load_data(&model->buffer_sizes[i], &fp, Dtype_uint32_t, 1, "buffer_size",
                   TAG_FIELD, TAG_CHECK);
  }

  // Allocate layer storage (simplified - single allocation)
  model->layers =
      (updl_layer_t *)updl_alloc(ctx, sizeof(updl_layer_t) * model->num_layers);
  if (!model->layers) {
    updl_Error("%s", "Failed to allocate layer storage!\n");
    return NULL;
  }

  // Initialize layer data
  memset(model->layers, 0, model->num_layers * sizeof(updl_layer_t));

  // Parse layer parameters
  for (size_t i = 0; i < model->num_layers; i++) {
    updl_layer_t *layer = &model->layers[i];
    layer->serial = i;
    updl_Debug("Layer %d @ 0x%08x\n", layer->serial + 1, layer);
    if (updl_load_layer_params(&layer, &fp) != 0) {
      updl_Error("Failed to load parameters for layer %d\n", i);
      return NULL;
    }
  }

  // Precompute quantization parameters for all layers (one-time calculation)
  updl_Debug("Precomputing quantization parameters...\n");
  for (size_t i = 0; i < model->num_layers; i++) {
    updl_layer_t *layer = &model->layers[i];

    // Quantization parameter chaining:
    // - First layer: Input comes from converted int8->int16 data (symmetric,
    // zp=0)
    // - Subsequent layers: Input is previous layer's output (all symmetric,
    // zp=0)
    float input_scale;
    int32_t input_zp;

    if (i == 0) {
      // Scale and zero-point from model's global input quantization parameters
      input_scale = model->input_scale;
      input_zp = 0;
    } else {
      // Subsequent layers: Chain from previous layer's output quantization
      input_scale = model->layers[i - 1].act_scale;
      input_zp = model->layers[i - 1].act_zp; // Zero for symmetric quantization
    }

    float weight_scale = layer->weight_scale;
    int32_t weight_zp = layer->weight_zp; // Zero for symmetric quantization
    float bias_scale = layer->bias_scale;
    int32_t bias_zp = layer->bias_zp; // Zero for symmetric quantization
    float output_scale = layer->act_scale;
    int32_t output_zp = layer->act_zp; // Zero for symmetric quantization

    // Compute effective requantization scale
    float eff_scale = (input_scale * weight_scale) / output_scale;
    requant_params_t eff = updl_scale_to_multiplier_shift(eff_scale);
    layer->effective_multiplier = eff.multiplier;
    layer->effective_shift = eff.shift;

#ifdef UPDL_UDL_SHIFT_ONLY_MODE
    // Verify UDL mode constraint: multiplier must be 1
    if (eff.multiplier != 1) {
      updl_Error("UDL mode violation: multiplier=%d (expected 1) "
                 "for eff_scale=%f\n",
                 eff.multiplier, eff_scale);
    }
#endif

    // Compute effective bias scale: bias_scale / (input_scale * weight_scale)
    float expected_bias_scale = input_scale * weight_scale;
    float eff_bias_scale = bias_scale / expected_bias_scale;

    requant_params_t eff_bias =
        updl_bias_scale_to_multiplier_shift(eff_bias_scale);

    // Calculate reconstructed scale from multiplier/shift
    float reconstructed_scale;
    if (eff_bias.shift >= 0) {
      reconstructed_scale = (float)eff_bias.multiplier / (1 << eff_bias.shift);
    } else {
      reconstructed_scale =
          (float)eff_bias.multiplier * (1 << (-eff_bias.shift));
    }

    float approximation_error = eff_bias_scale - reconstructed_scale;
    float relative_error = fabsf(approximation_error / eff_bias_scale);

    if (relative_error > 0.01) { // >1% error
      // INVESTIGATION: Check if power-of-2 approximation is causing bias error
      updl_Warning("Layer %d: Power-of-2 approximation analysis:\n", i);
      updl_Warning("  Original eff_bias_scale: %.8f\n", eff_bias_scale);
      updl_Warning("  Multiplier: %d, Shift: %d\n", eff_bias.multiplier,
                   eff_bias.shift);
      updl_Warning("  Reconstructed scale: %.8f\n", reconstructed_scale);
      updl_Warning("  Relative error: %.6f%%\n", relative_error * 100.0f);
    }

    layer->effective_bias_multiplier = eff_bias.multiplier;
    layer->effective_bias_shift = eff_bias.shift;

#ifdef UPDL_UDL_SHIFT_ONLY_MODE
    // Verify UDL mode constraint: bias multiplier must be 1
    if (eff_bias.multiplier != 1) {
      updl_Error("UDL bias mode violation: multiplier=%d "
                 "(expected 1) for eff_bias_scale=%f\n",
                 eff_bias.multiplier, eff_bias_scale);
    }
#endif

    // Store quantization parameters for kernel use
    layer->input_scale = input_scale; // Store input scale for layers that need
                                      // it (e.g., softmax)
    layer->input_zp = input_zp;
    layer->output_zp = output_zp;
    layer->weight_zp = weight_zp;

#if UPDL_ENABLE_DEBUG
    // Debug logging
    updl_print_eff_quant_param(i, eff_scale, input_scale, weight_scale,
                               output_scale, eff.multiplier, eff.shift,
                               eff_bias_scale, bias_scale, eff_bias.multiplier,
                               eff_bias.shift);
#endif
  }

  return model;
}

updl_executor_t *updl_create_executor(const updl_model_t *model,
                                      updl_memory_pool_t *memory_pool) {
  if (!model || !memory_pool) {
    updl_Error("%s", "Invalid parameters for executor creation\n");
    return NULL;
  }

  updl_context_t *ctx = model->context;
  if (!ctx) {
    updl_Error("%s", "Model context is NULL\n");
    return NULL;
  }

  // Allocate executor from context
  updl_Debug(
      "Creating executor: memory_pool=%p, memory_pool->max_buffer_size=%d\n",
      (void *)memory_pool, memory_pool->max_buffer_size);

  // Allocate executor from context (no more static limitation)
  updl_executor_t *executor = (updl_executor_t *)updl_alloc(ctx, sizeof(updl_executor_t));
  if (!executor) {
    updl_Error("%s", "Failed to allocate executor!\n");
    return NULL;
  }

  // Clear the executor structure
  memset(executor, 0, sizeof(updl_executor_t));

  updl_Debug("Executor allocated at %p\n", (void *)executor);

  executor->model = model;
  executor->memory_pool = memory_pool;
  executor->state = rstate_invalid;
  executor->current_layer = 0;

  updl_Debug("After assignment: executor->memory_pool=%p, max_buffer_size=%d\n",
             (void *)executor->memory_pool,
             executor->memory_pool->max_buffer_size);

  // Allocate activation buffers array from memory pool
  executor->activation_buffers = (int16_t **)updl_alloc(ctx, sizeof(int16_t *) * model->buffer_count);
  if (!executor->activation_buffers) {
    updl_Error("%s", "Failed to allocate activation buffers array!\n");
    return NULL;
  }

  // Allocate completion tracking array
  executor->layer_completed = (bool *)updl_alloc(ctx, sizeof(bool) * model->num_layers);
  if (!executor->layer_completed) {
    updl_Error("%s", "Failed to allocate layer completion array!\n");
    return NULL;
  }

  // Initialize completion tracking
  for (size_t i = 0; i < model->num_layers; i++) {
    executor->layer_completed[i] = false;
  }

  // Allocate each activation buffer based on static analysis
  for (size_t i = 0; i < model->buffer_count; i++) {
    size_t buffer_size_bytes = model->buffer_sizes[i];

    executor->activation_buffers[i] = (int16_t *)updl_alloc(ctx, buffer_size_bytes);
    if (!executor->activation_buffers[i]) {
      updl_Error("Failed to allocate activation buffer %d (%d bytes)\n", i, buffer_size_bytes);
      return NULL;
    }

    updl_Debug("Allocated buffer %d: %d bytes\n", i, buffer_size_bytes);
  }

  // Allocate execution layer contexts
  executor->exec_layers = (updl_exec_layer_t *)updl_alloc(ctx, sizeof(updl_exec_layer_t) * model->num_layers);
  if (!executor->exec_layers) {
    updl_Error("%s", "Failed to allocate execution layer contexts!\n");
    return NULL;
  }

  // Clear the execution layer contexts
  memset(executor->exec_layers, 0, sizeof(updl_exec_layer_t) * model->num_layers);

  for (size_t i = 0; i < model->num_layers; i++) {
    const updl_layer_t *layer = &model->layers[i];
    updl_exec_layer_t *exec_layer = &executor->exec_layers[i];

    // Calculate layer sizes
    exec_layer->input_size = 1;
    exec_layer->output_size = 1;

    for (int32_t j = 0; j < 4; j++) {
      if (layer->input_shape[j] > 0) {
        exec_layer->input_size *= layer->input_shape[j];
      }
      if (layer->output_shape[j] > 0) {
        exec_layer->output_size *= layer->output_shape[j];
      }
    }

    // Stream processing: all layers will use current input/output buffers
    // Actual pointers will be set during execution based on stream state
    exec_layer->input_ptr = NULL;  // Will be set during execution
    exec_layer->output_ptr = NULL; // Will be set during execution

    // Set up weight and bias pointers (const references to model data)
    exec_layer->weights = (const int16_t *)layer->weights.weight;
    exec_layer->bias = (const int16_t *)layer->bias.weight;

    // Calculate weight and bias sizes
    exec_layer->weight_size = 1;
    for (int32_t j = 0; j < layer->weights.weight_shape_d; j++) {
      if (layer->weights.weight_shape[j] > 0) {
        exec_layer->weight_size *= layer->weights.weight_shape[j];
      }
    }

    exec_layer->bias_size = 1;
    for (int32_t j = 0; j < layer->bias.weight_shape_d; j++) {
      if (layer->bias.weight_shape[j] > 0) {
        exec_layer->bias_size *= layer->bias.weight_shape[j];
      }
    }
  }

  executor->state = rstate_idle;

  return executor;
}

void updl_free_executor(updl_executor_t *executor) {
  if (executor) {
    updl_Debug("Freeing executor for model '%s'\n",
               executor->model ? executor->model->model_name : "unknown");
    // Note: Memory allocated via updl_alloc doesn't need individual freeing
    // It will be freed when the context is reset
    executor->state = rstate_invalid;
  }
}

int32_t updl_execute(updl_executor_t *executor, const void *input,
                     void *output) {
  if (!executor || !input || !output) {
    updl_Error(
        "Execute failed: NULL pointer (executor=%p, input=%p, output=%p)\n",
        (void *)executor, input, output);
    return -1;
  }

  if (!executor->memory_pool) {
    updl_Error("%s", "Execute failed: NULL memory pool in executor\n");
    return -1;
  }

  if (!executor->model) {
    updl_Error("%s", "Execute failed: NULL model in executor\n");
    return -1;
  }

  if (executor->state != rstate_idle) {
    const char *state_names[] = {"invalid", "unsupported", "idle",
                                 "running_soft", "running_hard"};
    const char *state_name = (executor->state >= 0 && executor->state <= 4)
                                 ? state_names[executor->state]
                                 : "unknown";

    // Only attempt recovery for certain states
    if (executor->state == rstate_invalid ||
        executor->state == rstate_running_soft) {
      updl_Warning("Executor state was %s, attempting recovery\n", state_name);
      if (updl_reset_executor(executor) != 0) {
        updl_Error("%s", "Failed to reset executor, cannot proceed\n");
        return -1;
      }
    } else {
      updl_Error("Executor not ready: state=%s (%d), expected=idle\n",
                 state_name, executor->state);
      return -1;
    }
  }

  const updl_model_t *model = executor->model;
  updl_memory_pool_t *pool = executor->memory_pool;

  updl_Debug("Execute starting: executor->memory_pool=%p, max_buffer_size=%d\n",
             (void *)pool, pool ? pool->max_buffer_size : 0);

  // Reset memory pool for fresh inference
  updl_reset_memory_pool(pool);

  // Initialize input data for graph execution
  if (model->num_layers > 0) {
    updl_exec_layer_t *first_exec_layer = &executor->exec_layers[0];
    size_t input_bytes = first_exec_layer->input_size * sizeof(int16_t);

    // Copy input data to buffer 0 (reserved for input)
    memcpy(executor->activation_buffers[0], input, input_bytes);

    // Print first few input values for debugging
#if UPDL_ENABLE_DEBUG
    const int16_t *input_data = (const int16_t *)input;
    uint32_t input_h = model->layers[0].input_shape[1];  // NHWC input shape
    uint32_t input_w = model->layers[0].input_shape[2];  // NHWC input shape
    uint32_t input_ch = model->layers[0].input_shape[3]; // NHWC input shape
    updl_print_2d_array("Input", input_data, input_h, input_w, 1,
                        model->input_scale, 0, 1);
#endif
  }

  executor->state = rstate_running_soft;

  // Execute layers using dependency-based graph execution
  size_t completed_layers = 0;
  while (completed_layers < model->num_layers) {
    bool made_progress = false;

    for (size_t i = 0; i < model->num_layers; i++) {
      if (executor->layer_completed[i]) {
        continue; // Layer already completed
      }

      const updl_layer_t *layer = &model->layers[i];
      updl_exec_layer_t *exec_layer = &executor->exec_layers[i];

      // Check if all dependencies are satisfied
      bool can_execute = true;
      for (uint16_t dep = 0; dep < layer->num_inputs; dep++) {
        uint16_t input_layer_idx = layer->input_layer_indices[dep];
        if (input_layer_idx != i && input_layer_idx < model->num_layers) {
          if (!executor->layer_completed[input_layer_idx]) {
            can_execute = false;
            break;
          }
        }
      }

      if (!can_execute) {
        continue; // Dependencies not ready
      }

      updl_Debug("Layer %d details: type=%d, input_size=%d, output_size=%d\n", i,
                 layer->type, exec_layer->input_size, exec_layer->output_size);

      executor->current_layer = i;

      // Set up static buffer pointers for current layer
      exec_layer->output_ptr = (uint16_t *)executor->activation_buffers[layer->buffer_id];

      // For multi-input layers (Add), handle multiple inputs
      if (layer->num_inputs > 1) {
        // For Add layers, input_ptr points to first input buffer
        uint16_t first_input_layer = layer->input_layer_indices[0];
        exec_layer->input_ptr = (uint16_t *)executor->activation_buffers[
            model->layers[first_input_layer].buffer_id];
      } else {
        // Single input layer
        if (i == 0) {
          // First layer uses input data (already copied to buffer 0)
          exec_layer->input_ptr = (uint16_t *)executor->activation_buffers[0];
        } else {
          uint16_t input_layer_idx = layer->input_layer_indices[0];
          exec_layer->input_ptr = (uint16_t *)executor->activation_buffers[
              model->layers[input_layer_idx].buffer_id];
        }
      }

    // Execute layer based on type
#if UPDL_ENABLE_DEBUG
    updl_print_layer_metadata(i, layer);
    uint32_t updl_start = updl_get_current_ticks();
#endif

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
      // Copy data from input to output buffer to maintain proper ping-pong
      // buffering Flatten is just a reshape operation - no computation, but we
      // need to copy for buffer consistency
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

#if UPDL_ENABLE_DEBUG
    updl_profile("layer execute", updl_start);

    // print weight
    if (i < 0) {
      uint32_t kernel_h = layer->kernel_size[0];
      uint32_t kernel_w = layer->kernel_size[1];
      int16_t *weight_data = exec_layer->weights;
      updl_print_2d_array("- weights", weight_data, kernel_h, kernel_w, 1,
                          layer->weight_scale, layer->weight_zp, 1);
    }
    if (i == 11) {
      uint32_t input_ch = layer->input_shape[1]; // NHWC input shape
      int16_t *weight_data = exec_layer->weights;
      updl_print_2d_array("- weights", weight_data, 2, input_ch, 1,
                          layer->weight_scale, layer->weight_zp, 1);
    }
    // print output
    if (i < 9) {
      size_t h = layer->output_shape[1];
      size_t w = layer->output_shape[2];
      size_t c = layer->output_shape[3];
      int16_t *output_data = (int16_t *)exec_layer->output_ptr;
      updl_print_2d_array("- output", output_data, h, w, c, layer->act_scale,
                          layer->act_zp, 2);
    } else if (i >= 9) {
      int16_t *output_data = (int16_t *)exec_layer->output_ptr;
      updl_print_1d_array("- output", output_data, exec_layer->output_size,
                          layer->act_scale, layer->act_zp, 5);
    }
#endif

      if (result != 0) {
        updl_Error("Layer %d execution failed with error %d\n", i, result);
        executor->state = rstate_invalid;
        return result;
      }

      // Call layer callback if registered
      if (executor->layer_callback) {
        executor->layer_callback(i, (const int16_t *)exec_layer->output_ptr,
                                 exec_layer->output_size,
                                 executor->callback_user_data);
      }

      // Mark layer as completed
      executor->layer_completed[i] = true;
      completed_layers++;
      made_progress = true;

      updl_Debug("Layer %d completed (%d/%d total)\n", i, completed_layers, model->num_layers);
    }

    // Check for deadlock (no progress made in this iteration)
    if (!made_progress) {
      updl_Error("Graph execution deadlock detected at layer iteration\n");
      executor->state = rstate_invalid;
      return -1;
    }
  }

  // Copy output data from final layer's buffer
  if (model->num_layers > 0) {
    updl_exec_layer_t *last_exec_layer = &executor->exec_layers[model->num_layers - 1];
    const updl_layer_t *last_layer = &model->layers[model->num_layers - 1];
    size_t output_bytes = last_exec_layer->output_size * sizeof(int16_t);

    // Copy from the last layer's assigned buffer
    int16_t *last_buffer = executor->activation_buffers[last_layer->buffer_id];
    memcpy(output, last_buffer, output_bytes);
  }

  executor->state = rstate_idle;
  return 0;
}

void updl_set_layer_callback(updl_executor_t *executor,
                             updl_layer_callback_t callback, void *user_data) {
  if (executor) {
    executor->layer_callback = callback;
    executor->callback_user_data = user_data;
  }
}
