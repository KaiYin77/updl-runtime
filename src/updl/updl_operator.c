/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#include "updl/updl_operator.h"
#include "updl/updl_kernels.h"
#include "updl/updl_utility.h"
#include "updl/updl_utility_tanh.h"
#include "updl/updl_debug.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

updl_model_t *updl_load_model(updl_context_t *ctx, uint8_t *model_data) {
  if (!ctx || !model_data) {
    updl_Error("Invalid parameters for model loading\n");
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
  updl_load_data(&model->input_scale, &fp, Dtype_float32_t, 1, "input_scale", TAG_FIELD,
                 TAG_CHECK);

  // Allocate layer storage (simplified - single allocation)
  model->layers =
      (updl_layer_t *)updl_alloc(ctx, sizeof(updl_layer_t) * model->num_layers);
  if (!model->layers) {
    updl_Error("Failed to allocate layer storage!\n");
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
    // - First layer: Input comes from converted int8->int16 data (symmetric, zp=0)
    // - Subsequent layers: Input is previous layer's output (all symmetric, zp=0)
    float input_scale;
    int32_t input_zp;
    
    if (i == 0) {
      // Scale and zero-point from model's global input quantization parameters
      input_scale = model->input_scale;
      input_zp = 0;                  
    } else {
      // Subsequent layers: Chain from previous layer's output quantization  
      input_scale = model->layers[i - 1].act_scale;
      input_zp = model->layers[i - 1].act_zp;  // Zero for symmetric quantization
    }

    float weight_scale = layer->weight_scale;
    int32_t weight_zp  = layer->weight_zp;    // Zero for symmetric quantization
    float bias_scale = layer->bias_scale;
    int32_t bias_zp = layer->bias_zp;         // Zero for symmetric quantization
    float output_scale = layer->act_scale;
    int32_t output_zp  = layer->act_zp;       // Zero for symmetric quantization

    // Compute effective requantization scale
    float eff_scale = (input_scale * weight_scale) / output_scale;
    requant_params_t eff = updl_scale_to_multiplier_shift(eff_scale);
    layer->effective_multiplier = eff.multiplier;
    layer->effective_shift = eff.shift;
    
#ifdef UPDL_UDL_SHIFT_ONLY_MODE
    // Verify UDL mode constraint: multiplier must be 1
    if (eff.multiplier != 1) {
        updl_Error("[UPDL][ERROR] UDL mode violation: multiplier=%d (expected 1) for eff_scale=%f\n", 
                   eff.multiplier, eff_scale);
    }
#endif 
    
    // Compute effective bias scale: bias_scale / (input_scale * weight_scale)
    float eff_bias_scale = bias_scale / (input_scale * weight_scale);
    requant_params_t eff_bias = updl_bias_scale_to_multiplier_shift(eff_bias_scale);
    layer->effective_bias_multiplier = eff_bias.multiplier;
    layer->effective_bias_shift = eff_bias.shift;
    
#ifdef UPDL_UDL_SHIFT_ONLY_MODE
    // Verify UDL mode constraint: bias multiplier must be 1
    if (eff_bias.multiplier != 1) {
        updl_Error("[UPDL][ERROR] UDL bias mode violation: multiplier=%d (expected 1) for eff_bias_scale=%f\n", 
                   eff_bias.multiplier, eff_bias_scale);
    }
#endif
    
    // Store quantization parameters for kernel use
    layer->input_zp = input_zp;
    layer->output_zp = output_zp;
    layer->weight_zp = weight_zp;
    
#if UPDL_ENABLE_DEBUG    
    // Debug logging
    updl_print_eff_quant_param(i, eff_scale, input_scale, weight_scale, output_scale,
                                  eff.multiplier, eff.shift, eff_bias_scale, bias_scale,
                                  eff_bias.multiplier, eff_bias.shift);
#endif  
  }

  return model;
}

updl_executor_t *updl_create_executor(const updl_model_t *model,
                                      updl_memory_pool_t *memory_pool) {
  if (!model || !memory_pool) {
    updl_Error("Invalid parameters for executor creation\n");
    return NULL;
  }

  // Allocate executor from context
  updl_Debug(
      "Creating executor: memory_pool=%p, memory_pool->max_buffer_size=%d\n",
      (void *)memory_pool, memory_pool->max_buffer_size);

  // Use static allocation to avoid memory overlap with memory pool
  static updl_executor_t static_executor;
  updl_executor_t *executor = &static_executor;
  // Clear the static executor structure
  for (int i = 0; i < sizeof(updl_executor_t); i++) {
    ((uint8_t *)executor)[i] = 0;
  }

  updl_Debug("Executor allocated at %p (static)\n", (void *)executor);

  executor->model = model;
  executor->memory_pool = memory_pool;
  executor->state = rstate_invalid;
  executor->current_layer = 0;
  
  updl_Debug(
      "After assignment: executor->memory_pool=%p, max_buffer_size=%d\n",
      (void *)executor->memory_pool, executor->memory_pool->max_buffer_size);

  // Allocate execution layer contexts using static allocation to avoid overlap
  static updl_exec_layer_t static_exec_layers[24]; // Max 16 layers
  if (model->num_layers > 24) {
    updl_Error("Too many layers (%d), maximum supported is 24\n",
               model->num_layers);
    return NULL;
  }

  executor->exec_layers = static_exec_layers;
  // Clear the static exec layers structure
  for (int i = 0; i < sizeof(static_exec_layers); i++) {
    ((uint8_t *)static_exec_layers)[i] = 0;
  }

  for (size_t i = 0; i < model->num_layers; i++) {
    const updl_layer_t *layer = &model->layers[i];
    updl_exec_layer_t *exec_layer = &executor->exec_layers[i];

    // Calculate layer sizes
    exec_layer->input_size = 1;
    exec_layer->output_size = 1;

    for (int j = 0; j < 4; j++) {
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
    for (int j = 0; j < layer->weights.weight_shape_d; j++) {
      if (layer->weights.weight_shape[j] > 0) {
        exec_layer->weight_size *= layer->weights.weight_shape[j];
      }
    }

    exec_layer->bias_size = 1;
    for (int j = 0; j < layer->bias.weight_shape_d; j++) {
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

int updl_execute(updl_executor_t *executor, const void *input, void *output) {
  if (!executor || !input || !output) {
    updl_Error(
        "Execute failed: NULL pointer (executor=%p, input=%p, output=%p)\n",
        (void *)executor, input, output);
    return -1;
  }

  if (!executor->memory_pool) {
    updl_Error("Execute failed: NULL memory pool in executor\n");
    return -1;
  }

  if (!executor->model) {
    updl_Error("Execute failed: NULL model in executor\n");
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
        updl_Error("Failed to reset executor, cannot proceed\n");
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

  updl_Debug(
      "Execute starting: executor->memory_pool=%p, max_buffer_size=%d\n",
      (void *)pool, pool ? pool->max_buffer_size : 0);

  // Reset memory pool for fresh inference
  updl_reset_memory_pool(pool);

  // Initialize stream processing with input data
  if (model->num_layers > 0) {
    updl_exec_layer_t *first_exec_layer = &executor->exec_layers[0];
    size_t input_bytes = first_exec_layer->input_size * sizeof(int16_t);
    size_t output_bytes = first_exec_layer->output_size * sizeof(int16_t);
    // Ensure stream buffers can handle first layer
    if (updl_ensure_buffer_capacity(pool, input_bytes, output_bytes) != 0) {
      executor->state = rstate_invalid;
      return -1;
    }

    // Copy input data to current input buffer
    memcpy(updl_get_input_buffer(pool), input, input_bytes);
      
    // Print first few input values for debugging
#if UPDL_ENABLE_DEBUG
    const int16_t *input_data = (const int16_t *)input;
    uint32_t input_h = model->layers[0].input_shape[1];   // NHWC input shape
    uint32_t input_w = model->layers[0].input_shape[2];   // NHWC input shape
    uint32_t input_ch = model->layers[0].input_shape[3];   // NHWC input shape
    updl_print_2d_array("Input", input_data, input_h, input_w, 1, model->input_scale, 0, 1);
#endif
  }

  executor->state = rstate_running_soft;

  // Execute each layer with stream processing
  for (size_t i = 0; i < model->num_layers; i++) {

    const updl_layer_t *layer = &model->layers[i];
    updl_exec_layer_t *exec_layer = &executor->exec_layers[i];
    
    updl_Debug("Layer %d details: type=%d, input_size=%d, output_size=%d\n", 
               i, layer->type, exec_layer->input_size, exec_layer->output_size);

    executor->current_layer = i;

    // Set up stream buffers for current layer
    size_t input_bytes = exec_layer->input_size * sizeof(int16_t);
    size_t output_bytes = exec_layer->output_size * sizeof(int16_t);

    // Ensure capacity for current layer (usually no-op after first layer)
    if (i > 0) {
      if (updl_ensure_buffer_capacity(pool, input_bytes, output_bytes) != 0) {
        executor->state = rstate_invalid;
        return -1;
      }
    }

    // Set current stream buffer pointers for this layer
    exec_layer->input_ptr = (uint16_t *) updl_get_input_buffer(pool);
    exec_layer->output_ptr = (uint16_t *) updl_get_output_buffer(pool);

    // Execute layer based on type
#if UPDL_ENABLE_DEBUG
    updl_print_layer_metadata(i, layer);
    uint32_t updl_start = updl_get_current_ticks();
#endif 

    int result = 0;
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
      // Copy data from input to output buffer to maintain proper ping-pong buffering
      // Flatten is just a reshape operation - no computation, but we need to copy for buffer consistency
      if (exec_layer->input_size > 0) {
        memcpy(exec_layer->output_ptr, exec_layer->input_ptr, exec_layer->input_size * sizeof(int16_t));
      }
      result = 0;
      break;
    case Ltype_lambda:
      result = updl_l2_norm(layer, exec_layer);
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
  if(i < 0) {
    uint32_t kernel_h = layer->kernel_size[0];
    uint32_t kernel_w = layer->kernel_size[1]; 
    int16_t *weight_data = exec_layer->weights;
    updl_print_2d_array("- weights", weight_data, kernel_h, kernel_w, 1, 
                        layer->weight_scale, layer->weight_zp, 1);
  }
  if(i == 11) {
    uint32_t input_ch = layer->input_shape[1];   // NHWC input shape
    int16_t *weight_data = exec_layer->weights;
    updl_print_2d_array("- weights", weight_data, 2, input_ch, 1, 
                        layer->weight_scale, layer->weight_zp, 1);
  }
  // print output
  if(i < 9) {
    size_t h = layer->output_shape[1];
    size_t w = layer->output_shape[2];
    size_t c = layer->output_shape[3];
    int16_t *output_data = (int16_t*)exec_layer->output_ptr;
    updl_print_2d_array("- output", output_data, h, w, c, layer->act_scale, layer->act_zp, 2);
  } else if (i >= 9) {
    int16_t *output_data = (int16_t*)exec_layer->output_ptr;
    updl_print_1d_array("- output", output_data, exec_layer->output_size, 
                        layer->act_scale, layer->act_zp, 5);
  }
#endif

    if (result != 0) {
      updl_Error("Layer %d execution failed with error %d\n", i, result);
      executor->state = rstate_invalid;
      return result;
    }

    // Swap buffers for next layer (output becomes input)
    if (i < model->num_layers - 1) {
      updl_Debug("Swapping buffers for next layer (current layer: %d)\n", i);
      updl_swap_stream_buffers(pool);
    } 
  }

  // Copy output data from current output buffer
  if (model->num_layers > 0) {
    updl_exec_layer_t *last_exec_layer =
        &executor->exec_layers[model->num_layers - 1];
    size_t output_bytes = last_exec_layer->output_size * sizeof(int16_t);
    memcpy(output, updl_get_output_buffer(pool), output_bytes);
  }

  executor->state = rstate_idle;
  return 0;
}
