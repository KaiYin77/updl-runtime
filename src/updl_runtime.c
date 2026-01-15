/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#include <updl/updl_runtime.h>

#include <stdlib.h>
#include <string.h>

#include <updl/updl_test_runner_utils.h>

int updl_runtime_setup(updl_runtime_handle_t *handle,
                       const uint8_t *model_data,
                       size_t tensor_arena_size,
                       size_t min_buffer_size) {
  if (!handle || !model_data || tensor_arena_size == 0) {
    return -1;
  }

  memset(handle, 0, sizeof(*handle));
  handle->tensor_arena = (uint8_t *)malloc(tensor_arena_size);
  if (!handle->tensor_arena) {
    return -1;
  }
  handle->tensor_arena_size = tensor_arena_size;

  handle->ctx = updl_init(handle->tensor_arena, tensor_arena_size);
  if (!handle->ctx) {
    updl_runtime_cleanup(handle, true);
    return -1;
  }

  handle->model = updl_load_model(handle->ctx, (uint8_t *)model_data);
  if (!handle->model) {
    updl_runtime_cleanup(handle, true);
    return -1;
  }

  size_t max_layer_size = min_buffer_size;
  for (uint16_t i = 0; i < handle->model->num_layers; i++) {
    const updl_layer_t *layer = &handle->model->layers[i];
    size_t input_size = updl_calc_shape_size(layer->input_shape);
    size_t output_size = updl_calc_shape_size(layer->output_shape);

    if (input_size > max_layer_size) {
      max_layer_size = input_size;
    }
    if (output_size > max_layer_size) {
      max_layer_size = output_size;
    }
  }

  handle->executor = updl_create_executor(handle->model, NULL);
  if (!handle->executor) {
    updl_runtime_cleanup(handle, true);
    return -1;
  }

  size_t available_memory =
      handle->ctx->memory_size - handle->ctx->memory_offset;
  uint8_t *pool_memory = handle->ctx->memory_base + handle->ctx->memory_offset;

  handle->memory_pool =
      updl_create_memory_pool(pool_memory, available_memory, max_layer_size,
                              handle->model);
  if (!handle->memory_pool) {
    updl_runtime_cleanup(handle, true);
    return -1;
  }
  handle->executor->memory_pool = handle->memory_pool;

  handle->initialized = true;
  return 0;
}

int updl_runtime_cleanup(updl_runtime_handle_t *handle, bool full_cleanup) {
  if (!handle) {
    return -1;
  }

  if (!handle->initialized) {
    return -1;
  }

  if (full_cleanup) {
    if (handle->executor) {
      updl_free_executor(handle->executor);
    }

    handle->executor = NULL;
    handle->model = NULL;
    handle->memory_pool = NULL;
    handle->ctx = NULL;
    handle->initialized = false;

    if (handle->tensor_arena) {
      free(handle->tensor_arena);
      handle->tensor_arena = NULL;
      handle->tensor_arena_size = 0;
    }

    return 0;
  }

  if (!handle->executor) {
    return -1;
  }

  if (updl_reset_executor(handle->executor) != 0) {
    return -1;
  }

  return 0;
}
