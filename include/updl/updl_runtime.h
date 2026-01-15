/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#ifndef UPDL_RUNTIME_H
#define UPDL_RUNTIME_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <updl/updl_interpreter.h>
#include <updl/updl_operator.h>

typedef struct {
  updl_context_t *ctx;
  updl_model_t *model;
  updl_executor_t *executor;
  updl_memory_pool_t *memory_pool;
  uint8_t *tensor_arena;
  size_t tensor_arena_size;
  bool initialized;
} updl_runtime_handle_t;

int updl_runtime_setup(updl_runtime_handle_t *handle,
                       const uint8_t *model_data,
                       size_t tensor_arena_size,
                       size_t min_buffer_size);

int updl_runtime_cleanup(updl_runtime_handle_t *handle, bool full_cleanup);

#endif // UPDL_RUNTIME_H
