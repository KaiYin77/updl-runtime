/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

// ============================================================================
// INCLUDES
// ============================================================================

#include "updl/updl_utility.h"
#include "updl/updl_interpreter.h"
#include "updl/updl_operator.h"
#include <string.h>
#include <stdlib.h>  // For malloc/free and aligned_alloc (C11)
#include <stdint.h>  // For uintptr_t
#ifdef _WIN32
#include <malloc.h>  // For _aligned_malloc/_aligned_free on Windows
#endif

// ============================================================================
// MEMORY CONTEXT MANAGEMENT
// ============================================================================

void *updl_alloc(updl_context_t *ctx, size_t size) {
  if (!ctx || size == 0) {
    updl_Error("Invalid allocation parameters\n");
    return NULL;
  }

  // Align size to 8-byte boundary for embedded systems
  size = (size + 7) & ~7;

  if (ctx->memory_offset + size > ctx->memory_size) {
    updl_Error("Memory allocation failed: need %d bytes, available %d\n",
               size, ctx->memory_size - ctx->memory_offset);
    return NULL;
  }

  void *ptr = ctx->memory_base + ctx->memory_offset;
  ctx->memory_offset += size;

  // Zero-initialize allocated memory to prevent garbage data
  memset(ptr, 0, size);

  return ptr;
}

updl_context_t *updl_init(uint8_t *memory, size_t memory_size) {
  if (!memory) {
    updl_Error("Invalid memory parameters\n");
    return NULL;
  }

  updl_context_t *ctx = (updl_context_t *)memory;

  ctx->memory_base = memory + sizeof(updl_context_t);
  ctx->memory_size = memory_size - sizeof(updl_context_t);
  ctx->memory_offset = 0;

  return ctx;
}

// ============================================================================
// MEMORY POOL MANAGEMENT
// ============================================================================

updl_memory_pool_t *updl_create_memory_pool(uint8_t *memory, size_t total_size,
                                            size_t max_layer_size,
                                            const updl_model_t *model) {
  if (!memory || total_size < sizeof(updl_memory_pool_t)) {
    updl_Error("Invalid memory pool parameters\n");
    return NULL;
  }

  updl_memory_pool_t *pool = (updl_memory_pool_t *)memory;

  pool->base_address = memory + sizeof(updl_memory_pool_t);
  pool->total_size = total_size - sizeof(updl_memory_pool_t);
  pool->used_size = 0;

  // Calculate buffer sizes for stream processing - only 2 buffers total
  size_t max_buffer_size = max_layer_size * sizeof(int16_t);

  // Ensure minimum buffer size for safety
  if (max_buffer_size < 1024 * sizeof(int16_t)) {
    max_buffer_size = 1024 * sizeof(int16_t); // Minimum 1K elements = 2KB
    updl_Debug("Using minimum buffer size: %d bytes\n", max_buffer_size);
  }

  size_t total_stream_buffers = max_buffer_size * 2; // Input + Output
  
  {
    size_t required_memory = total_stream_buffers;
    if (required_memory > pool->total_size) {
      updl_Error("Insufficient memory for stream buffers: need %d bytes, have %d bytes\n",
                 required_memory, pool->total_size);
      return NULL;
    }
    pool->max_buffer_size = max_buffer_size;
  }

  // Set up memory regions - stream processing buffers
  pool->input_buffer = pool->base_address;
  pool->output_buffer = pool->base_address + pool->max_buffer_size;
  pool->current_input_size = 0;
  pool->current_output_size = 0;
  pool->buffers_swapped = false;

  return pool;
}

void updl_reset_memory_pool(updl_memory_pool_t *pool) {
  if (!pool) {
    updl_Error("Cannot reset NULL memory pool\n");
    return;
  }

  pool->used_size = 0;
  pool->current_input_size = 0;
  pool->current_output_size = 0;
  pool->buffers_swapped = false;
  
  // Weight caching removed - no cleanup needed

  // Clear stream buffers for fresh inference
  if (pool->input_buffer && pool->max_buffer_size > 0) {
    memset(pool->input_buffer, 0, pool->max_buffer_size);
  }
  if (pool->output_buffer && pool->max_buffer_size > 0) {
    memset(pool->output_buffer, 0, pool->max_buffer_size);
  }
}

// ============================================================================
// STREAM BUFFER MANAGEMENT
// ============================================================================

int updl_ensure_buffer_capacity(updl_memory_pool_t *pool, size_t input_size,
                                size_t output_size) {
  if (!pool) {
    updl_Error("NULL memory pool\n");
    return -1;
  }

  // Check if current buffers are large enough
  if (input_size <= pool->max_buffer_size &&
      output_size <= pool->max_buffer_size) {
    pool->current_input_size = input_size;
    pool->current_output_size = output_size;
    return 0; // No reallocation needed
  }

  updl_Error("Buffer capacity exceeded: need input=%d, output=%d, max=%d\n",
             input_size, output_size, pool->max_buffer_size);
  return -1;
}

void updl_swap_stream_buffers(updl_memory_pool_t *pool) {
  if (!pool)
    return;

  // Swap input and output buffers by swapping pointers
  uint8_t *temp = pool->input_buffer;
  pool->input_buffer = pool->output_buffer;
  pool->output_buffer = temp;

  // Update sizes
  pool->current_input_size = pool->current_output_size;
  pool->current_output_size = 0; // Will be set by next layer

  pool->buffers_swapped = !pool->buffers_swapped;
}

uint8_t *updl_get_input_buffer(updl_memory_pool_t *pool) {
  return pool ? pool->input_buffer : NULL;
}

uint8_t *updl_get_output_buffer(updl_memory_pool_t *pool) {
  return pool ? pool->output_buffer : NULL;
}


// ============================================================================
// EXECUTOR RECOVERY FUNCTIONS
// ============================================================================

int updl_reset_executor(updl_executor_t *executor) {
  if (!executor) {
    updl_Error("Cannot reset NULL executor\n");
    return -1;
  }

  if (!executor->memory_pool) {
    updl_Error("Cannot reset executor with NULL memory pool\n");
    return -1;
  }

  // Reset memory pool to clean state
  updl_reset_memory_pool(executor->memory_pool);

  // Reset executor state
  executor->state = rstate_idle;
  executor->current_layer = 0;

  return 0;
}

// ============================================================================
// PROFILING FUNCTIONS
// ============================================================================

uint32_t updl_get_current_ticks(void) {
  return *((volatile uint32_t *)0x0200BFF8);
}

void updl_profile(const char *label, uint32_t start_time) {
  uint32_t end_time = *((volatile uint32_t *)0x0200BFF8);
  uint32_t ticks = end_time - start_time;
  uint32_t time_us = ticks * 30; /* 1 tick = 30us */

  /* Calculate milliseconds using integer math */
  uint32_t time_ms = time_us / 1000;

  /* If you want to show 1 decimal place without using float */
  uint32_t time_ms_tenth =
      (time_us % 1000) / 100; /* Get the first decimal digit */

  updl_Info("%s costs: %u.%u ms\n", label, time_ms, time_ms_tenth);
}

// ============================================================================
// PORTABLE ALIGNED MEMORY ALLOCATION
// ============================================================================

void* aligned_malloc(size_t alignment, size_t size) {
#if defined(_WIN32)
    return _aligned_malloc(size, alignment);
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    // C11 standard aligned_alloc
    return aligned_alloc(alignment, size);
#else
    // Portable implementation using malloc + alignment
    void *ptr = malloc(size + alignment - 1 + sizeof(void*));
    if (!ptr) return NULL;
    
    void *aligned_ptr = (void*)(((uintptr_t)ptr + sizeof(void*) + alignment - 1) & ~(alignment - 1));
    ((void**)aligned_ptr)[-1] = ptr;  // Store original pointer for free
    return aligned_ptr;
#endif
}

void aligned_free(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    // C11 standard free
    free(ptr);
#else
    // Portable implementation - retrieve original pointer
    if (ptr) {
        void *original_ptr = ((void**)ptr)[-1];
        free(original_ptr);
    }
#endif
}