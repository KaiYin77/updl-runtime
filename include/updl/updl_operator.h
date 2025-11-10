/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#ifndef UPDL_OPERATOR_H
#define UPDL_OPERATOR_H

// ============================================================================
// INCLUDES
// ============================================================================

#include <updl/updl_interpreter.h>
#include <updl/updl_utility.h>

#include <stdbool.h>
#include <stdint.h>

// ============================================================================
// CONSTANTS AND MACROS
// ============================================================================

#define UPDL_PTR_OFFSET_16(base, byte_offset)                                  \
  ((int16_t *)((uint8_t *)(base) + (byte_offset)))

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

// Runner state enumeration
typedef enum rstate_t {
  rstate_invalid,      // runner data dirty/uninitialized
  rstate_unsupported,  // model can NOT be fully supported by hardware/software
  rstate_idle,         // runner initialized and ready to go
  rstate_running_soft, // busy executing
  rstate_running_hard, // busy executing
} rstate_t;

// Layer output callback (for testing/debugging)
// Called after each layer completes, before buffer swap
typedef void (*updl_layer_callback_t)(uint16_t layer_idx, const int16_t *output,
                                      size_t output_size, void *user_data);

// Execution layer context (streamlined)
typedef struct updl_exec_layer_t {
  uint16_t *input_ptr;  // Points to working buffer (input data)
  uint16_t *output_ptr; // Points to working buffer (output data)
  uint32_t input_size;
  uint32_t output_size;

  // Weight/bias pointers (into model data)
  const int16_t *weights;
  const int16_t *bias;
  uint32_t weight_size;
  uint32_t bias_size;
} updl_exec_layer_t;

// Forward declaration (removed - struct defined in updl_interpreter.h)

// Execution context (streamlined)
struct updl_executor_t {
  const updl_model_t *model;       // Reference to model (not copy)
  updl_exec_layer_t *exec_layers;  // Array of layer execution contexts
  updl_memory_pool_t *memory_pool; // Memory management
  rstate_t state;                  // Current execution state
  uint16_t current_layer;          // Current layer being processed

  // Debug/testing callback (optional)
  updl_layer_callback_t layer_callback; // Called after each layer completes
  void *callback_user_data;             // User data passed to callback
};

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Executor management functions (streamlined API)
updl_executor_t *updl_create_executor(const updl_model_t *model,
                                      updl_memory_pool_t *memory_pool);
void updl_free_executor(updl_executor_t *executor);

// Inference functions
int32_t updl_execute(updl_executor_t *executor, const void *input,
                     void *output);

// Callback registration (for testing/debugging)
void updl_set_layer_callback(updl_executor_t *executor,
                             updl_layer_callback_t callback, void *user_data);

#endif // UPDL_OPERATOR_H
