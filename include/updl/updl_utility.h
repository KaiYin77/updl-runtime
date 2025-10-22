/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#ifndef UPDL_UTILITY_H
#define UPDL_UTILITY_H

// ============================================================================
// INCLUDES
// ============================================================================

#include <utils/colors.h>

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// ============================================================================
// LOGGING CONFIGURATION
// ============================================================================

// Log level enables (0=Off, 1=On)
#define UPDL_ENABLE_DEBUG 0      // Detailed debug info (development only)
#define UPDL_ENABLE_INFO 1       // General information messages
#define UPDL_ENABLE_WARNING 1    // Important warnings (should be enabled)
#define UPDL_ENABLE_ERROR 1      // Critical errors (must be enabled)

// ============================================================================
// LOGGING MACROS
// ============================================================================

// Debug logging macros
#if UPDL_ENABLE_DEBUG
#define updl_Debug(...) _COLOR_PRINTF_FMT(COLOR_DEBUG, "UPDL ", "DEBUG  ", __VA_ARGS__)
#define updl_Debug2(...) _COLOR_PRINTF(COLOR_DEBUG, __VA_ARGS__)
#else
#define updl_Debug(...)
#define updl_Debug2(...)
#endif

// Info logging macros
#if UPDL_ENABLE_INFO
#define updl_Info(...) _COLOR_PRINTF_FMT(COLOR_INFO, "UPDL ", "INFO   ", __VA_ARGS__)
#define updl_Info2(...) _COLOR_PRINTF(COLOR_INFO, __VA_ARGS__)
#else
#define updl_Info(...)
#define updl_Info2(...)
#endif

// Warning logging macros
#if UPDL_ENABLE_WARNING
#define updl_Warning(...) _COLOR_PRINTF_FMT(COLOR_WARNING, "UPDL ", "WARNING", __VA_ARGS__)
#else
#define updl_Warning(...)
#endif

// Error logging macros
#if UPDL_ENABLE_ERROR
#define updl_Error(...) _COLOR_PRINTF_FMT(COLOR_ERROR, "UPDL ", "ERROR  ", __VA_ARGS__)
#else
#define updl_Error(...)
#endif

// ============================================================================
// FORWARD DECLARATIONS
// ============================================================================

typedef struct updl_context_t updl_context_t;
struct updl_model_t; // Forward declaration

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

// Memory context structure for embedded-safe memory management
typedef struct updl_context_t {
    uint8_t *memory_base;        // Base memory address
    size_t memory_size;          // Total memory available
    size_t memory_offset;        // Current allocation offset
} updl_context_t;

// Memory pool for stream processing inference
typedef struct updl_memory_pool_t {
    uint8_t *base_address;       // Base address of the pool
    size_t total_size;           // Total pool size
    size_t used_size;            // Currently used size
    
    // Stream processing buffers
    uint8_t *input_buffer;       // Current layer input buffer
    uint8_t *output_buffer;      // Current layer output buffer
    size_t current_input_size;   // Current input buffer size in bytes
    size_t current_output_size;  // Current output buffer size in bytes
    size_t max_buffer_size;      // Maximum buffer size allocated
    
    // Buffer management
    bool buffers_swapped;        // Track if input/output are swapped
} updl_memory_pool_t;

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Memory management functions
updl_context_t* updl_init(uint8_t *memory, size_t memory_size);
void* updl_alloc(updl_context_t *ctx, size_t size);

// Memory pool management functions
updl_memory_pool_t* updl_create_memory_pool(uint8_t *memory, size_t total_size, size_t max_layer_size, const struct updl_model_t *model);
void updl_reset_memory_pool(updl_memory_pool_t *pool);

// Stream buffer management functions
int updl_ensure_buffer_capacity(updl_memory_pool_t *pool, size_t input_size, size_t output_size);
void updl_swap_stream_buffers(updl_memory_pool_t *pool);
uint8_t* updl_get_input_buffer(updl_memory_pool_t *pool);
uint8_t* updl_get_output_buffer(updl_memory_pool_t *pool);


// Forward declaration
typedef struct updl_executor_t updl_executor_t;

// Executor recovery functions
int updl_reset_executor(updl_executor_t *executor);

// Profiling functions
uint32_t updl_get_current_ticks(void);
void updl_profile(const char *label, uint32_t start_time);

// Portable aligned memory allocation functions
void* aligned_malloc(size_t alignment, size_t size);
void aligned_free(void* ptr);

#endif // UPDL_UTILITY_H