/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#ifndef UPDL_INTERPRETER_H
#define UPDL_INTERPRETER_H

// ============================================================================
// INCLUDES
// ============================================================================

#include <updl/updl_utility.h>
#include <up301/up301.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// CONSTANTS AND MACROS
// ============================================================================

#define TAG_FIELD 1
#define TAG_CHECK 1

#define DESCRIPTION_LENGTH 32
#define TAG_LENGTH 16
#define STRING_FIELD_LENGTH 16

#define Size_of(x) (x == Dtype_uint32_t) ? sizeof(uint32_t) : sizeof(uint8_t)
#define dtype(type) Dtype_##type

// ============================================================================
// FORWARD DECLARATIONS
// ============================================================================

typedef struct updl_executor_t updl_executor_t;

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

// Data types supported by the interpreter
typedef enum dtype_t {
  Dtype_uint8_t,
  Dtype_uint16_t,
  Dtype_uint32_t,
  Dtype_int8_t,
  Dtype_int16_t,
  Dtype_int32_t,
  Dtype_float32_t,
  Dtype_bool,
  Dtype_char,
  Dtype_dtype_t,
  Dtype_ltype_t,
  Dtype_ptype_t,
  Dtype_atype_t
} dtype_t;

// Layer types supported by the interpreter
typedef enum ltype_t {
  Ltype_conv_1d,
  Ltype_conv_2d,
  Ltype_depthwise_conv_2d,
  Ltype_max_pooling_2d,
  Ltype_average_pooling_2d,
  Ltype_dense,
  Ltype_flatten,
  Ltype_lambda,
  Ltype_add,
  Ltype_softmax,
} ltype_t;

// Padding types
typedef enum ptype_t { 
  Ptype_valid, 
  Ptype_same 
} ptype_t;

// Activation types
typedef enum atype_t {
  Atype_none,
  Atype_linear,
  Atype_relu,
  Atype_leakyrelu,
  Atype_softmax,
  Atype_sigmoid,
  Atype_tanh,
} atype_t;

// Weight data pointer (simplified for int16_t only)
typedef int16_t* weight_data_t;

// Weight structure with quantization information
typedef struct weights_t {
  dtype_t dtype;               // Always Dtype_int16_t for simplified design
  uint16_t weight_shape_d;     // Number of dimensions
  uint16_t weight_shape[4];    // Shape array
  weight_data_t weight;        // Direct int16_t pointer
  // Note: weight_shift removed - calculated at runtime from weight_scale
} weights_t;

// Memory pool structure defined in updl_utility.h

// Layer structure containing all layer parameters
typedef struct updl_layer_t {
  // Basic layer information
  int serial;
  char name[16];
  ltype_t type;

  // Shape information
  uint16_t input_shape[4];
  uint16_t output_shape[4];

  // Layer parameters
  uint16_t filters;
  uint16_t kernel_size[2];
  uint16_t strides[2];
  uint16_t units;
  uint16_t pool_size[2];
  uint16_t depth_multiplier;

  // Layer configuration
  ptype_t padding;
  atype_t activation;

  // Weight and bias data
  weights_t weights;
  weights_t bias;

  // Activation quantization parameters (layer outputs)
  float act_scale;              // Activation dequantization scale
  int16_t act_zp;               // Activation zero point (short for act_zero_point)
  
  // Quantization parameters
  float input_scale;            // Input tensor dequantization scale
  int16_t input_zp;             // Input tensor zero point

  float weight_scale;           // Weight dequantization scale
  int16_t weight_zp;            // Weight zero point (usually 0)

  float bias_scale;             // Bias dequantization scale
  int16_t bias_zp;              // Bias zero point (usually 0)

  float output_scale;           // Output tensor dequantization scale
  int16_t output_zp;            // Output tensor zero point

  // Precomputed quantization parameters (calculated during model loading)
  int32_t effective_multiplier; // Combined multiplier for (input * weight) -> output requantization
  int16_t effective_shift;      // Combined shift for requantization
  int32_t effective_bias_multiplier; // Combined multiplier for bias scaling: (input_scale * weight_scale) / bias_scale
  int16_t effective_bias_shift;      // Combined shift for bias scaling

} updl_layer_t;

// Streamlined model structure (static data only)
typedef struct updl_model_t {
  char description[32];          // Model description
  char model_name[16];           // Model name
  uint16_t num_layers;           // Number of layers
  uint16_t batch_input_shape[4]; // Input shape
  dtype_t dtype;                 // Data type
  float input_scale;             // Global input quantization scale

  updl_layer_t *layers;          // Array of layers (not pointers)
  updl_context_t *context;       // Memory context
} updl_model_t;


// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

// Context and memory management functions
updl_context_t *updl_init(uint8_t *memory, size_t memory_size);
void *updl_alloc(updl_context_t *ctx, size_t size);

// Memory pool management functions defined in updl_utility.h

// Model management functions
updl_model_t *updl_load_model(updl_context_t *ctx, uint8_t *model_data);

// Layer data parsing functions  
uint8_t updl_load_layer_params(updl_layer_t **layer, uint8_t **fp);
void updl_load_data(void *data, uint8_t **fp, dtype_t dtype, uint32_t count, 
                    const char *tag_name, uint8_t tag_field, uint8_t tag_check);

#endif // UPDL_INTERPRETER_H
