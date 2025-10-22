/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#ifndef UPDL_DEBUG_H
#define UPDL_DEBUG_H

// ============================================================================
// INCLUDES
// ============================================================================

#include <updl/updl_interpreter.h>

// ============================================================================
// DEBUG UTILITY FUNCTIONS
// ============================================================================

// Layer type to string conversion for debugging and logging
const char *updl_layer_type_to_string(ltype_t layer_type);

// Array printing functions for debugging
void updl_print_1d_array(const char *name, const int16_t *data, size_t size, 
                          float scale, int16_t zero_point, size_t elements_per_line);

void updl_print_2d_array(const char *name, const int16_t *data, 
                         size_t height, size_t width, size_t channels,
                         float scale, int16_t zero_point, size_t max_channels);

// Layer metadata printing function for debugging
void updl_print_layer_metadata(size_t layer_index, const updl_layer_t *layer);

// Quantization debug logging function
void updl_print_eff_quant_param(size_t layer_index, float eff_scale, float input_scale, 
                                   float weight_scale, float output_scale, int32_t eff_mult, 
                                   int16_t eff_shift, float eff_bias_scale, float bias_scale, 
                                   int32_t eff_bias_mult, int16_t eff_bias_shift);

#endif // UPDL_DEBUG_H