/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#include "updl/updl_kernels_support.h"

/**
 * @brief Calculate padding for SAME padding mode
 * 
 * Following TensorFlow/CMSIS-NN padding calculation
 */
void updl_calculate_padding(const uint32_t input_size,
                               const uint32_t kernel_size,
                               const uint32_t stride,
                               uint32_t *pad_before,
                               uint32_t *pad_after) {
    
    uint32_t pad_along_dim = 0;
    
    if (input_size % stride == 0) {
        pad_along_dim = (kernel_size > stride) ? (kernel_size - stride) : 0;
    } else {
        pad_along_dim = (kernel_size > (input_size % stride)) 
                        ? (kernel_size - (input_size % stride)) : 0;
    }
    
    if (pad_before) {
        *pad_before = pad_along_dim / 2;
    }
    
    if (pad_after) {
        *pad_after = pad_along_dim - (pad_along_dim / 2);
    }
}