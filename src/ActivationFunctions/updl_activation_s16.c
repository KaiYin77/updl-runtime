/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#include "updl/updl_kernels_support.h"
#include "updl/updl_utility_tanh.h"
#include "updl/updl_operator.h"

/**
 * @brief Calculate tanh using fixed-point arithmetic
 * 
 * Extracted from original updl_kernels.c and optimized
 */
static int16_t updl_calculate_tanh_fix(int32_t x) {
    int32_t d = (int32_t)1 << 10; // 2^-5
    int32_t s = d >> 1;           // 2^-6
    int32_t M = (int32_t)5 << 15;
    int32_t kernel_width;
    int32_t dx;
    int32_t xi;
    int8_t is_neg;
    int16_t result;

    if (x == 0)
        return 0;

    if (x < 0) {
        is_neg = 1;
        xi = -x;
    } else {
        is_neg = 0;
        xi = x;
    }

    if (xi >= M)
        result = 0x7fff;
    else {
        kernel_width = (xi - s) >> 10;
        if (kernel_width < 0) kernel_width = 0;
        dx = xi - s - (kernel_width << 10);

        // Interpolation similar to Python implementation
        int32_t y0 = coeffs_tanh[(kernel_width << 1)];
        int32_t dy = coeffs_tanh[(kernel_width << 1) + 1];

        // Interpolation calculation
        int32_t interp = (dx * dy) >> 15;
        result = (int16_t)(y0 + interp);
        if (result < 0) result = 0;
    }

    if (is_neg)
        result = -result;

    return result;
}

/**
 * @brief Calculate sigmoid using tanh: σ(x) = (tanh(x/2) + 1) / 2
 */
static inline int16_t updl_calculate_sigmoid_fix(int32_t x) {
    int32_t h = (int32_t)1 << 14;
    int16_t result;
    int32_t xi = x >> 1;

    result = updl_calculate_tanh_fix(xi);
    result >>= 1;
    result += h;

    return result;
}

// Individual activation functions
static inline int32_t updl_activation_none(int32_t input) {
    return input;
}

static inline int32_t updl_activation_relu(int32_t input) {
    return input > 0 ? input : 0;
}

static inline int32_t updl_activation_leakyrelu(int32_t input) {
    return input > 0 ? input : input / 100;
}

static inline int32_t updl_activation_sigmoid(int32_t input) {
    return (int32_t)updl_calculate_sigmoid_fix(input);
}

static inline int32_t updl_activation_tanh(int32_t input) {
    return (int32_t)updl_calculate_tanh_fix(input);
}

static inline int32_t updl_activation_softmax(int32_t input) {
    // Softmax is typically computed at layer level, not element level
    return input;
}

// Function pointer type for activation functions
typedef int32_t (*activation_fn_t)(int32_t input);

// Activation function lookup table
static const activation_fn_t activation_functions[] = {
    [Atype_none] = updl_activation_none,
    [Atype_linear] = updl_activation_none,
    [Atype_relu] = updl_activation_relu,
    [Atype_leakyrelu] = updl_activation_leakyrelu,
    [Atype_softmax] = updl_activation_softmax,
    [Atype_sigmoid] = updl_activation_sigmoid,
    [Atype_tanh] = updl_activation_tanh,
};

#define ACTIVATION_FUNCTION_COUNT (sizeof(activation_functions) / sizeof(activation_functions[0]))

/**
 * @brief Apply activation function to input value
 * 
 * Main activation function dispatcher
 */
int32_t updl_activation(int32_t input, atype_t activation) {
    if (activation < 0 || activation >= ACTIVATION_FUNCTION_COUNT) {
        // Invalid activation type, return input unchanged
        return input;
    }
    
    return activation_functions[activation](input);
}
