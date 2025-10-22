/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#ifndef UPDL_UTILITY_TANH_H
#define UPDL_UTILITY_TANH_H

#include <stdint.h>

/**
 * Look-up table for tanh approximation in Q15 fixed-point format
 * Contains pairs of values [dy, y] for evaluating tanh(x)
 */
extern const int16_t coeffs_tanh[];

#endif /* UPDL_UTILITY_TANH_H */