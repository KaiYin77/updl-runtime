/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#ifndef UPDL_PROFILE_H
#define UPDL_PROFILE_H

#include <stdint.h>

typedef void (*updl_log_fn_t)(const char *fmt, ...);

void updl_profile_ticks(const char *label, uint32_t start_ticks,
                        uint32_t end_ticks, uint32_t tick_us,
                        updl_log_fn_t log_fn, const char *prefix);

#endif // UPDL_PROFILE_H
