/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#include <updl/updl_profile.h>

void updl_profile_ticks(const char *label, uint32_t start_ticks,
                        uint32_t end_ticks, uint32_t tick_us,
                        updl_log_fn_t log_fn, const char *prefix) {
  if (!label || !log_fn) {
    return;
  }

  uint32_t ticks = end_ticks - start_ticks;
  uint32_t time_us = ticks * tick_us;
  uint32_t time_ms = time_us / 1000;
  uint32_t time_ms_tenth = (time_us % 1000) / 100;

  if (prefix) {
    log_fn("%s%s costs: %u.%u ms\n", prefix, label, time_ms, time_ms_tenth);
  } else {
    log_fn("%s costs: %u.%u ms\n", label, time_ms, time_ms_tenth);
  }
}
