/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#ifndef COLORS_H
#define COLORS_H

#include <stdio.h>

/* ANSI Escape Code Colors for Terminal Output */
// Format: \033[<attribute>;<foreground>;<background>m

/* ANSI Color Codes */
#define ANSI_RESET "\033[0m"
#define ANSI_BOLD "\033[1m"
#define ANSI_DIM "\033[2m"
#define ANSI_BLUE "\033[34m"
#define ANSI_CYAN "\033[36m"
#define ANSI_GREEN "\033[32m"
#define ANSI_RED "\033[31m"
#define ANSI_MAGENTA "\033[35m"
#define ANSI_WHITE "\033[37m"
#define ANSI_GRAY "\033[38:5:250m"
#define ANSI_ORANGE "\033[38;2;217;120;88m"

/* Combined text and background */
#define ANSI_WHITE_ON_ORANGE "\033[37;48:5:208m"

/* Color definitions for log levels */
#define COLOR_DEBUG ANSI_BLUE
#define COLOR_INFO ANSI_WHITE
#define COLOR_WARNING ANSI_GRAY
#define COLOR_ERROR ANSI_RED
#define COLOR_HIGHLIGHT ANSI_ORANGE

#define _COLOR_PRINTF(color, format, ...) \
    printf(color format ANSI_RESET, ##__VA_ARGS__)
#define _COLOR_PRINTF_FMT(color, module, level, format, ...)     \
    printf(ANSI_RESET "[" module "][" color level ANSI_RESET \
                          "] " color format ANSI_RESET,          \
               ##__VA_ARGS__)
#define _COLOR_PRINTF_FMT2(color, level, format, ...)                              \
    printf(ANSI_RESET "[" color level ANSI_RESET "] " color format ANSI_RESET, \
               ##__VA_ARGS__)
#endif /* COLORS_H */