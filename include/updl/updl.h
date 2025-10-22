/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#ifndef UPDL_H
#define UPDL_H

// ============================================================================
// UPDL - Ultra-Portable Deep Learning Library
// Three Layer Architecture:
// - updl_utility:     Memory management, logging, utilities
// - updl_interpreter: Model parsing, serialization, loading
// - updl_operator:    Neural network operations, execution, hardware
// acceleration
// ============================================================================

#include <updl/updl_interpreter.h> // Interpreter layer: model parsing, serialization
#include <updl/updl_operator.h> // Operator layer: neural network operations
#include <updl/updl_utility.h>  // Utility layer: logging, memory, helpers

#endif /* UPDL_H */
