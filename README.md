# UPDL Runtime

This repository provides the UPDL Runtime modules for the UP201/UP301 SoC, developed and maintained by Upbeat Technology.

## Overview

UPDL Runtime is a C library that provides neural network inference capabilities optimized for Upbeat Technology's UP201/UP301 System-on-Chip (SoC) platforms. The library includes optimized implementations of common neural network operations including convolution, activation functions, pooling, and fully connected layers.

## Features

- Optimized neural network kernels for UP201/UP301 hardware
- Support for quantized model inference
- Comprehensive test framework for validation
- Static and shared library builds
- Hardware-specific optimization utilities

## Library Components

### Core Modules (Hierarchical Architecture)
- **UPDL Interpreter**: Top-level inference engine and model execution coordinator
- **UPDL Operator**: Neural network operation implementations and layer management
- **UPDL Kernels**: Hardware-optimized computation kernels and mathematical operations
- **Kernel Implementation**: Low-level function implementations for neural network operations
- **UPDL Utility**: Support functions and utilities for memory management and debugging

### Neural Network Functions
- **Activation Functions**: ReLU, Tanh, Sigmoid, and other activation layers
- **Convolution Functions**: Optimized 2D convolution and depthwise convolution implementations
- **Fully Connected Functions**: Dense layer operations
- **Pooling Functions**: Max and average pooling operations
- **Support Functions**: Memory management and tensor utilities

### Testing Framework
- **Implementation Test Runner**: Validates kernel implementations
- **Propagation Test Runner**: Tests end-to-end model inference
- **Quantization Test Runner**: Validates quantized model accuracy

## Integration as Git Submodule

This repository is designed to be used as a Git submodule within the Upbeat Technology SDK.

### Adding as Submodule

```bash
git submodule add https://github.com/KaiYin77/updl-runtime.git
git submodule update --init --recursive
```

### Usage

The parent SDK handles all build and compilation processes. Please reference the Upbeat Tech official demo code for using UPDL Runtime.

## Hardware Support

This library is specifically optimized for UP201/301 series SoC with AI accelerated hardware.

## License

Copyright (c) Upbeat Technology. All rights reserved.

## Support

For technical support and documentation, visit [www.upbeattechtw.com](https://www.upbeattechtw.com/)