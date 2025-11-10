# UPDL Library Compilation Makefile
# For sanity checking before RISC-V deployment

CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -pedantic -g -O2 -Wno-unused-parameter -Wno-format -Wno-unused-variable
INCLUDES = -Iinclude
SRCDIR = src
OBJDIR = obj

# Source files
SOURCES = $(wildcard $(SRCDIR)/*.c) \
          $(wildcard $(SRCDIR)/ActivationFunctions/*.c) \
          $(wildcard $(SRCDIR)/ConvolutionFunctions/*.c) \
          $(wildcard $(SRCDIR)/FullyConnectedFunctions/*.c) \
          $(wildcard $(SRCDIR)/NNSupportFunctions/*.c) \
          $(wildcard $(SRCDIR)/PoolingFunctions/*.c)

# Object files
OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

# Library name
LIBNAME = libupdl.a
SONAME = libupdl.so

# Default target
all: $(LIBNAME) $(SONAME)

# Create object directory structure
$(OBJDIR):
	mkdir -p $(OBJDIR)
	mkdir -p $(OBJDIR)/ActivationFunctions
	mkdir -p $(OBJDIR)/ConvolutionFunctions
	mkdir -p $(OBJDIR)/FullyConnectedFunctions
	mkdir -p $(OBJDIR)/NNSupportFunctions
	mkdir -p $(OBJDIR)/PoolingFunctions

# Static library
$(LIBNAME): $(OBJDIR) $(OBJECTS)
	ar rcs $@ $(OBJECTS)
	@echo "Static library created: $(LIBNAME)"

# Shared library
$(SONAME): $(OBJDIR) $(OBJECTS)
	$(CC) -shared -o $@ $(OBJECTS)
	@echo "Shared library created: $(SONAME)"

# Compile object files
$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) $(INCLUDES) -fPIC -c $< -o $@

# Sanity check compilation (compile without linking)
check:
	@echo "Performing sanity check compilation..."
	@for src in $(SOURCES); do \
		echo "Checking: $$src"; \
		$(CC) $(CFLAGS) $(INCLUDES) -fsyntax-only -c $$src || exit 1; \
	done
	@echo "All source files compile successfully!"

# Clean build artifacts
clean:
	rm -rf $(OBJDIR) $(LIBNAME) $(SONAME)

# Show compilation info
info:
	@echo "Compiler: $(CC)"
	@echo "Flags: $(CFLAGS)"
	@echo "Includes: $(INCLUDES)"
	@echo "Sources found: $(words $(SOURCES)) files"
	@echo "Source files:"
	@for src in $(SOURCES); do echo "  $$src"; done

# Test individual file compilation
test-file:
	@read -p "Enter C file to test (e.g., updl_interpreter.c): " file; \
	if [ -f "$$file" ]; then \
		echo "Testing compilation of $$file..."; \
		$(CC) $(CFLAGS) $(INCLUDES) -fsyntax-only -c "$$file" && echo "✓ $$file compiles successfully"; \
	else \
		echo "File $$file not found"; \
	fi

.PHONY: all clean check info test-file