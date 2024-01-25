# Location of the kernel source tree
KERNEL_SOURCE := /usr/src/linux-headers-$(shell uname -r)

# Path to CUDA toolkit
CUDA_PATH := /usr/local/cuda

# Compiler
CC := $(CROSS_COMPILE)gcc
HOSTCC := gcc

# Flags
EXTRA_CFLAGS := -I$(PWD)/include -I$(KERNEL_SOURCE)/include -I$(CUDA_PATH)/include
EXTRA_LDFLAGS := -L$(CUDA_PATH)/lib64

# Module name
MODULE_NAME := gpu_kernel_module

# Source files
MODULE_SRC := gpu_kernel_module.c

# Objects
obj-m := $(MODULE_NAME).o

# Targets
all: module

module:
    $(MAKE) -C $(KERNEL_SOURCE) M=$(PWD) modules

clean:
    $(MAKE) -C $(KERNEL_SOURCE) M=$(PWD) clean