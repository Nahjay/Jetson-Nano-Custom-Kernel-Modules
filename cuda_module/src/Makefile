obj-m += gpu_kernel_module.o

KERNEL_DIR := /usr/src/linux-headers-4.9.337-tegra-ubuntu18.04_aarch64/kernel-4.9/
CUDA_DIR := /usr/local/cuda

all:
    make -C $(KERNEL_DIR) M=$(PWD) modules
    nvcc -c cuda_kernel.cu -o cuda_kernel.o
    $(CC) -o gpu_kernel_module.o -c gpu_kernel_module.c
    $(CC) -o gpu_kernel_module $(obj-m) cuda_kernel.o -L$(CUDA_DIR)/lib64 -lcudart

clean:
    make -C $(KERNEL_DIR) M=$(PWD) clean
    rm -f gpu_kernel_module