obj-m += gpu_kernel_module.o

KERNEL_DIR := /usr/src/linux-headers-4.9.337-tegra-ubuntu18.04_aarch64/kernel-4.9/

CUDA_DIR := /usr/local/cuda

all:
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) modules
	nvcc -c cuda_kernel.cu -o cuda_kernel.o -I$(CUDA_DIR)/include

clean:
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) clean
	rm -f cuda_kernel.o
