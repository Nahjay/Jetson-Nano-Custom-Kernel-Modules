KERNEL_DIR := /usr/src/linux-headers-4.9.337-tegra-ubuntu18.04_aarch64/kernel-4.9/
CUDA_DIR := /usr/local/cuda

obj-m += gpu_kernel_module.o

all:
	make -C $(KERNEL_DIR) M=$(PWD) modules

clean:
	make -C $(KERNEL_DIR) M=$(PWD) clean