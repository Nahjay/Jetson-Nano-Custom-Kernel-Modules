EXTRA_CFLAGS += -I/usr/src/linux-headers-4.9.337-tegra-ubuntu18.04_aarch64/kernel-4.9/include \
               -I/usr/local/cuda-10.2/targets/aarch64-linux/include \
               -I/usr/include/linux

obj-m += gpu_kernel_module.o

KERNEL_DIR := /usr/src/linux-headers-4.9.337-tegra-ubuntu18.04_aarch64/kernel-4.9/
CUDA_DIR := /usr/local/cuda-10.2/targets/aarch64-linux/

all:
	/usr/local/cuda-10.2/bin/nvcc -c -o cuda_kernel.o cuda_kernel.cu
	make -C $(KERNEL_DIR) M=$(PWD) modules
	$(CC) -o gpu_kernel_module gpu_kernel_module.c cuda_kernel.o -I$(CUDA_DIR)/include -L$(CUDA_DIR)/lib64 -lcudart

clean:
	make -C $(KERNEL_DIR) M=$(PWD) clean
	rm -f gpu_kernel_module cuda_kernel.o