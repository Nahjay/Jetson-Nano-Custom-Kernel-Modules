EXTRA_CFLAGS += -I/usr/src/linux-headers-4.9.337-tegra-ubuntu18.04_aarch64/kernel-4.9/include -I/usr/local/cuda-10.2/targets/aarch64-linux/include

obj-m += gpu_kernel_module.o

all:
	make -C /usr/src/linux-headers-4.9.337-tegra-ubuntu18.04_aarch64/kernel-4.9 M=$(PWD) modules
clean:
	make -C /usr/src/linux-headers-4.9.337-tegra-ubuntu18.04_aarch64/kernel-4.9 M=$(PWD) clean

