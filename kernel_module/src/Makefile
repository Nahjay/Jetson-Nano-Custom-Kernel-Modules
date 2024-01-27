EXTRA_CFLAGS += -I/usr/src/linux-headers-4.9.337-tegra-ubuntu18.04_aarch64

obj-m += kernel_module.o

all:
	make -C /usr/src/linux-headers-4.9.337-tegra-ubuntu18.04_aarch64 M=$(PWD) modules

clean:
	make -C /usr/src/linux-headers-4.9.337-tegra-ubuntu18.04_aarch64 M=$(PWD) clean


