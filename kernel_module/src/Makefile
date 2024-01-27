EXTRA_CFLAGS += -I/usr/src/linux-headers-4.9.337-tegra-ubuntu18.04_aarch64/kernel-4.9/include

obj-m += kernel_module.o

all:
    make -C /usr/src/linux-headers-4.9.337-tegra-ubuntu18.04_aarch64/kernel-4.9 M=$(PWD) modules

clean:
    make -C /usr/src/linux-headers-4.9.337-tegra-ubuntu18.04_aarch64/kernel-4.9 M=$(PWD) clean
