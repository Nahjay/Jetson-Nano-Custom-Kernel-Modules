EXTRA_CFLAGS += -I/usr/src/linux-headers-4.15.0-20-generic/include

obj-m += kernel_module.o

all:
	make -C /usr/src/linux-headers-4.15.0-20-generic M=$(PWD) modules

clean:
	make -C /usr/src/linux-headers-4.15.0-20-generic M=$(PWD) clean
