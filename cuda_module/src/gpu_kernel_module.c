// Creating GPU kernel module that leverages CUDA module

#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/slab.h>
#include <linux/uaccess.h>




MODULE_LICENSE("GPL");
MODULE_AUTHOR("Nahjay Battieste");
MODULE_DESCRIPTION("Kernel module for changing the colors of a PPM image using GPU");

#define IMAGE_WIDTH 1280
#define IMAGE_HEIGHT 853
#define IMAGE_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT * 3) // Assuming RGB format

