// Creating GPU kernel module that leverages CUDA module

#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/cuda.h>



MODULE_LICENSE("GPL");
MODULE_AUTHOR("Nahjay Battieste");
MODULE_DESCRIPTION("Kernel module for changing the colors of a PPM image using GPU");

#define IMAGE_WIDTH 1280
#define IMAGE_HEIGHT 853
#define MAX_PATH 256
#define IMAGE_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT * 3) // Assuming RGB format

// CUDA kernel
extern void cuda_kernel(unsigned char *data, size_t width, size_t height, size_t metadata_size);

// Create ioctl_handler
static long ioctl_handler(struct file *file, unsigned int cmd, unsigned long arg) {


    // Copy image from user space to kernel space
    char image_path[MAX_PATH];

    if (copy_from_user(image_path, (char *)arg, MAX_PATH)) {
        printk(KERN_ERR "Failed to copy image path from user space\n");
        return -EFAULT;
    }

    // Open the image file
    struct file *image_file = filp_open(image_path, O_RDWR, 0);
    if (IS_ERR(image_file)) {
        printk(KERN_ERR "Failed to open image file\n");
        return PTR_ERR(image_file);
    }

    // Allocate GPU memory for the image data
    unsigned char *gpu_image_data;
    if (cudaMalloc((void **)&gpu_image_data, IMAGE_SIZE)) {
        printk(KERN_ERR "Failed to allocate GPU memory for image data\n");
        return -ENOMEM;
    }

    // Read the image file into the GPU memory
    ssize_t read_size = kernel_read(image_file, 0, gpu_image_data, IMAGE_SIZE);
    if (read_size < 0) {
        cudaFree(gpu_image_data);
        printk(KERN_ERR "Failed to read image file\n");
        filp_close(image_file, NULL);
        return read_size;
    }

    // Open Cuda kernel
    cuda_kernel(gpu_image_data, IMAGE_WIDTH, IMAGE_HEIGHT, METADATA);

    // Synchonize the GPU
    if (cudaDeviceSynchronize()) {
        cudaFree(gpu_image_data);
        printk(KERN_ERR "Failed to synchronize GPU\n");
        filp_close(image_file, NULL);
        return -EFAULT;
    }

    // Check for CUDA specific errors
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        cudaFree(gpu_image_data);
        printk(KERN_ERR "CUDA error: %s\n", cudaGetErrorString(cuda_error));
        filp_close(image_file, NULL);
        return -EFAULT;
    }

    // Write the modified image data back to the image file
    ssize_t write_size = kernel_write(image_file, 0, gpu_image_data, IMAGE_SIZE);
    if (write_size < 0) {
        cudaFree(gpu_image_data);
        printk(KERN_ERR "Failed to write image file\n");
        filp_close(image_file, NULL);
        return write_size;
    }

    // Cleanup
    cudaFree(gpu_image_data);
    filp_close(image_file, NULL);

    return 0;
}

static const struct file_operations fops = {
    .owner = THIS_MODULE,
    .unlocked_ioctl = ioctl_handler
};


static struct miscdevice misc_device = {
    .minor = MISC_DYNAMIC_MINOR,
    .name = "gpu_kernel_module",
    .fops = &fops
};


static int __init gpu_kernel_module_init(void) {
    int ret = misc_register(&misc_device);
    if (ret) {
        printk(KERN_ERR "Failed to register misc device\n");
        return ret;
    }

    printk(KERN_INFO "GPU kernel module loaded\n");

    return 0;
}

static void __exit gpu_kernel_module_exit(void) {
    misc_deregister(&misc_device);
    printk(KERN_INFO "GPU kernel module unloaded\n");
}

module_init(gpu_kernel_module_init);
module_exit(gpu_kernel_module_exit);
