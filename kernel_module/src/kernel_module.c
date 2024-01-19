// /*  
//  *  hello-1.c - The simplest kernel module.
//  */
// #include <linux/module.h>	/* Needed by all modules */
// #include <linux/kernel.h>	/* Needed for KERN_INFO */


// MODULE_LICENSE("GPL");
// MODULE_AUTHOR("Nahjay");
// MODULE_DESCRIPTION("Hello world driver!");

// int init_module(void)
// {
// 	printk(KERN_INFO "Hello world 1.\n");

// 	/* 
// 	 * A non 0 return means init_module failed; module can't be loaded. 
// 	 */
// 	return 0;
// }

// void cleanup_module(void)
// {
// 	printk(KERN_INFO "Goodbye world 1.\n");
// }
#include <linux/miscdevice.h>
#include <linux/fs.h>
// #include <linux/miscdevice.h>
// #include <linux/uaccess.h>
#include <linux/slab.h>
// #include <slab.h>
#include <linux/module.h>


#define MAX_PATH 256

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("Kernel module for flipping images horizontally");

#define IMAGE_WIDTH 640
#define IMAGE_HEIGHT 480
#define IMAGE_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT * 3) // Assuming RGB format

static void flip_image_horizontally(unsigned char *data, size_t width, size_t height) {
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width / 2; ++x) {
            size_t left_index = y * width * 3 + x * 3;
            size_t right_index = y * width * 3 + (width - x - 1) * 3;

            // Swap RGB values
            for (size_t channel = 0; channel < 3; ++channel) {
                unsigned char temp = data[left_index + channel];
                data[left_index + channel] = data[right_index + channel];
                data[right_index + channel] = temp;
            }
        }
    }
}

static int process_image(const char __user *user_image_path) {
    struct file *file;
    loff_t pos = 0;
    int ret = 0;

    // Copy user space image path to kernel space
    char image_path[MAX_PATH];
    if (copy_from_user(image_path, user_image_path, MAX_PATH)) {
        printk(KERN_ERR "Failed to copy image path from user space\n");
        return -EFAULT;
    }

    // Open the image file
    file = filp_open(image_path, O_RDWR, 0);
    if (IS_ERR(file)) {
        printk(KERN_ERR "Failed to open image file\n");
        return PTR_ERR(file);
    }

    // Allocate memory for the image data
    unsigned char *image_data = kmalloc(IMAGE_SIZE, GFP_KERNEL);
    if (!image_data) {
        printk(KERN_ERR "Failed to allocate memory for image data\n");
        ret = -ENOMEM;
        goto cleanup;
    }

    // Read the image data
    ret = kernel_read(file, image_data, IMAGE_SIZE, &pos);
    if (ret < 0) {
        printk(KERN_ERR "Failed to read image data from file\n");
        goto cleanup;
    }

    // Process the image data (flip horizontally)
    flip_image_horizontally(image_data, IMAGE_WIDTH, IMAGE_HEIGHT);

	// Debug print: Print the first few bytes of the modified image data
    printk(KERN_INFO "Modified Image Data (first 16 bytes): %*ph\n", 16, image_data);


    // Save the modified image data back to the file
    pos = 0;
    ret = kernel_write(file, image_data, IMAGE_SIZE, &pos);
    if (ret < 0) {
        printk(KERN_ERR "Failed to write modified image data to file\n");
        goto cleanup;
    }

    printk(KERN_INFO "Image flipped successfully\n");

cleanup:
    // Cleanup
    kfree(image_data);
    filp_close(file, NULL);

    return ret;
}

static long ioctl_handler(struct file *file, unsigned int cmd, unsigned long arg) {
    const char __user *user_image_path = (const char __user *)arg;
    return process_image(user_image_path);
}

static const struct file_operations fops = {
    .unlocked_ioctl = ioctl_handler,
};

static struct miscdevice flip_image_misc_device = {
    .minor = MISC_DYNAMIC_MINOR,
    .name = "flip_image_kernel_module",
    .fops = &fops,
};

static int __init flip_image_module_init(void) {
    int ret = misc_register(&flip_image_misc_device);
    if (ret) {
        pr_err("Failed to register misc device\n");
        return ret;
    }
    pr_info("Flip image kernel module loaded\n");
    return 0;
}

static void __exit flip_image_module_exit(void) {
    misc_deregister(&flip_image_misc_device);
    pr_info("Flip image kernel module unloaded\n");
}
module_init(flip_image_module_init);
module_exit(flip_image_module_exit);