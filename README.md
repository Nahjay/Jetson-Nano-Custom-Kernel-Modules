# Jetson-Nano-Custom-Kernel-Modules

# Project README

This project demonstrates the use of both a Linux kernel module and a CUDA kernel for image processing on an NVIDIA Jetson Nano platform. 

## Overview

The project consists of two main components:

1. **Linux Kernel Module**: A kernel module is a piece of code that can be dynamically loaded and unloaded into the Linux kernel. In this project, the Linux kernel module is responsible for processing image data on the CPU. It can be loaded and unloaded into the Linux kernel as needed. 

2. **CUDA Kernel**: CUDA (Compute Unified Device Architecture) is a parallel computing platform and application programming interface (API) model created by NVIDIA. It allows developers to use a CUDA-enabled graphics processing unit (GPU) for general-purpose processing. The CUDA kernel in this project is responsible for processing image data on the GPU.

## Hardware

The project was developed and tested on the following hardware:

- NVIDIA Jetson Nano Developer Kit

## Software

The project was developed and tested on the following software:

- Ubuntu 18.04 LTS
- CUDA Toolkit 10.2
- Linux Kernel 4.9.337-tegra

## Linux Kernel Module

The Linux kernel module is written in C and is responsible for loading image data from a file, processing it using CPU-based algorithms, and optionally writing the processed image data back to a file. The module interacts with the Linux kernel and system calls to perform its tasks.

### Usage

To build and load the Linux kernel module, follow these steps:

1. Navigate to the `kernel_module` directory.
2. Run `make` to build the module.
3. Load the module using `sudo insmod <module_name>.ko`.
4. Optionally, unload the module using `sudo rmmod <module_name>`.

## CUDA Kernel

The CUDA kernel is written in CUDA C/C++ and is responsible for offloading image processing tasks to the GPU. It utilizes the parallel processing capabilities of the GPU to perform computations on image data more efficiently compared to traditional CPU-based approaches.

## Image Format

The image format used in this project is the Portable Pixmap (PPM) format. It is a lowest common denominator color image file format that can be used to save images in both binary and ASCII formats. The PPM format is a convenient format for image processing because it is simple, portable, and widely supported.

The images supplied in the images directory are in the PPM format. They can be opened and viewed using any image viewer that supports the PPM format. They are the target images for the image processing algorithms in this project. The target resolution for the images is 1280x853.

## Image Processing Algorithms

The image processing algorithms used in this project are:

### Linux Kernel Module

``` 
static void modify_ppm_colors(unsigned char *data, size_t width, size_t height, size_t metadata_size) {
    size_t y, x; 
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            size_t index = y * width * 3 + x * 3;

            // Skip modifying metadata
            if (index < metadata_size) {
                continue;
            }

            size_t channel;

            // Increment RGB values by a constant value
            for (channel = 0; channel < 3; ++channel) {
                data[index + channel] = (data[index + channel] + 100) % 256;
            }
        }
    }
} 
```
This algorithm modifies the RGB values of each pixel in the image by a constant value of 100. It is implemented in the `modify_ppm_colors` function in `kernel_module.c`.


### CUDA Kernel

```
__global__ void modify_ppm_colors(unsigned char *data, size_t width, size_t height, size_t metadata_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < width * height) {
        int y = idx / width;
        int x = idx % width;
        size_t index = y * width * 3 + x * 3;

        // Skip modifying metadata
        if (index >= metadata_size) {
            // Increment RGB values by 100
            for (int channel = 0; channel < 3; ++channel) {
                data[index + channel] = (data[index + channel] + 100) % 256;
            }
        }

        idx += blockDim.x * gridDim.x;
    }
}

...

// Launch CUDA kernel
    unsigned char *d_data;
    cudaMalloc(&d_data, IMAGE_SIZE * sizeof(unsigned char));
    cudaMemcpy(d_data, data, IMAGE_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice);
    modify_ppm_colors<<<(IMAGE_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_data, IMAGE_WIDTH, IMAGE_HEIGHT, METADATA);
    cudaMemcpy(data, d_data, IMAGE_SIZE * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
```

This algorithm modifies the RGB values of each pixel in the image by a constant value of 100. It is implemented in the `modify_ppm_colors` function in `cuda_kernel.cu`. It is launched as a CUDA kernel using the `modify_ppm_colors<<<(IMAGE_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_data, IMAGE_WIDTH, IMAGE_HEIGHT, METADATA);` statement in `user_app.cu`.

This algorithm is functionally equivalent to the algorithm used in the Linux kernel module. However, it is implemented using parallel processing techniques to take advantage of the GPU's parallel processing capabilities.

## Usage

To compile and run the CUDA kernel, follow these steps:

1. Ensure that you have the CUDA Toolkit installed on your system.
2. Navigate to the `user_app` directory.
3. Run `nvcc -c user_app user_app.c` to compile the CUDA kernel.

## Example Usage

Once both the Linux kernel module and CUDA kernel are compiled and loaded, you can use them together through the compiled binary `user_app`. This binary is responsible for loading image data from a file, determined by a file path passed in as an argument, and passing it to the Linux kernel module for processing. The processed image data is the written back to the file. It can also be used to run the CUDA kernel on the GPU, when you pass in a second image path. The second image is processed by the CUDA kernel and the processed image data is written back to the file.

```bash
# Load the Linux kernel module
sudo insmod kernel_module.ko

# Compile the user app that includes the CUDA kernel and the Linux kernel module interface
nvcc -c user_app user_app.c

# Run the user application to process image data using the Linux kernel module
sudo ./user_app input_image.ppm input_image_2.ppm
```

## License

This project is licensed under the MIT License. See the LICENSE file for more information.


