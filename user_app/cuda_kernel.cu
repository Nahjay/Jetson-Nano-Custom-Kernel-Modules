#include "/usr/local/cuda-10.2/include/cuda_runtime.h"
#include "cuda_kernel.h"
#include "/usr/local/cuda-10.2/include/cuda.h"
#include "/usr/local/cuda-10.2/include/device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>


#define IMAGE_WIDTH 1280
#define IMAGE_HEIGHT 853
#define METADATA 100
// Updated CUDA kernel function to process image data
__global__ void cuda_kernel(char *image_data, size_t width, size_t height, size_t metadata_size) {
    // Calculate global index
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Skip modifying metadata
    if (index < metadata_size) {
        return;
    }

    // Calculate pixel index
    int pixel_index = (index - metadata_size) / 3;

    // Ensure index is within image data bounds
    if (pixel_index < width * height) {
        // Increment RGB values by 10
        image_data[index] = (image_data[index] + 10) % 256;
    }
}

// Function to process image data using Cuda
void process_image_data(char *image_data) {


    // Get the size of the image data
    size_t image_data_size = sizeof(*image_data);

    // Create pointers to the image data
    char *d_image_data;

    // Allocate memory for the image data on the device
    cudaMalloc((void**) &d_image_data, image_data_size);

    // Check if memory was allocated successfully
    if (d_image_data == NULL) {
        fprintf(stderr, "Failed to allocate memory for image data on device\n");
        exit(EXIT_FAILURE);

        // cleanup
        cudaFree(d_image_data);
        d_image_data = NULL;
    }
    else {
        printf("Successfully allocated memory for image data on device\n");
    }

    // Copy image data from host to device
    cudaMemcpy(d_image_data, image_data, image_data_size, cudaMemcpyHostToDevice);

    // Check if image data was copied successfully
    if (cudaMemcpy(d_image_data, image_data, image_data_size, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy image data from host to device\n");
        exit(EXIT_FAILURE);

        // cleanup
        cudaFree(d_image_data);
        d_image_data = NULL;
    }
    else {
        printf("Successfully copied image data from host to device\n");
    }

    // Define grid and block dimensions
    int blockSize = 256;
    int gridSize = (image_data_size + blockSize - 1) / blockSize;

    // Launch the CUDA kernel
    cuda_kernel<<<gridSize, blockSize>>>(d_image_data, IMAGE_WIDTH, IMAGE_HEIGHT, METADATA);
    // Check for kernel launch errors
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(cuda_error));
        exit(EXIT_FAILURE);
    }

    // Copy the modified image data back to host memory
    cudaMemcpy(image_data, d_image_data, image_data_size, cudaMemcpyDeviceToHost);

    // // Write the modified image data to a file
    // FILE *output_file = fopen("output_image.ppm", "wb");
    // fwrite(image_data, sizeof(char), image_data_size, output_file);
    // fclose(output_file);

    // Write the modified image data to a file
    FILE *output_file = fopen("output_image.ppm", "wb");
    if (output_file == NULL) {
        fprintf(stderr, "Failed to open output file\n");
        exit(EXIT_FAILURE);
    }

    // Write the PPM header
    fprintf(output_file, "P6\n%d %d\n255\n", IMAGE_WIDTH, IMAGE_HEIGHT);

    // Write the image data to the file
    size_t pixel_count = IMAGE_WIDTH * IMAGE_HEIGHT * 3; // Assuming 3 bytes per pixel (RGB)
    size_t bytes_written = fwrite(image_data, sizeof(char), pixel_count, output_file);
    if (bytes_written != pixel_count) {
        fprintf(stderr, "Error writing image data to file\n");
        exit(EXIT_FAILURE);
    }

    // Close the file
    fclose(output_file);

    // Check if image data was copied back successfully
    cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        fprintf(stderr, "Failed to copy modified image data back to host: %s\n", cudaGetErrorString(cuda_error));
        exit(EXIT_FAILURE);
    }

    // Free device memory
    cudaFree(d_image_data);

    // Check if device memory was freed successfully
    cuda_error = cudaGetLastError();

    if (cuda_error != cudaSuccess) {
        fprintf(stderr, "Failed to free device memory: %s\n", cudaGetErrorString(cuda_error));
        exit(EXIT_FAILURE);
    }
    else {
        printf("Successfully freed device memory\n");
    }

    // Sync the device
    cudaDeviceSynchronize();
 
}
