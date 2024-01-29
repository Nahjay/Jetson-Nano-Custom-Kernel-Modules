#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "/usr/local/cuda-10.2/include/cuda_runtime.h"
#include "/usr/local/cuda-10.2/include/cuda.h"
#include "/usr/local/cuda-10.2/include/device_launch_parameters.h"

#define MAX_PATH 256
#define METADATA 100
#define IMAGE_WIDTH 1280
#define IMAGE_HEIGHT 853
#define IMAGE_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT * 3) // Assuming RGB format
#define BLOCK_SIZE 256

__global__ void modify_ppm_colors(unsigned char *data, size_t width, size_t height, size_t metadata_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < width * height) {
        int y = idx / width;
        int x = idx % width;
        size_t index = y * width * 3 + x * 3;

        // Skip modifying metadata
        if (index >= metadata_size) {
            // Increment RGB values by 10
            for (int channel = 0; channel < 3; ++channel) {
                data[index + channel] = (data[index + channel] + 10) % 256;
            }
        }

        idx += blockDim.x * gridDim.x;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input_image.ppm>\n", argv[0]);
        return 1;
    }

    FILE *file = fopen(argv[1], "rb+");
    if (!file) {
        perror("Error opening file");
        return 1;
    }

    unsigned char *data = (unsigned char *)malloc(IMAGE_SIZE * sizeof(unsigned char));
    if (!data) {
        perror("Error allocating memory");
        fclose(file);
        return 1;
    }

    // Read the data from the file
    size_t bytesRead = fread(data, sizeof(unsigned char), IMAGE_SIZE, file);
    if (bytesRead != IMAGE_SIZE) {
        perror("Error reading file");
        fclose(file);
        free(data);
        return 1;
    }

    // Close the file
    fclose(file);

    // Launch CUDA kernel
    unsigned char *d_data;
    cudaMalloc(&d_data, IMAGE_SIZE * sizeof(unsigned char));
    cudaMemcpy(d_data, data, IMAGE_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice);
    modify_ppm_colors<<<(IMAGE_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_data, IMAGE_WIDTH, IMAGE_HEIGHT, METADATA);
    cudaMemcpy(data, d_data, IMAGE_SIZE * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Write back modified data to the file
    FILE *output_file = fopen(argv[1], "wb");
    if (!output_file) {
        perror("Error opening output file");
        free(data);
        return 1;
    }
    fwrite(data, sizeof(unsigned char), IMAGE_SIZE, output_file);
    fclose(output_file);

    // Free allocated memory
    free(data);

    return 0;
}