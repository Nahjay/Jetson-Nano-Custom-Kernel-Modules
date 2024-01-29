#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <errno.h>
#include "user_app.h"
#include <dlfcn.h>
#include "/usr/local/cuda-10.2/include/cuda_runtime.h"
#include "/usr/local/cuda-10.2/include/cuda.h"
#include "/usr/local/cuda-10.2/include/device_launch_parameters.h"


#define DEVICE_1 "/dev/modify_ppm_colors_cpu"
#define MAX_PATH 256
#define IMAGE_WIDTH 1280
#define IMAGE_HEIGHT 853
#define IMAGE_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT * 3)
#define METADATA 100
#define IOCTL_CMD_PROCESS_IMAGE _IOWR('k', 1, char *)
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

// Function to open kernel module
int open_kernel_module(const char *device_path) {
    int fd;
    fd = open(device_path, O_RDWR);
    if (fd < 0) {
        perror("Failed to open device");
        exit(EXIT_FAILURE);
    }
    else {
        printf("Successfully opened device\n");
        return fd;
    }
}

// Function to close kernel module
int close_kernel_module(int fd) {
    // close(fd);

    // Check if close was successful
    if (close(fd) == -1) {
        perror("Failed to close device");
        exit(EXIT_FAILURE);
    }
    else {
        printf("Successfully closed device\n");
        return 0;
    }
}

// Function to clean up image path memory
int clean_up_image_path(char *image_path) {
    free(image_path);
    image_path = NULL;

    return 0 ? image_path == NULL : -1;
}

// Function to process image
void process_image(int *fd, const char* input_path) {

    // Allocate memory for image_path and cast to char*
    char* image_path = (char*) malloc(MAX_PATH * sizeof(char));

    
    // Check if input_path is valid
    if (input_path == NULL) {
        fprintf(stderr, "Please input a valid path\n");
        exit(EXIT_FAILURE);

        // cleanup
        clean_up_image_path(image_path);

    }
    // Check if input_path is too long
    else if (strlen(input_path) > MAX_PATH) {
        fprintf(stderr, "Please input a path that is less than %d characters\n", MAX_PATH);
        exit(EXIT_FAILURE);

        // cleanup
        clean_up_image_path(image_path);

        }

    // Copy input_path to image_path
    strcpy(image_path, input_path);

    if (ioctl(*fd, IOCTL_CMD_PROCESS_IMAGE, image_path) == -1) {
        perror("IOCTL_CMD_PROCESS_IMAGE failed");

        // Debugging print statements
        fprintf(stderr, "IOCTL_CMD_PROCESS_IMAGE failed\n");
        fprintf(stderr, "image_path: %s\n", image_path);
        fprintf(stderr, "strlen(image_path): %ld\n", strlen(image_path));
        fprintf(stderr, "sizeof(image_path): %ld\n", sizeof(image_path));
        fprintf(stderr, "Error: %s", strerror(errno));

        // cleanup
        clean_up_image_path(image_path);

        exit(EXIT_FAILURE);
    }
    else {
        printf("IOCTL_CMD_PROCESS_IMAGE successful\n");

        // cleanup
        clean_up_image_path(image_path);
    }
}

int main (int argc, char *argv[]) {

    // Quick error checking to make sure user app is being used correctly
    printf("User app started\n");
    printf("You have inputed %d arguments\n", argc);

    if (argc != 3) {
        fprintf(stderr, "Please only input two files. Usage: %s <image_path>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    else {
        printf("You have inputed '%s' and '%s' as the image path\n", argv[1], argv[2]);
    }

    // Check if the file exists before passing it to the kernel module
    if (access(argv[1], F_OK) == -1) {
        fprintf(stderr, "File does not exist\n");
        exit(EXIT_FAILURE);
    }
    else {
        printf("File exists\n");
    }

    // Check if the file exists before passing it to the cuda shared library
    if (access(argv[2], F_OK) == -1) {
        fprintf(stderr, "File does not exist\n");
        exit(EXIT_FAILURE);
    }
    else {
        printf("File exists\n");
    }

   // Open kernel module
    int fd = open_kernel_module(DEVICE_1);

    // Process image
    process_image(&fd, argv[1]);

    // Close kernel module
    close_kernel_module(fd);

    // Update user
    printf("User app finished for first kernel module.\n");

    // Check if cuda function is being called correctly
    printf("Calling cuda function\n");

    FILE *file = fopen(argv[2], "rb+");
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
    FILE *output_file = fopen(argv[2], "wb");
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