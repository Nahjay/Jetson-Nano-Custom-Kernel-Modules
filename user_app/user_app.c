// Create User App to interact with the kernel module
// User app will pass in an image file to the kernel module

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
#include "cuda_kernel.h"


#define DEVICE_1 "/dev/modify_ppm_colors_cpu"
#define MAX_PATH 256
#define IMAGE_WIDTH 1280
#define IMAGE_HEIGHT 853
#define IMAGE_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT * 3)
#define METADATA 100
#define IOCTL_CMD_PROCESS_IMAGE _IOWR('k', 1, char *)

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

// Function to read image data from second file passed in and return it as a char*
char *read_image_data(const char *image_path) {
    // Open file
    FILE *file = fopen(image_path, "rb");

    // Check if file was opened successfully
    if (file == NULL) {
        fprintf(stderr, "Failed to open file\n");
        exit(EXIT_FAILURE);
    }
    else {
        printf("Successfully opened file\n");
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    rewind(file);

    // Allocate memory for image data
    char *image_data = (char*) malloc(file_size * sizeof(char));

    // Check if memory was allocated successfully
    if (image_data == NULL) {
        fprintf(stderr, "Failed to allocate memory for image data\n");
        exit(EXIT_FAILURE);

        // cleanup
        free(image_data);
        image_data = NULL;
    }
    else {
        printf("Successfully allocated memory for image data\n");
    }

    // Read image data from file
    size_t result = fread(image_data, 1, file_size, file);

    // Check if image data was read successfully
    if (result != file_size) {
        fprintf(stderr, "Failed to read image data\n");
        exit(EXIT_FAILURE);

        // cleanup
        free(image_data);
        image_data = NULL;
    }
    else {
        printf("Successfully read image data\n");
    }

    // Close file
    fclose(file);

    return image_data;
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
    cuda_kfernel<<<gridSize, blockSize>>>(d_image_data, IMAGE_WIDTH, IMAGE_HEIGHT, METADATA);

    // Check for kernel launch errors
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(cuda_error));
        exit(EXIT_FAILURE);
    }

    // Copy the modified image data back to host memory
    cudaMemcpy(image_data, d_image_data, image_data_size, cudaMemcpyDeviceToHost);

    // Check if image data was copied back successfully
    cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        fprintf(stderr, "Failed to copy modified image data back to host: %s\n", cudaGetErrorString(cuda_error));
        exit(EXIT_FAILURE);
    }

    // Free device memory
    cudaFree(d_image_data);
}

// Main function
int main (int argc, char *argv[]) {

    // Quick error checking to make sure user app is being used correctly
    printf("User app started\n");
    printf("You have inputed %d arguments\n", argc);

    if (argc != 4) {
        fprintf(stderr, "Please only input two files, and the path to the cuda shared library. Usage: %s <image_path>\n", argv[0]);
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

    // Check if the path to the cuda shared library is valid
    if (access(argv[3], F_OK) == -1) {
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

    // Obtain image data from file
    char *image_data = read_image_data(argv[2]); 

    // Process image data using Cuda
    process_image_data(image_data);

    // Update user
    printf("User app finished for cuda implementation.\n");

    return 0;
        
}



