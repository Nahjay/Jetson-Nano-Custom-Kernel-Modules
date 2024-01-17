// Create User App to interact with the kernel module
// User app will pass in an image file to the kernel module

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include "user_app.h"



#define DEVICE_1 "/dev/kernel_module"
#define DEVICE_2 "/dev/kernel_module_2"
#define MAX_PATH 256
#define IOCTL_CMD_PROCESS_IMAGE _IOWR('k', 1, char *)


// Function to open kernel module
int open_kernel_module(char *device) {
    int fd;
    fd = open(device, O_RDWR);
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
    close(fd);

    // Check if close was successful
    if (fd < 0) {
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

    return 0 ? image_path == NULL : 1;
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

    if (ioctl(*fd, IOCTL_CMD_PROCESS_IMAGE, image_path) < 0) {
        perror("IOCTL_CMD_PROCESS_IMAGE failed");

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

    if (argc != 2) {
        fprintf(stderr, "Please only input one file name. Usage: %s <image_path>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    else {
        printf("You have inputed '%s' as the image path\n", argv[1]);
    }

    // Open kernel module
    int fd = open_kernel_module(DEVICE_1);
    







    
}



