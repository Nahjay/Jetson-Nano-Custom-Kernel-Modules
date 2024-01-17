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

// Open device
int open_device(char *device) {
    int fd = open(device, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "Could not open device %s\n", device);
        exit(EXIT_FAILURE);
    }
    return fd;
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







    
}



