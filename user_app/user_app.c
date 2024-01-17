// Create User App to interact with the kernel module
// User app will pass in an image file to the kernel module

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <fcntl.h>


#define DEVICE_1 "/dev/kernel_module"
#define DEVICE_2 "/dev/kernel_module2"



int main (int argc, char *argv[]) {


    // Quick error checking to make sure user app is being used correctly
    printf("User app started\n");
    printf("You have inputed %d arguments\n", argc);

    if (argc != 2) {
        printf("Please only input one file name\n");
        return -1;
    }
    
}



