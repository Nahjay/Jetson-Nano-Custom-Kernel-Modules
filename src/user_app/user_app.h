#ifndef USER_APP_H
#define USER_APP_H

#define MAX_PATH 256

int open_kernel_module(const char* device_path);
int close_kernel_module(int fd);
void process_image(int *fd, const char* input_path);
int clean_up_image_path(char* image_path);
char *read_image_data(const char *image_path);

#endif // USER_APP_H