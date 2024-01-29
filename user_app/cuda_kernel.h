// cuda_kernel.h

#ifndef CUDA_KERNEL_H
#define CUDA_KERNEL_H

__global__ void cuda_kernel(unsigned char *image_data, size_t width, size_t height, size_t metadata_size);

#endif /* CUDA_KERNEL_H */
