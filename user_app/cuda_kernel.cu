#include "cuda_kernel.h"
// Updated CUDA kernel function to process image data
__global__ void cuda_kernel(unsigned char *image_data, size_t width, size_t height, size_t metadata_size) {
    // Calculate global index
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Skip modifying metadata
    if (index < metadata_size) {
        return;
    }

    // Calculate pixel index
    int pixel_index = (index - metadata_size) / 3;

    // Calculate channel index
    int channel_index = (index - metadata_size) % 3;

    // Ensure index is within image data bounds
    if (pixel_index < width * height) {
        // Increment RGB values by 10
        image_data[index] = (image_data[index] + 10) % 256;
    }
}

