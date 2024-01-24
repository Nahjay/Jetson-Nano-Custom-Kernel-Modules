// Create Cuda Kernel for image processing.

// Create global function for image processing.

__global__ void cuda_kernel(unsigned char *data, size_t width, size_t height, size_t metadata_size) {

    // Calculate global index in the array.
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Skip modifying the metadata.
    if (index < metadata_size) {
        return;
    }

    // Calculate the pixel index.
    if (index < width * height * 3) {

        // Increment RGB values by 10.
        data[index] = (data[index] + 10) % 256;
    }


}