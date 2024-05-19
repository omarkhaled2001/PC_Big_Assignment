#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib> // for atoi

using namespace cv;

#define CHECK_CUDA_ERROR(call)                                           \
    do                                                                   \
    {                                                                    \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess)                                          \
        {                                                                \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

// Kernel to perform erosion
__global__ void erosionKernel(unsigned char *input, unsigned char *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int pixel_value = 255; // Assume a binary image where 255 is white and 0 is black
        for (int ky = -1; ky <= 1; ++ky)
        {
            for (int kx = -1; kx <= 1; ++kx)
            {
                int ix = x + kx;
                int iy = y + ky;
                if (ix >= 0 && ix < width && iy >= 0 && iy < height)
                {
                    pixel_value = min(pixel_value, input[iy * width + ix]);
                }
            }
        }
        output[y * width + x] = pixel_value;
    }
}

// Kernel to perform dilation
__global__ void dilationKernel(unsigned char *input, unsigned char *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int pixel_value = 0; // Assume a binary image where 255 is white and 0 is black
        for (int ky = -1; ky <= 1; ++ky)
        {
            for (int kx = -1; kx <= 1; ++kx)
            {
                int ix = x + kx;
                int iy = y + ky;
                if (ix >= 0 && ix < width && iy >= 0 && iy < height)
                {
                    pixel_value = max(pixel_value, input[iy * width + ix]);
                }
            }
        }
        output[y * width + x] = pixel_value;
    }
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        fprintf(stderr, "Usage: %s <image_path>, %d <erosion_iterations> , %d <dilation_iterations>  \n", argv[0], argv[1], argv[2]);
        return -1;
    }

    // Load image using OpenCV
    Mat image = imread(argv[1], IMREAD_GRAYSCALE);
    if (image.empty())
    {
        fprintf(stderr, "Could not open or find the image\n");
        return -1;
    }

    int width = image.cols;
    int height = image.rows;

    // Number of iterations for erosion
    int erosion_iterations = atoi(argv[2]);

    // Number of iterations for dilation
    int dilation_iterations = atoi(argv[3]);

    // Allocate memory for the input and output images on the device
    unsigned char *d_input;
    unsigned char *d_output;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_input, width * height * sizeof(unsigned char)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_output, width * height * sizeof(unsigned char)));

    // Copy the input image to device memory
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, image.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Perform multiple iterations of the erosion kernel
    for (int i = 0; i < erosion_iterations; ++i)
    {
        erosionKernel<<<grid, block>>>(d_input, d_output, width, height);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaMemcpy(d_input, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToDevice));
    }

    // Copy the result back to host and save it
    CHECK_CUDA_ERROR(cudaMemcpy(image.data, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    imwrite("eroded_image.png", image);

    // Reset the input image on device memory for dilation
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, image.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Perform dilation kernel once for demonstration
    for (int i = 0; i < dilation_iterations; ++i)
    {
        dilationKernel<<<grid, block>>>(d_input, d_output, width, height);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaMemcpy(d_input, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToDevice));
    }

    // Copy the result back to host and save it
    CHECK_CUDA_ERROR(cudaMemcpy(image.data, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    imwrite("dilated_image.png", image);

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));

    return 0;
}
