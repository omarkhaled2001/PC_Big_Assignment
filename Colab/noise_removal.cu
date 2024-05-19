#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

using namespace cv;

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << " Error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void unsharpMaskKernel(unsigned char* d_in, unsigned char* d_out, int width, int height, int channels, float* d_kernel, int kernelRadius) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        float sharpenedPixel[3] = {0.0f, 0.0f, 0.0f};
        float originalPixel[3] = {0.0f, 0.0f, 0.0f};

        int pixIdx = (row * width + col) * channels;
        for (int c = 0; c < channels; ++c) {
            originalPixel[c] = d_in[pixIdx + c];
        }

        for (int blurRow = -kernelRadius; blurRow <= kernelRadius; ++blurRow) {
            for (int blurCol = -kernelRadius; blurCol <= kernelRadius; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                if (curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
                    int curIdx = (curRow * width + curCol) * channels;
                    float kernelVal = d_kernel[(blurRow + kernelRadius) * (2 * kernelRadius + 1) + (blurCol + kernelRadius)];
                    for (int c = 0; c < channels; ++c) {
                        sharpenedPixel[c] += d_in[curIdx + c] * kernelVal;
                    }
                }
            }
        }

        for (int c = 0; c < channels; ++c) {
            d_out[pixIdx + c] = min(max(5 * originalPixel[c] - sharpenedPixel[c], 0.0f), 255.0f);
        }
    }
}

void createUnsharpMaskKernel(float* kernel, int radius, float sigma) {
    int size = 2 * radius + 1;
    float sum = 0.0f;
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            float exponent = -(x * x + y * y) / (2 * sigma * sigma);
            kernel[(y + radius) * size + (x + radius)] = exp(exponent);
            sum += kernel[(y + radius) * size + (x + radius)];
        }
    }
    for (int i = 0; i < size * size; ++i) {
        kernel[i] /= sum;
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image-path>" << std::endl;
        return -1;
    }

    // Load the image
    Mat img = imread(argv[1], IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Could not open or find the image" << std::endl;
        return -1;
    }

    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();
    int imgSize = width * height * channels;

    // Allocate host memory
    unsigned char* h_in = img.data;
    unsigned char* h_out = (unsigned char*)malloc(imgSize);

    // Allocate device memory
    unsigned char *d_in, *d_out;
    checkCudaError(cudaMalloc((void**)&d_in, imgSize), "cudaMalloc d_in failed");
    checkCudaError(cudaMalloc((void**)&d_out, imgSize), "cudaMalloc d_out failed");

    // Copy input data from host to device
    checkCudaError(cudaMemcpy(d_in, h_in, imgSize, cudaMemcpyHostToDevice), "cudaMemcpy h_in to d_in failed");

    // Define Unsharp Mask kernel parameters
    int kernelRadius = 2;
    float sigma = 1.0f;
    int kernelSize = 2 * kernelRadius + 1;
    float* h_kernel = (float*)malloc(kernelSize * kernelSize * sizeof(float));
    createUnsharpMaskKernel(h_kernel, kernelRadius, sigma);

    // Allocate device memory for kernel
    float* d_kernel;
    checkCudaError(cudaMalloc((void**)&d_kernel, kernelSize * kernelSize * sizeof(float)), "cudaMalloc d_kernel failed");

    // Copy kernel data from host to device
    checkCudaError(cudaMemcpy(d_kernel, h_kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy h_kernel to d_kernel failed");

    // Define block and grid sizes
    int blockSize = 16;
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

    // Launch the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    unsharpMaskKernel<<<dimGrid, dimBlock>>>(d_in, d_out, width, height, channels, d_kernel, kernelRadius);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU Time: " << milliseconds << " ms" << std::endl;

    // Copy output data from device to host
    checkCudaError(cudaMemcpy(h_out, d_out, imgSize, cudaMemcpyDeviceToHost), "cudaMemcpy d_out to h_out failed");

    // Create output image and save it
    Mat outImg(height, width, CV_8UC3, h_out);
    imwrite("sharpened_image.jpg", outImg);

    // Clean up
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_kernel);
    free(h_kernel);
    free(h_out);

    std::cout << "Image sharpening completed!" << std::endl;
    return 0;
}
