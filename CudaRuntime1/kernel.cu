#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 16
#define KERNEL_SIZE 3             

using namespace std;

__global__ void laplacianFilter(unsigned char *srcImage, unsigned char *dstImage, unsigned int width, unsigned int height)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;

   //float kernel[3][3] = {0, -1, 0, -1, 4, -1, 0, -1, 0};
   //float kernel[3][3] = { 0, 1, 0, 1, -4, 1, 0, 1, 0 };
   float kernel[3][3] = {1, 4, 1, 4, -20, 4, 1, 4, 1};
   //float kernel[3][3] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};   
   if((x >= KERNEL_SIZE /2) && (x < (width - KERNEL_SIZE /2)) && (y >= KERNEL_SIZE /2) && (y < (height - KERNEL_SIZE /2)))
   {
         float sum = 0;
         for(int ky = -KERNEL_SIZE / 2; ky <= KERNEL_SIZE / 2; ky++) {
            for(int kx = -KERNEL_SIZE / 2; kx <= KERNEL_SIZE / 2; kx++) {
               float src = srcImage[((y + ky) * width + (x + kx))]; 
               sum += src * kernel[ky + KERNEL_SIZE / 2][kx + KERNEL_SIZE / 2];
            }
         }
         dstImage[(y * width + x)] =  sum;
   }
}


void laplacianFilter_GPU_wrapper(const cv::Mat& input, cv::Mat& output)
{
        const int inputSize = input.cols * input.rows;
        const int outputSize = output.cols * output.rows;
        unsigned char *d_input, *d_output;
        
        cudaMalloc<unsigned char>(&d_input, inputSize);
        cudaMalloc<unsigned char>(&d_output, outputSize);

        cudaMemcpy(d_input, input.ptr(), inputSize, cudaMemcpyHostToDevice);
        const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 grid((output.cols + block.x - 1) / block.x, (output.rows + block.y - 1) / block.y);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        laplacianFilter<<<grid,block>>>(d_input, d_output, output.cols, output.rows);
        cudaEventRecord(stop);

        cudaMemcpy(output.ptr(), d_output, outputSize, cudaMemcpyDeviceToHost);
        cudaFree(d_input);
        cudaFree(d_output);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cout<< "\nTime in miliseconds: " << milliseconds << "\n";
}

int main(int argc, char** argv) {

    string input_file = "test.jpg";
    cv::Mat srcImage = cv::imread(input_file, cv::ImreadModes::IMREAD_UNCHANGED);
    if (srcImage.empty())
    {
        std::cout << "no image";
        return -1;
    }

    cv::cvtColor(srcImage, srcImage, cv::COLOR_BGR2GRAY);
    cv::Mat dstImage(srcImage.size(), srcImage.type());
    laplacianFilter_GPU_wrapper(srcImage, dstImage);
    imwrite("output3.jpg", dstImage);

    return 0;
}