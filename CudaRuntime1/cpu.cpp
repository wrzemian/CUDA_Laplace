#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>

using namespace std;
using namespace cv;

// The wrapper is use to call laplacian filter 
extern "C" void laplacianFilter_CPU(const cv::Mat & input, cv::Mat & output)
{
	cv::Mat input_gray;
	int kernel_size = 3;
	int scale = 1;
	int delta = 0;

	/// Remove noise by blurring with a Gaussian filter
	GaussianBlur(input, input, Size(3, 3), 0, 0, BORDER_DEFAULT);

	// laplacian filter
	Laplacian(input, output, CV_16S, kernel_size, scale, delta, BORDER_DEFAULT);


}

