#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <cuda_runtime.h>
#include <chrono>  // for high_resolution_clock

using namespace std; 

void colorManagement(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& lut, int dimX, int dimY);

int main(int argc, char** argv)
{
    cv::namedWindow("Original video", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Original Dst", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Processed video", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

    cv::VideoCapture video (argv[1]);

    if (!video.isOpened()) {
        std::cout << "Error opening video stream or file" << endl;
        return -1;
    }

    cv::Mat_<cv::Vec3b> h_lut = cv::imread(argv[2]);
    cv::cuda::GpuMat d_lut;
    d_lut.upload(h_lut);

    while (1) {

        cv::Mat_<cv::Vec3b> frame;
        cv::Mat_<cv::Vec3b> h_result;
        cv::cuda::GpuMat d_img;
        cv::cuda::GpuMat d_result;


        video >> frame;      
        if (frame.empty())
            break;

        d_img.upload(frame);
        d_result.upload(frame);

        colorManagement(d_img, d_result, d_lut, 32, 32);

        d_result.download(h_result);

        cv::imshow("Original video", frame);
        cv::imshow("Original Dst", h_lut);
        cv::imshow("Processed video", h_result);

        char c = (char)cv::waitKey(25);
        if (c == 27)
            break;
    }

    video.release();
    cv::destroyAllWindows();

    return 0;
}