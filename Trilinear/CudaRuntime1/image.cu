#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

__device__ int3 trilinear(float3 pPoint)
{
    float3 below = { floor((double)pPoint.x), floor((double)pPoint.y), floor((double)pPoint.z) };
    float3 above = { ceil((double)pPoint.x), ceil((double)pPoint.y), ceil((double)pPoint.z) };

    float deltaX = (pPoint.x - below.x) / (above.x - below.x);
    float deltaY = (pPoint.y - below.y) / (above.y - below.y);
    float deltaZ = (pPoint.z - below.z) / (above.z - below.z);

    deltaX = deltaX > 0 ? deltaX : 0 ;
    deltaY = deltaY > 0 ? deltaY : 0 ;
    deltaZ = deltaZ > 0 ? deltaZ : 0 ;

    float3 p000 = { below.x, below.y, above.z };
    float3 p001 = { below.x, above.y, above.z };
    float3 p010 = { below.x, below.y, below.z };
    float3 p011 = { below.x, above.y, below.z };
    float3 p100 = { above.x, below.y, above.z };
    float3 p101 = { above.x, above.y, above.z };
    float3 p110 = { above.x, below.y, below.z };
    float3 p111 = { above.x, above.y, below.z };

    float3 p00 = { p000.x * (1 - deltaX) + p100.x * deltaX, p000.y * (1 - deltaX) + p100.y * deltaX , p000.z * (1 - deltaX) + p100.z * deltaX };
    float3 p01 = { p001.x * (1 - deltaX) + p101.x * deltaX, p001.y * (1 - deltaX) + p101.y * deltaX , p001.z * (1 - deltaX) + p101.z * deltaX };
    float3 p11 = { p010.x * (1 - deltaX) + p110.x * deltaX, p010.y * (1 - deltaX) + p110.y * deltaX , p010.z * (1 - deltaX) + p110.z * deltaX };
    float3 p10 = { p011.x * (1 - deltaX) + p111.x * deltaX, p010.y * (1 - deltaX) + p110.y * deltaX , p010.z * (1 - deltaX) + p110.z * deltaX };

    float3 p0 = { p00.x * (1 - deltaY) + p10.x * deltaY, p00.y * (1 - deltaY) + p10.y * deltaY, p00.z * (1 - deltaY) + p10.z * deltaY };
    float3 p1 = { p01.x * (1 - deltaY) + p11.x * deltaY, p01.y * (1 - deltaY) + p11.y * deltaY, p01.z * (1 - deltaY) + p11.z * deltaY };

    return { (int)(p0.x * (1 - deltaZ) + p1.x * deltaZ), (int)(p0.y * (1 - deltaZ) + p1.y * deltaZ), (int)(p0.z * (1 - deltaZ) + p1.z * deltaZ) };
}

__global__ void color(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, cv::cuda::PtrStep<uchar3> lut, int rows, int cols)
{
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;
    int3 finalP;

    if (dst_x < cols&& dst_y < rows)
    {
        float3 pPoint = { (float)src(dst_y, dst_x).x / 4.0, (float)src(dst_y, dst_x).y / 4.0, (float)src(dst_y, dst_x).z / 4.0 };

        pPoint.x = pPoint.x > 63 ? 63 : pPoint.x;
        pPoint.y = pPoint.y > 63 ? 63 : pPoint.y;
        pPoint.z = pPoint.z > 63 ? 63 : pPoint.z;

        finalP = trilinear(pPoint);

        dst(dst_y, dst_x) = lut(finalP.z, finalP.y * 64 + finalP.x);
    }
}

int divUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void colorManagement(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& lut, int dimX, int dimY)
{
    const dim3 block(dimX, dimY);
    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));
    color <<<grid, block >>> (src, dst, lut, dst.rows, dst.cols);
}