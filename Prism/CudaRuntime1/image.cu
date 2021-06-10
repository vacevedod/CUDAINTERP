#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

__device__ int3 prism(float3 pPoint)
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
    int3 p = { 0 };
    p.x = (deltaX > deltaZ) ? p000.x + (p100.x - p000.x) * deltaX + (p110.x - p100.x) * deltaZ 
        + (p001.x - p000.x) * deltaY + (p101.x - p001.x - p100.x + p000.x) * deltaX * deltaY
        + (p111.x - p101.x - p110.x + p100.x) * deltaZ * deltaY
        : p000.x + (p110.x - p010.x) * deltaX + (p010.x - p000.x) * deltaZ 
        + (p001.x - p000.x) * deltaY + (p111.x - p011.x - p110.x + p010.x) * deltaX * deltaY
        + (p011.x - p001.x - p010.x + p000.x) * deltaZ * deltaY;
        
    p.y = (deltaX > deltaZ) ? p000.y + (p100.y - p000.y) * deltaX + (p110.y - p100.y) * deltaZ 
        + (p001.y - p000.y) * deltaY + (p101.y - p001.y - p100.y + p000.y) * deltaX * deltaY
        + (p111.y - p101.y - p110.y + p100.y) * deltaZ * deltaY :
        p000.y + (p110.y - p010.y) * deltaX + (p010.y - p000.y) * deltaZ 
        + (p001.y - p000.y) * deltaY + (p111.y - p011.y - p110.y + p010.y) * deltaX * deltaY
        + (p011.y - p001.y - p010.y + p000.y) * deltaZ * deltaY;

    p.z = (deltaX > deltaZ) ? p000.z + (p100.z - p000.z) * deltaX + (p110.z - p100.z) * deltaZ 
        + (p001.z - p000.z) * deltaY + (p101.z - p001.z - p100.z + p000.z) * deltaX * deltaY
        + (p111.z - p101.z - p110.z + p100.z) * deltaZ * deltaY:    
        p000.z + (p110.z - p010.z) * deltaX + (p010.z - p000.z) * deltaZ 
        + (p001.z - p000.z) * deltaY + (p111.z - p011.z - p110.z + p010.z) * deltaX * deltaY
        + (p011.z - p001.z - p010.z + p000.z) * deltaZ * deltaY;
    
    return { p };
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

        finalP = prism (pPoint);

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