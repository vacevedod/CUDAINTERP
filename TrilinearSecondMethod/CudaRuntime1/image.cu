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

    float3 c0 = { p000 };
    float3 c1 = { p100.x -p000.x,p100.y - p000.y,p100.z - p000.z };
    float3 c2 = { p010.x -p000.x,p010.y - p000.y,p010.z - p000.z };
    float3 c3 = { p001.x -p000.x,p001.y - p000.y,p001.z - p000.z };
    float3 c4 = { p110.x -p010.x-p100.x+p000.x,p110.y - p010.y - p100.y + p000.y ,p110.z - p010.z - p100.z + p000.z };
    float3 c5 = { p011.x - p001.x - p010.x + p000.x,p011.y - p001.y - p010.y + p000.y ,p011.z - p001.z - p010.z + p000.z };
    float3 c6 = { p101.x - p001.x - p100.x + p000.x, p101.y - p001.y - p100.y + p000.y,p101.z - p001.z - p100.z + p000.z };
    float3 c7 = { p111.x -p011.x-p101.x - p110.x+p100.x + p001.x+p010.x-p000.x, p111.y - p011.y - p101.y - p110.y + p100.y + p001.y + p010.y - p000.y, p111.z - p011.z - p101.z - p110.z + p100.z + p001.z + p010.z - p000.z };


    return { (int)(c0.x + c1.x * deltaX + c2.x * deltaY + c3.x * deltaY + c4.x * deltaX * deltaY + c5.x * deltaY * deltaZ + c6.x * deltaZ * deltaX + c7.x * deltaX * deltaY * deltaZ),
             (int)(c0.y + c1.y * deltaX + c2.y * deltaY + c3.y * deltaY + c4.y * deltaX * deltaY + c5.y * deltaY * deltaZ + c6.y * deltaZ * deltaX + c7.y * deltaX * deltaY * deltaZ),
             (int)(c0.z + c1.z * deltaX + c2.z * deltaY + c3.z * deltaY + c4.z * deltaX * deltaY + c5.z * deltaY * deltaZ + c6.z * deltaZ * deltaX + c7.z * deltaX * deltaY * deltaZ) };
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