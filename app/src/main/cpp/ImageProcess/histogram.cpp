//
// Created by lisan on 2018/9/19.
//
#include "../Utils/OpenCLUtils.h"
#include "../Utils/LogUtils.h"

void histogramTest(){

    const char * src = R"(

    __constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

    __kernel void hist(read_only image2d_t img, __global int * histdata)
{

}
)";

}
