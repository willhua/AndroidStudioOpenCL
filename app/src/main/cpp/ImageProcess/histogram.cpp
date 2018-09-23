//
// Created by lisan on 2018/9/19.
//
#include <string.h>
#include "../Utils/OpenCLUtils.h"
#include "../Utils/LogUtils.h"
#include <opencv2/opencv.hpp>

void histogramTest(const uchar *img, const int w, const int h, int *outhist){
    LOGD("histogramTest   w%d h%d", w,h);

    const char * src = R"(

    __constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

    __kernel void hist(read_only image2d_t img, int num_pixel,  __global int * histdata)
{
    int imgw = get_image_width(img);
    int imgh = get_image_height(img);

    local int tmp_hist[256];  //本wg的hist
    //histdata中的索引起始值
    int global_index = 256 * (get_group_id(1) * get_num_groups(0) + get_group_id(0));
    int local_id = get_local_id(1) * get_local_size(0) + get_local_id(0);
    int local_size = get_local_size(0) * get_local_size(1);

    int rang = 256, index = local_id;

    //init tmp hist
    while(index < 256)
    {
        tmp_hist[index] = 0;
        index += local_size;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int x = get_global_id(0); x < imgw ; x += get_global_size(0) ){
        for(int y = get_global_id(1); y < imgh; y += get_global_size(1)){
            int2 coord = (int2)(x, y);
            uint4 gray = read_imageui(img, sampler, coord);
            atomic_inc(&tmp_hist[gray.x]);
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    if(get_local_id(0) == 0 && get_local_id(1) == 0){
        for(int i = 0, y = global_index; i < 256; ++i, ++y){
            histdata[y] = tmp_hist[i];
        }
    }
}
)";

    const char *src_merge = R"(
    __kernel void merge(__global int * tmp_hist, __global int *result, int cnt)
{

}
)";

    cl_device_id device = createDevice(CL_DEVICE_TYPE_GPU);
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_program program = buildProgram(context, device, src, strlen(src));

    cl_int err;

    const unsigned int pixel_per_work_item = 10;
    const float scale = w / (float)h;
    const unsigned int WGX = 10;
    const unsigned int WGY = (int)(WGX / scale + 1);
    const unsigned int WT = w / WGX / pixel_per_work_item + 1;

    size_t work_group_cnt[] = {WGX, WGY};   //每个维度的wg数
    size_t local_work_item[] = {WT, WT};    //每个wg的work_item数
    //全局的work_item数
    size_t global_item_size[] = {work_group_cnt[0] * local_work_item[0],
                                 work_group_cnt[1] * local_work_item[1]};
    LOGD("wg:%d %d, local:%d %d, global item:%d %d", work_group_cnt[0], work_group_cnt[1], local_work_item[0], local_work_item[1],
         global_item_size[0], global_item_size[1]);
    //用来接收每个wg统计的hist
    const int hist_size = sizeof(int32_t) * 256 * work_group_cnt[0] * work_group_cnt[1];
    int32_t *tmp_hist = (int32_t*)malloc(hist_size);
    //memset(tmp_hist, 0, sizeof(int32_t) * 256 * work_group_cnt[0] * work_group_cnt[1]);

    cl_kernel kernel = clCreateKernel(program, "hist", &err);
    checkCLError(err, "clCreateKernel");
    cl_mem tmp_hist_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, hist_size,
                                         nullptr, &err);
    checkCLError(err, "clCreateBuffer  tmp_hist_mem ");
    cl_image_format image_format;
    image_format.image_channel_order = CL_R;
    image_format.image_channel_data_type = CL_UNSIGNED_INT8;
    cl_mem img_mem = clCreateImage2D(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
            &image_format, w, h, w, (void*)img, &err);
    checkCLError(err, "clCreateImage2D   ");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &img_mem);
    checkCLError(err, "clSetKernelArg 0   ");
    err = clSetKernelArg(kernel, 1, sizeof(int), &pixel_per_work_item);
    checkCLError(err, "clSetKernelArg 1   ");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &tmp_hist_mem);
    checkCLError(err, "clSetKernelArg 2   ");

    cl_command_queue  cmdqueue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    checkCLError(err, "clCreateCommandQueue   ");
    size_t offset[] = {0, 0};
    err = clEnqueueNDRangeKernel(cmdqueue, kernel, 2, offset, global_item_size, local_work_item,
                                 0, nullptr, nullptr);
    clFinish(cmdqueue);
    checkCLError(err, "clEnqueueNDRangeKernel   ");



    err = clEnqueueReadBuffer(cmdqueue, tmp_hist_mem, CL_TRUE, 0, hist_size, tmp_hist, 0,
                              nullptr, nullptr);
    clFinish(cmdqueue);
    checkCLError(err, "clEnqueueReadBuffer   ");

    clReleaseMemObject(img_mem);
    clReleaseMemObject(tmp_hist_mem);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdqueue);
    clReleaseContext(context);

    //merge
    const int wg_cnt = work_group_cnt[0] * work_group_cnt[1];

    for(int i = 0; i < 256; ++i){
        outhist[i] = 0;
        for(int j = 0; j < wg_cnt; ++j){
            outhist[i] += tmp_hist[i + j * 256];
        }
    }
    free(tmp_hist);



}
