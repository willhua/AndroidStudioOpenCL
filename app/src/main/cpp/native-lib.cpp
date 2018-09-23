#include <jni.h>
#include <stdlib.h>
#include <string>
#include <opencl.h>
#include <android/log.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <sys/time.h>
#include<time.h>
#include<stdio.h>
#include <cl.h>
#include <string.h>
#include "Utils/OpenCLUtils.h"
#include "MapReduce/MapReduce.h"

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ImageProcess/imageprocess.h"

#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "LYH", __VA_ARGS__)


AAssetManager *pAssetManager;
cv::Mat readImage(const char *path , int mode = CV_LOAD_IMAGE_GRAYSCALE, AAssetManager *assetManager = nullptr){
    LOGD("readimage %p   mode:%d", assetManager, mode);
    if(assetManager == nullptr){
        return cv::imread(path, mode);
    }else{
        AAsset *file = AAssetManager_open(assetManager, path, AASSET_MODE_BUFFER);
        const char *data = (const char *)AAsset_getBuffer(file);
        std::vector<char> vec(data, data + AAsset_getLength(file));
        AAsset_close(file);
        return cv::imdecode(vec, mode);
    }
}


void mycl() {
    const char *src2 = R"(
__kernel void reduction_scalar(__global float* data,
      __local float* partial_sums, __global float* output) {

   int lid = get_local_id(0);
   int group_size = get_local_size(0);

   partial_sums[lid] = data[get_global_id(0)];
   barrier(CLK_LOCAL_MEM_FENCE);

   for(int i = group_size/2; i>0; i >>= 1) {
      if(lid < i) {
         partial_sums[lid] += partial_sums[lid + i];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if(lid == 0) {
      output[get_group_id(0)] = partial_sums[0];
   }
}
)";
    const size_t SIZE = 1 << 20;


    int err = 0;
    size_t localSize = 0;
    cl_device_id device = createDevice(CL_DEVICE_TYPE_GPU);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(localSize), &localSize, NULL);
    LOGD("LOCALSIZE work group:%d", localSize);
    int groupCnt = SIZE / localSize;
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_program program = buildProgram(context, device, src2, strlen(src2));
    cl_command_queue cmdqueue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE,
                                                     &err);
    cl_event prof_event;

    float *data = (float *) malloc(sizeof(float) * SIZE);
    for (int i = 0; i < SIZE; ++i) {
        data[i] = 1.0 * i;
    }
    float *group_sum = (float *) malloc(sizeof(float) * groupCnt);
    for (int i = 0; i < groupCnt; ++i) { group_sum[i] = 0.0; }
    cl_mem inbuffer, outbuffer;
    //注意使用CL_MEM_USE_HOST_PTR
    inbuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                              SIZE * sizeof(float), data, &err);
    outbuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                               sizeof(float) * groupCnt, group_sum, &err);

    cl_kernel kernel1 = clCreateKernel(program, "reduction_scalar", &err);
    err = clSetKernelArg(kernel1, 0, sizeof(cl_mem), &inbuffer);
    err = clSetKernelArg(kernel1, 1, sizeof(float) * localSize, NULL);
    err = clSetKernelArg(kernel1, 2, sizeof(cl_mem), &outbuffer);
    err = clEnqueueNDRangeKernel(cmdqueue, kernel1, 1, NULL, &SIZE, &localSize, 0, NULL,
                                 &prof_event);

    clFinish(cmdqueue);

    size_t time_start, time_end;
    clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START,
                            sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end,
                            NULL);
    LOGD("cmd time %lu", time_end - time_start);

    err = clEnqueueReadBuffer(cmdqueue, outbuffer, CL_TRUE, 0, groupCnt * sizeof(float), group_sum,
                              0, NULL, NULL);
    clFinish(cmdqueue);
    float sum = 0.0;
    for (int i = 0; i < groupCnt; ++i) {
        sum += group_sum[i];
    }
    LOGD("cl sum:%f, %f", sum, (1.0 + SIZE) * SIZE / 2.0);

    clReleaseKernel(kernel1);
    clReleaseMemObject(inbuffer);
    clReleaseMemObject(outbuffer);
    free(group_sum);
    free(data);
    clReleaseCommandQueue(cmdqueue);
    clReleaseProgram(program);
    clReleaseContext(context);

}


extern "C" JNIEXPORT jstring
JNICALL
Java_willhua_androidstudioopencl_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    LOGD("test start");
    //test();
    //mycl();
    //mapReduceByOCL();
    if(true){
        cv::Mat img = readImage("dir/cs.jpg", CV_LOAD_IMAGE_GRAYSCALE, pAssetManager);
        int hist[256] = {0};
        memset(hist, 0, sizeof(int)*256);
        int hist2[256] = {0};
        memset(hist2, 0, sizeof(int)*256);
        double time = cv::getTickCount();
        histogramTest(img.data, img.cols, img.rows, hist);
        LOGD("cl time:%f", (1000.0 * (cv::getTickCount() - time) / cv::getTickFrequency()));

        time = cv::getTickCount();
        for(int i = 0; i < img.rows; ++i){
            uchar *ptr = img.ptr<uchar>(i);
            for(int j = 0; j < img.cols; ++j){
                ++hist2[ptr[j]];
            }
        }
        LOGD("cpu  time:%f", (1000.0 * (cv::getTickCount() - time) / cv::getTickFrequency()));
        int wrong_cnt = 0;
        for(int i = 0; i < 256; ++i){
            if(hist[i] != hist2[i]){
                LOGD("wrong  %d  diff:%d - %d = %d", i, hist[i], hist2[i], (hist[i] - hist2[i]));
                ++wrong_cnt;
                //break;
            }
        }
        LOGD("compare wrong_cnt:%d", wrong_cnt);
    /*
        int channel[] = {0};
        std::vector<int> histvec;
        float histrange[]  = {0, 255};
        int histsize = 256;
        cv::MatND outhist;
        cv::calcHist(&img, 1,  &channel, cv::Mat(), outhist, 1, &histsize,  &histrange, false, false); */


    }
    LOGD("test end");
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}


extern "C"
JNIEXPORT void JNICALL
Java_willhua_androidstudioopencl_MainActivity_setNativeAssetManager(JNIEnv *env, jobject instance,
                                                                    jobject assetManager) {
    pAssetManager = AAssetManager_fromJava(env, assetManager);
    LOGD("setNativeAssetManager  %p", pAssetManager);
}