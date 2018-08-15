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
#include<stdlib.h>
#include <cl.h>

#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "LYH", __VA_ARGS__)


#include "opencl.h"
#include "OpenCLUtils.h"

const int ARRAY_SIZE = 100000;

//一、 选择OpenCL平台并创建一个上下文
cl_context CreateContext() {
    cl_int errNum;
    cl_uint numPlatforms;
    cl_context context = NULL;
    cl_platform_id firstPlatformId;

    //选择可用的平台中的第一个
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0) {
        LOGD("Failed to find any OpenCL platforms.");
        return NULL;
    }

    char platforminfo[128];
    clGetPlatformInfo(firstPlatformId, CL_PLATFORM_NAME, sizeof(platforminfo), platforminfo, NULL);
    LOGD("platforminfo:%s", platforminfo);

    //创建一个OpenCL上下文环境
    cl_context_properties contextProperties[] =
            {
                    CL_CONTEXT_PLATFORM,
                    (cl_context_properties) firstPlatformId,
                    0
            };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errNum);

    return context;
}


//四、 创建设备并创建命令队列
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device) {
    cl_device_id *devices;
    cl_int errNum;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // 获取设备缓冲区大小
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);

    if (deviceBufferSize <= 0) {
        std::cerr << "No devices available.";
        return NULL;
    }

    // 为设备分配缓存空间
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);

    //选取可用设备中的第一个
    commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);

    *device = devices[0];

    cl_bool edian;
    clGetDeviceInfo(devices[0], CL_DEVICE_ENDIAN_LITTLE, sizeof(cl_bool), &edian, NULL);
    LOGD("EDIAN %d", edian);


    delete[] devices;
    return commandQueue;
}


// 二、创建和构建程序对象
cl_program CreateProgram(cl_context context, cl_device_id device, const char *fileName) {
    cl_int errNum;
    cl_program program;
/*
    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        LOGD("Failed to open file for reading: %s\n" , fileName );
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();
    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
*/

    const char *srcStr = R"(
__kernel void hello_kernel(__global const float *a,
                           __global const float *b,
                           __global float *result)
{
    int gid = get_global_id(0);

    result[gid] = a[gid] + b[gid];
}
)";
    program = clCreateProgramWithSource(context, 1,
                                        (const char **) &srcStr,
                                        NULL, NULL);

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    char programsource[1024];
    clGetProgramInfo(program, CL_PROGRAM_SOURCE, 1024, programsource, NULL);
    LOGD("programsource  %s", programsource);

    return program;
}

//创建和构建程序对象
bool CreateMemObjects(cl_context context, cl_mem memObjects[3],
                      float *a, float *b) {
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * ARRAY_SIZE, a, NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * ARRAY_SIZE, b, NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * ARRAY_SIZE, NULL, NULL);
    return true;
}


// 释放OpenCL资源
void Cleanup(cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_kernel kernel, cl_mem memObjects[3]) {
    for (int i = 0; i < 3; i++) {
        if (memObjects[i] != 0)
            clReleaseMemObject(memObjects[i]);
    }
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    if (kernel != 0)
        clReleaseKernel(kernel);

    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);
}


int test() {

    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel = 0;
    cl_mem memObjects[3] = {0, 0, 0};
    cl_int errNum;
    // uint64_t t1,t2,t3;
    clock_t t1, t2, t3;


    const char *filename = "HelloWorld.cl";
    // 一、选择OpenCL平台并创建一个上下文
    context = CreateContext();


    //二、创建和构建程序对象
    program = CreateProgram(context, device, filename);//"HelloWorld.cl");

    // 三、 创建OpenCL内核并分配内存空间
    kernel = clCreateKernel(program, "hello_kernel", NULL);

    // 四、 创建设备并创建命令队列
    commandQueue = CreateCommandQueue(context, &device);

    //创建要处理的数据
    float result[ARRAY_SIZE];
    float a[ARRAY_SIZE];
    float b[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        a[i] = (float) i;
        b[i] = (float) (ARRAY_SIZE - i);
    }

    t1 = clock();  //mach_absolute_time();
    LOGD("t1 = %.8f\n", (double) t1);
    for (int j = 0; j < ARRAY_SIZE; j++) {
        result[j] = a[j] + b[j];
    }

    t2 = clock(); //mach_absolute_time();
    LOGD("t2 = %.8f\n", (double) t2);

    //创建内存对象
    //memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    //                               sizeof(float) * ARRAY_SIZE, a, NULL);
    //memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    //                               sizeof(float) * ARRAY_SIZE, b, NULL);
    //memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
    //                               sizeof(float) * ARRAY_SIZE, NULL, NULL);
    if (!CreateMemObjects(context, memObjects, a, b)) {
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }

    // 五、 设置内核数据并执行内核
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);

    size_t globalWorkSize[1] = {ARRAY_SIZE};
    size_t localWorkSize[1] = {1};

    errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    0, NULL, NULL);

    // 六、 读取执行结果并释放OpenCL资源
    errNum = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE,
                                 0, ARRAY_SIZE * sizeof(float), result,
                                 0, NULL, NULL);

    t3 = clock();  //mach_absolute_time();




    LOGD("cpu t = %.8f\n", (float) (t2 - t1) / CLOCKS_PER_SEC);
    LOGD("gpu t = %.8f \n", (double) (t3 - t2) / CLOCKS_PER_SEC);


    LOGD("Executed program succesfully.");
    getchar();
    Cleanup(context, commandQueue, program, kernel, memObjects);

    return 0;
}



void mycl()
{
    const char *src1 = R"(__kernel void reduce(__global float *data,  __global float *outdata, __local float *tmp)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);

    tmp[lid] = data[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = get_local_size(0) / 2; i > 0; i >= 1){
        if(lid < i){
            tmp[lid] += tmp[lid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(lid == 0){
        outdata[get_group_id(0)] = tmp[0];
    }
})";

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
    float sum = 0;
    size_t localSize = 0;
    cl_device_id device = createDevice(CL_DEVICE_TYPE_GPU);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(localSize), &localSize, NULL);
    LOGD("LOCALSIZE work group:%d", localSize);
    int groupCnt = SIZE / localSize;
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_program program = buildProgram(context, device, src2, strlen(src2));
    //cl_program program = buildProgramFromFile(context, device, "CLKernels.cl");
    cl_command_queue cmdqueue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);


    float *data = (float*)malloc(sizeof(float) * SIZE);
    for (int i = 0; i < SIZE; ++i) {
        data[i] = 1.0 * i;
    }
    float *tmpdata = (float*)malloc(sizeof(float) * groupCnt);
    for(int i = 0; i < groupCnt; ++i){ tmpdata[i] = 0.0; }
    cl_mem inbuffer, sumbuffer, tmpbuffer;
    //注意使用CL_MEM_USE_HOST_PTR
    inbuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, SIZE * sizeof(float), data, &err);
    tmpbuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , sizeof(float) * groupCnt, tmpdata, &err);
    //sumbuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float), NULL, &err);

    cl_kernel kernel1 = clCreateKernel(program, "reduction_scalar", &err);
    err = clSetKernelArg(kernel1, 0, sizeof(cl_mem), &inbuffer);
    err = clSetKernelArg(kernel1, 1, sizeof(float) * localSize, NULL);
    err = clSetKernelArg(kernel1, 2, sizeof(cl_mem), &tmpbuffer);
    err = clEnqueueNDRangeKernel(cmdqueue, kernel1, 1, NULL, &SIZE, &localSize, 0, NULL, NULL);

    clFinish(cmdqueue);

    err = clEnqueueReadBuffer(cmdqueue, tmpbuffer, CL_TRUE, 0, groupCnt * sizeof(float), tmpdata, 0, NULL, NULL);
    clFinish(cmdqueue);
    sum = 0.0;
    for (int i = 0; i < groupCnt; ++i) {
        sum += tmpdata[i];
    }
    LOGD("cl sum:%f, %f", sum, (1.0 + SIZE) * SIZE / 2.0);




}


extern "C" JNIEXPORT jstring

JNICALL
Java_willhua_androidstudioopencl_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    LOGD("test start");
    //test();
    mycl();
    LOGD("test end");
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}
