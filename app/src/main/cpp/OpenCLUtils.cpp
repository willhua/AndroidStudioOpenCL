//
// Created by lisan on 2018/8/12.
//

#include <stdlib.h>
#include "OpenCLUtils.h"


cl_device_id createDevice(cl_device_type type) {
    cl_platform_id platform;
    cl_device_id device = 0;
    int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err < 0){
        LOGE("get platform of OCL failed.");
    }

    err = clGetDeviceIDs(platform, type, 1, &device, NULL);
    if(err < 0){
        LOGE("get device id failed");
    }

    return device;
}

cl_program buildProgram(cl_context context, cl_device_id device, const char *src, size_t length)
{
    if(src == nullptr){
        LOGE("buildprogram will NULL src");
        return 0;
    }

    int err = 0;
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&src, &length, &err);
    if(err < 0){
        LOGE("create program with source failed");
        return 0;
    }

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(err < 0){
        size_t logsize;
        //得到log信息的大小
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsize);
        char *logmsg = (char*)malloc(logsize + 1);      //分配的内存加1来放置终止符
        logmsg[logsize] = '\0'; //注意最后一个要放终止符。
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logsize + 1, logmsg, NULL);
        LOGE("build program failed:%s", logmsg);
        free(logmsg);
        return 0;
    }

    return program;
}


cl_program buildProgramFromFile(cl_context context, cl_device_id device, const char * filename)
{
    FILE *stream = fopen(filename, "r");
    fseek(stream, 0, SEEK_END);
    int length = ftell(stream);
    rewind(stream);
    char *src = (char*)malloc(length + 1);
    src[length] = '\0';
    fread(src, 0, length, stream);
    fclose(stream);

    return buildProgram(context, device, src, length);
}
