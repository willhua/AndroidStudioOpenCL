//
// Created by lisan on 2018/8/12.
//

#ifndef ANDROIDSTUDIOOPENCL_OPENCLUTILS_H
#define ANDROIDSTUDIOOPENCL_OPENCLUTILS_H

#include "cl.h"
#include "LogUtils.h"

cl_device_id createDevice(cl_device_type type);
cl_program buildProgram(cl_context, cl_device_id, const char *, size_t);
cl_program buildProgramFromFile(cl_context, cl_device_id, const char *);



#endif //ANDROIDSTUDIOOPENCL_OPENCLUTILS_H
