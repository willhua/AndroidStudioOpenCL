//
// Created by lisan on 2018/8/12.
//

#ifndef ANDROIDSTUDIOOPENCL_LOGUTILS_H
#define ANDROIDSTUDIOOPENCL_LOGUTILS_H


#include <android/log.h>

#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "lyh", __VA_ARGS__);
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "lyh", __VA_ARGS__);

#endif //ANDROIDSTUDIOOPENCL_LOGUTILS_H
