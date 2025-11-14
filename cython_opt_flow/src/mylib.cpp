#include "mylib.h"
#include <fstream>
#include <iostream>
#include <memory>

#include <cuda.h>

#include "NvOFCuda.h"
#include "NvOFDataLoader.h"
#include "NvOFUtils.h"
#include "NvOFCmdParser.h"

double square(double x)
{
    std::cout << "accessing c++ code" << std::endl;
    // int gpuId = 0;
    // int nGpu = 0;
    // CUDA_DRVAPI_CALL(cuInit(0));
    // CUDA_DRVAPI_CALL(cuDeviceGetCount(&nGpu));
    // if (gpuId < 0 || gpuId >= nGpu)
    // {
    //     std::cout << "GPU ordinal out of range. Should be with in [" << 0 << ", " << nGpu - 1 << "]" << std::endl;
    //     return 1;
    // }
    // CUstream   inputStream = nullptr;
    // CUstream   outputStream = nullptr;
    // CUdevice cuDevice = 0;
    // CUDA_DRVAPI_CALL(cuDeviceGet(&cuDevice, gpuId));
    // char szDeviceName[80];
    // CUDA_DRVAPI_CALL(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
    // std::cout << "GPU in use: " << szDeviceName << std::endl;
    // CUcontext cuContext = nullptr;
    // CUDA_DRVAPI_CALL(cuCtxCreate(&cuContext, nullptr, 0, cuDevice));

    return 0.0;
}
