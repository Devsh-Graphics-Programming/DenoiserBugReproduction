/* 
 * Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "../inc/Application.h"
#include "../inc/CheckMacros.h"

#ifdef _WIN32
// The cfgmgr32 header is necessary for interrogating driver information in the registry.
#include <cfgmgr32.h>
// For convenience the library is also linked in automatically using the #pragma command.
#pragma comment(lib, "Cfgmgr32.lib")
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string.h>
#include <vector>

#include <time.h>

#include "../inc/MyAssert.h"

#ifdef _WIN32
// Code based on helper function in optix_stubs.h
static void* optixLoadWindowsDll(void)
{
  const char* optixDllName = "nvoptix.dll";
  void* handle = NULL;

  // Get the size of the path first, then allocate
  unsigned int size = GetSystemDirectoryA(NULL, 0);
  if (size == 0)
  {
    // Couldn't get the system path size, so bail
    return NULL;
  }

  size_t pathSize = size + 1 + strlen(optixDllName);
  char*  systemPath = (char*) malloc(pathSize);

  if (GetSystemDirectoryA(systemPath, size) != size - 1)
  {
    // Something went wrong
    free(systemPath);
    return NULL;
  }

  strcat(systemPath, "\\");
  strcat(systemPath, optixDllName);

  handle = LoadLibraryA(systemPath);

  free(systemPath);

  if (handle)
  {
    return handle;
  }

  // If we didn't find it, go looking in the register store.  Since nvoptix.dll doesn't
  // have its own registry entry, we are going to look for the OpenGL driver which lives
  // next to nvoptix.dll. 0 (null) will be returned if any errors occured.

  static const char* deviceInstanceIdentifiersGUID = "{4d36e968-e325-11ce-bfc1-08002be10318}";
  const ULONG        flags = CM_GETIDLIST_FILTER_CLASS | CM_GETIDLIST_FILTER_PRESENT;
  ULONG              deviceListSize = 0;

  if (CM_Get_Device_ID_List_SizeA(&deviceListSize, deviceInstanceIdentifiersGUID, flags) != CR_SUCCESS)
  {
    return NULL;
  }

  char* deviceNames = (char*) malloc(deviceListSize);

  if (CM_Get_Device_ID_ListA(deviceInstanceIdentifiersGUID, deviceNames, deviceListSize, flags))
  {
    free(deviceNames);
    return NULL;
  }

  DEVINST devID = 0;

  // Continue to the next device if errors are encountered.
  for (char* deviceName = deviceNames; *deviceName; deviceName += strlen(deviceName) + 1)
  {
    if (CM_Locate_DevNodeA(&devID, deviceName, CM_LOCATE_DEVNODE_NORMAL) != CR_SUCCESS)
    {
      continue;
    }

    HKEY regKey = 0;
    if (CM_Open_DevNode_Key(devID, KEY_QUERY_VALUE, 0, RegDisposition_OpenExisting, &regKey, CM_REGISTRY_SOFTWARE) != CR_SUCCESS)
    {
      continue;
    }

    const char* valueName = "OpenGLDriverName";
    DWORD       valueSize = 0;

    LSTATUS     ret = RegQueryValueExA(regKey, valueName, NULL, NULL, NULL, &valueSize);
    if (ret != ERROR_SUCCESS)
    {
      RegCloseKey(regKey);
      continue;
    }

    char* regValue = (char*) malloc(valueSize);
    ret = RegQueryValueExA(regKey, valueName, NULL, NULL, (LPBYTE) regValue, &valueSize);
    if (ret != ERROR_SUCCESS)
    {
      free(regValue);
      RegCloseKey(regKey);
      continue;
    }

    // Strip the OpenGL driver dll name from the string then create a new string with
    // the path and the nvoptix.dll name
    for (int i = valueSize - 1; i >= 0 && regValue[i] != '\\'; --i)
    {
      regValue[i] = '\0';
    }

    size_t newPathSize = strlen(regValue) + strlen(optixDllName) + 1;
    char*  dllPath = (char*) malloc(newPathSize);
    strcpy(dllPath, regValue);
    strcat(dllPath, optixDllName);

    free(regValue);
    RegCloseKey(regKey);

    handle = LoadLibraryA((LPCSTR) dllPath);
    free(dllPath);

    if (handle)
    {
      break;
    }
  }

  free(deviceNames);

  return handle;
}
#endif

OptixResult Application::initOptiXFunctionTable()
{
#ifdef _WIN32
    void* handle = optixLoadWindowsDll();
    if (!handle)
    {
        return OPTIX_ERROR_LIBRARY_NOT_FOUND;
    }

    void* symbol = reinterpret_cast<void*>(GetProcAddress((HMODULE)handle, "optixQueryFunctionTable"));
    if (!symbol)
    {
        return OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND;
    }
#else
    void* handle = dlopen("libnvoptix.so.1", RTLD_NOW);
    if (!handle)
    {
        return OPTIX_ERROR_LIBRARY_NOT_FOUND;
    }

    void* symbol = dlsym(handle, "optixQueryFunctionTable");
    if (!symbol)
    {
        return OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND;
    }
#endif

    OptixQueryFunctionTable_t* optixQueryFunctionTable = reinterpret_cast<OptixQueryFunctionTable_t*>(symbol);

    return optixQueryFunctionTable(OPTIX_ABI_VERSION, 0, 0, 0, &m_api, sizeof(OptixFunctionTable));
}

void Application::getSystemInformation()
{
  int versionDriver = 0;
  CU_CHECK( cuDriverGetVersion(&versionDriver) ); 
  
  // The version is returned as (1000 * major + 10 * minor).
  int major =  versionDriver / 1000;
  int minor = (versionDriver - major * 1000) / 10;
  std::cout << "Driver Version  = " << major << "." << minor << '\n';
  
  int countDevices = 0;
  CU_CHECK( cuDeviceGetCount(&countDevices) );
  std::cout << "Device Count    = " << countDevices << '\n';

  char name[1024];
  name[1023] = 0;

  for (CUdevice device = 0; device < countDevices; ++device)
  {
    CU_CHECK( cuDeviceGetName(name, 1023, device) );
    std::cout << "Device " << device << ": " << name << '\n';

    DeviceAttribute attr = {};

    CU_CHECK( cuDeviceGetAttribute(&attr.maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maxBlockDimX, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maxBlockDimY, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maxBlockDimZ, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maxGridDimX, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maxGridDimY, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maxGridDimZ, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maxSharedMemoryPerBlock, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.sharedMemoryPerBlock, CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.totalConstantMemory, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maxPitch, CU_DEVICE_ATTRIBUTE_MAX_PITCH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maxRegistersPerBlock, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.registersPerBlock, CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.textureAlignment, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.gpuOverlap, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.multiprocessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.kernelExecTimeout, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.canMapHostMemory, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.computeMode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture1dWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture3dWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture3dHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture3dDepth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dLayeredWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dLayeredHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dLayeredLayers, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dArrayWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dArrayHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dArrayNumslices, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.surfaceAlignment, CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.concurrentKernels, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.eccEnabled, CU_DEVICE_ATTRIBUTE_ECC_ENABLED, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.pciBusId, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.pciDeviceId, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.tccDriver, CU_DEVICE_ATTRIBUTE_TCC_DRIVER, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.memoryClockRate, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.globalMemoryBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.l2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maxThreadsPerMultiprocessor, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.asyncEngineCount, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.unifiedAddressing, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture1dLayeredWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture1dLayeredLayers, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.canTex2dGather, CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dGatherWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dGatherHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture3dWidthAlternate, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture3dHeightAlternate, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture3dDepthAlternate, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.pciDomainId, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.texturePitchAlignment, CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexturecubemapWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexturecubemapLayeredWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexturecubemapLayeredLayers, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface1dWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface2dWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface2dHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface3dWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface3dHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface3dDepth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface1dLayeredWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface1dLayeredLayers, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface2dLayeredWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface2dLayeredHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurface2dLayeredLayers, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurfacecubemapWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurfacecubemapLayeredWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumSurfacecubemapLayeredLayers, CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture1dLinearWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dLinearWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dLinearHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dLinearPitch, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dMipmappedWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture2dMipmappedHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maximumTexture1dMipmappedWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.streamPrioritiesSupported, CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.globalL1CacheSupported, CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.localL1CacheSupported, CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maxSharedMemoryPerMultiprocessor, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maxRegistersPerMultiprocessor, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.managedMemory, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.multiGpuBoard, CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.multiGpuBoardGroupId, CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.hostNativeAtomicSupported, CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.singleToDoublePrecisionPerfRatio, CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.pageableMemoryAccess, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.concurrentManagedAccess, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.computePreemptionSupported, CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.canUseHostPointerForRegisteredMem, CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.canUseStreamMemOps, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.canUse64BitStreamMemOps, CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.canUseStreamWaitValueNor, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.cooperativeLaunch, CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.cooperativeMultiDeviceLaunch, CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.maxSharedMemoryPerBlockOptin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.canFlushRemoteWrites, CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.hostRegisterSupported, CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.pageableMemoryAccessUsesHostPageTables, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, device) );
    CU_CHECK( cuDeviceGetAttribute(&attr.directManagedMemAccessFromHost, CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST, device) );

    m_deviceAttributes.push_back(attr);
  }
}

bool Application::initOptiX()
{
  CUresult cuRes = cuInit(0); // Initialize CUDA driver API.
  if (cuRes != CUDA_SUCCESS)
  {
    std::cerr << "ERROR: initOptiX() cuInit() failed: " << cuRes << '\n';
    return false;
  }

  getSystemInformation(); // Get device attributes of all found devices. Fills m_deviceAttributes.

  CUdevice device = 0;

  cuRes = cuCtxCreate(&m_cudaContext, CU_CTX_SCHED_SPIN, device); // DEBUG What is the best CU_CTX_SCHED_* setting here.
  if (cuRes != CUDA_SUCCESS)
  {
    std::cerr << "ERROR: initOptiX() cuCtxCreate() failed: " << cuRes << '\n';
    return false;
  }

  // PERF Use CU_STREAM_NON_BLOCKING if there is any work running in parallel on multiple streams.
  cuRes = cuStreamCreate(&m_cudaStream, CU_STREAM_DEFAULT);
  if (cuRes != CUDA_SUCCESS)
  {
    std::cerr << "ERROR: initOptiX() cuStreamCreate() failed: " << cuRes << '\n';
    return false;
  }

  OptixResult res = initOptiXFunctionTable();
  if (res != OPTIX_SUCCESS)
  {
    std::cerr << "ERROR: initOptiX() initOptiXFunctionTable() failed: " << res << '\n';
    return false;
  }

  OptixDeviceContextOptions options = {};

  options.logCallbackFunction = &Logger::callback;
  options.logCallbackData     = &m_logger;
  options.logCallbackLevel    = 4;

  res = m_api.optixDeviceContextCreate(m_cudaContext, &options, &m_context);
  if (res != OPTIX_SUCCESS)
  {
    std::cerr << "ERROR: initOptiX() optixDeviceContextCreate() failed: " << res << '\n';
    return false;
  }

  //initRenderer(); // Initialize all the rest.

  //initDenoiser();

  return true;
}