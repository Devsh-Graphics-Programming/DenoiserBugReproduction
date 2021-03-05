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

#pragma once

#ifndef APPLICATION_H
#define APPLICATION_H

#include <vector>

// DAR This version of the renderer only uses the CUDA Driver API!
// (CMake uses the CUDA_CUDA_LIBRARY which is nvcuda.lib. At runtime that loads nvcuda.dll from the driver.)
#include <cuda.h>
//#include <cuda_runtime.h>

#include <optix.h>

// OptiX 7 function table structure.
#include <optix_function_table.h>

#if defined(_WIN32)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif

#include <windows.h>
#endif

#include <iostream>
#include <map>
#include <string>

#include "../inc/Logger.h"

#define APP_EXIT_SUCCESS          0

#define APP_ERROR_UNKNOWN        -1
#define APP_ERROR_CREATE_WINDOW  -2
#define APP_ERROR_GLFW_INIT      -3
#define APP_ERROR_GLEW_INIT      -4
#define APP_ERROR_APP_INIT       -5

struct DeviceAttribute
{
  int maxThreadsPerBlock;
  int maxBlockDimX;
  int maxBlockDimY;
  int maxBlockDimZ;
  int maxGridDimX;
  int maxGridDimY;
  int maxGridDimZ;
  int maxSharedMemoryPerBlock;
  int sharedMemoryPerBlock;
  int totalConstantMemory;
  int warpSize;
  int maxPitch;
  int maxRegistersPerBlock;
  int registersPerBlock;
  int clockRate;
  int textureAlignment;
  int gpuOverlap;
  int multiprocessorCount;
  int kernelExecTimeout;
  int integrated;
  int canMapHostMemory;
  int computeMode;
  int maximumTexture1dWidth;
  int maximumTexture2dWidth;
  int maximumTexture2dHeight;
  int maximumTexture3dWidth;
  int maximumTexture3dHeight;
  int maximumTexture3dDepth;
  int maximumTexture2dLayeredWidth;
  int maximumTexture2dLayeredHeight;
  int maximumTexture2dLayeredLayers;
  int maximumTexture2dArrayWidth;
  int maximumTexture2dArrayHeight;
  int maximumTexture2dArrayNumslices;
  int surfaceAlignment;
  int concurrentKernels;
  int eccEnabled;
  int pciBusId;
  int pciDeviceId;
  int tccDriver;
  int memoryClockRate;
  int globalMemoryBusWidth;
  int l2CacheSize;
  int maxThreadsPerMultiprocessor;
  int asyncEngineCount;
  int unifiedAddressing;
  int maximumTexture1dLayeredWidth;
  int maximumTexture1dLayeredLayers;
  int canTex2dGather;
  int maximumTexture2dGatherWidth;
  int maximumTexture2dGatherHeight;
  int maximumTexture3dWidthAlternate;
  int maximumTexture3dHeightAlternate;
  int maximumTexture3dDepthAlternate;
  int pciDomainId;
  int texturePitchAlignment;
  int maximumTexturecubemapWidth;
  int maximumTexturecubemapLayeredWidth;
  int maximumTexturecubemapLayeredLayers;
  int maximumSurface1dWidth;
  int maximumSurface2dWidth;
  int maximumSurface2dHeight;
  int maximumSurface3dWidth;
  int maximumSurface3dHeight;
  int maximumSurface3dDepth;
  int maximumSurface1dLayeredWidth;
  int maximumSurface1dLayeredLayers;
  int maximumSurface2dLayeredWidth;
  int maximumSurface2dLayeredHeight;
  int maximumSurface2dLayeredLayers;
  int maximumSurfacecubemapWidth;
  int maximumSurfacecubemapLayeredWidth;
  int maximumSurfacecubemapLayeredLayers;
  int maximumTexture1dLinearWidth;
  int maximumTexture2dLinearWidth;
  int maximumTexture2dLinearHeight;
  int maximumTexture2dLinearPitch;
  int maximumTexture2dMipmappedWidth;
  int maximumTexture2dMipmappedHeight;
  int computeCapabilityMajor;
  int computeCapabilityMinor;
  int maximumTexture1dMipmappedWidth;
  int streamPrioritiesSupported;
  int globalL1CacheSupported;
  int localL1CacheSupported;
  int maxSharedMemoryPerMultiprocessor;
  int maxRegistersPerMultiprocessor;
  int managedMemory;
  int multiGpuBoard;
  int multiGpuBoardGroupId;
  int hostNativeAtomicSupported;
  int singleToDoublePrecisionPerfRatio;
  int pageableMemoryAccess;
  int concurrentManagedAccess;
  int computePreemptionSupported;
  int canUseHostPointerForRegisteredMem;
  int canUseStreamMemOps;
  int canUse64BitStreamMemOps;
  int canUseStreamWaitValueNor;
  int cooperativeLaunch;
  int cooperativeMultiDeviceLaunch;
  int maxSharedMemoryPerBlockOptin;
  int canFlushRemoteWrites;
  int hostRegisterSupported;
  int pageableMemoryAccessUsesHostPageTables;
  int directManagedMemAccessFromHost;
};

class Application
{
public:
  Application();
  ~Application();

  bool initOptiX();

private:

  void getSystemInformation();

  OptixResult initOptiXFunctionTable();

  std::vector<DeviceAttribute> m_deviceAttributes;

	// CUDA native types are prefixed with "cuda".
  CUcontext m_cudaContext;
  CUstream  m_cudaStream;

  // The handle for the registered OpenGL PBO when using interop.
  CUgraphicsResource m_cudaGraphicsResource;

  // All others are OptiX types.
  OptixFunctionTable m_api;
  OptixDeviceContext m_context;

  Logger m_logger;

  // Denoiser:
  OptixDenoiser       m_denoiser;
  OptixDenoiserSizes  m_sizesDenoiser;
  OptixDenoiserParams m_paramsDenoiser;
  CUdeviceptr         m_d_stateDenoiser;
  CUdeviceptr         m_d_scratchDenoiser;
  CUdeviceptr         m_d_denoisedBuffer;
  unsigned int        m_numInputLayers;
  // Helper variable to abstract the OptiX 7.0.0 OptixDenoiserSizes.recommendedScratchSizeInBytes and OptiX 7.1.0 OptixDenoiserSizes.withoutOverlapScratchSizeInBytes 
  size_t              m_scratchSizeInBytes;

  OptixImage2D m_inputImage[3]; // 0 = beauty, 1 = albedo, 2 = normal
  OptixImage2D m_outputImage;

  OptixTraversableHandle m_root;  // Scene root
  CUdeviceptr            m_d_ias; // Scene root's IAS (instance acceleration structure).

  // API Reference sidenote on optixLaunch (doesn't apply for this example):
  // Concurrent launches to multiple streams require separate OptixPipeline objects. 
  OptixPipeline m_pipeline;

  std::vector<OptixInstance> m_instances;

  // The Shader Binding Table and data.
  OptixShaderBindingTable m_sbt;

  CUdeviceptr m_d_sbtRecordRaygeneration;
  CUdeviceptr m_d_sbtRecordException;
  CUdeviceptr m_d_sbtRecordMiss;

  CUdeviceptr m_d_sbtRecordCallables;
};

#endif // APPLICATION_H

