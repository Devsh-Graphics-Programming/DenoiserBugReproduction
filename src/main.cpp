#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <map>
#include <cassert>

#include <cuda.h>
#include <optix.h>
#include <cudaGL.h>

constexpr uint32_t overlap = 64;
//constexpr uint32_t tileWidth = 1920/2, tileHeight = 1080/2;
constexpr uint32_t tileWidth = 1024, tileHeight = 1024;
constexpr uint32_t tileWidthWithOverlap = tileWidth + overlap * 2;
constexpr uint32_t tileHeightWithOverlap = tileHeight + overlap * 2;
constexpr uint32_t outputDimensions[] = { tileWidth , tileHeight };

void DBROptixDefaultCallback(unsigned int level, const char* tag, const char* message, void* cbdata)
{
	uint32_t contextID = reinterpret_cast<const uint32_t&>(cbdata);
	printf("OptiX Context:%d [%s]: %s\n", contextID, tag, message);
}

enum E_IMAGE_INPUT : uint32_t
{
	EII_COLOR,
	EII_ALBEDO,
	EII_NORMAL,
	EII_COUNT
};

struct DenoiserToUse
{
	OptixDenoiser denoiser = nullptr;
	size_t stateOffset = 0u;
	size_t stateSize = 0u;
	size_t scratchSize = 0u;
};

int main()
{
	bool status = true;

	uint32_t maxResolution[2] = { 0,0 }; // TODO load EXRs images

	/*
		...
	*/

	constexpr size_t MaxSLI = 4;

	struct Cuda
	{
		uint32_t foundDeviceCount = 0u;
		CUdevice devices[MaxSLI] = {};
		CUstream stream[MaxSLI];
	} cuda;

	// find device
	cuGLGetDevices_v2(&cuda.foundDeviceCount, cuda.devices, MaxSLI, CU_GL_DEVICE_LIST_ALL);

	// create context
	CUcontext contexts[MaxSLI] = {};
	bool ownContext[MaxSLI] = {};
	uint32_t suitableDevices = 0u;
	for (uint32_t i = 0u; i < cuda.foundDeviceCount; i++)
	{
		CUresult cudaResult = cuCtxCreate_v2(contexts + suitableDevices, CU_CTX_SCHED_YIELD | CU_CTX_MAP_HOST | CU_CTX_LMEM_RESIZE_TO_MAX, cuda.devices[suitableDevices]);

		if (cudaResult != CUDA_SUCCESS)
			continue;

		uint32_t version = 0u;

		cuCtxGetApiVersion(contexts[suitableDevices], &version);

		if (version < 3020)
		{
			cuCtxDestroy_v2(contexts[suitableDevices]);
			continue;
		}

		cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_L1);
		ownContext[suitableDevices++] = true;
	}

	if (!suitableDevices)
	{
		status = false;
		assert(status);
	}

	// cuda streams and function table (?)

	assert(_contextCount <= MaxSLI);

	// Initialize the OptiX API, loading all API entry points 
	//optixInit(); // do I need this? (68 Optix/Manger.cpp)

	for (uint32_t i = 0u; i < MaxSLI; i++)
	{
		if (cuStreamCreate(cuda.stream + i, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS)
		{
			contexts[i] = nullptr;
			ownContext[i] = false;
			cuda.stream[i] = false;
			continue;
		}
	}

	// Adding Optix Headers (?)  (83 Optix/Manager.cpp)

	/*
		Fetching Optix Stream
	*/

	auto m_cudaStream = cuda.stream[0];
	{
		if (!m_cudaStream)
			status = false;

		assert(status); // "Could not obtain CUDA stream!"
	}

	/*
		Init Optix Context
	*/

	OptixDeviceContext optixContext;
	if (optixDeviceContextCreate(contexts[0], {}, &optixContext) != OPTIX_SUCCESS)
		status = false;

	assert(status);

	optixDeviceContextSetLogCallback(optixContext, DBROptixDefaultCallback, reinterpret_cast<void*>(contexts[0]), 3);

	/*
		Creating Denoisers
	*/

	constexpr auto forcedOptiXFormat = OPTIX_PIXEL_FORMAT_HALF3;
	constexpr auto forcedOptiXFormatPixelStride = 6u;

	auto createDenoiser = [&](const OptixDenoiserOptions* options, OptixDenoiserModelKind model = OPTIX_DENOISER_MODEL_KIND_HDR, void* modelData = nullptr, size_t modelDataSizeInBytes = 0ull) -> OptixDenoiser
	{
		if (!options)
			return nullptr;

		OptixDenoiser denoiser = nullptr;
		if (optixDenoiserCreate(optixContext, options, &denoiser) != OPTIX_SUCCESS || !denoiser)
			return nullptr;

		if (optixDenoiserSetModel(denoiser, model, modelData, modelDataSizeInBytes) != OPTIX_SUCCESS)
			return nullptr;

		return denoiser;
	};
	
	DenoiserToUse denoisers[EII_COUNT];
	{
		OptixDenoiserOptions opts = { OPTIX_DENOISER_INPUT_RGB };

		denoisers[EII_COLOR].denoiser = createDenoiser(&opts);
		if (!denoisers[EII_COLOR].denoiser)
			status = false;

		assert(status); // "Could not create Optix Color Denoiser!"

		opts.inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO;

		denoisers[EII_ALBEDO].denoiser = createDenoiser(&opts);
		if (!denoisers[EII_ALBEDO].denoiser)
			status = false;

		assert(status); // "Could not create Optix Color-Albedo Denoiser!"

		opts.inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL;

		denoisers[EII_NORMAL].denoiser = createDenoiser(&opts);
		if (!denoisers[EII_NORMAL].denoiser)
			status = false;

		assert(status); // "Could not create Optix Color-Albedo-Normal Denoiser!"
	}


	/*
		Compute memory resources for denoiser
	*/

	size_t denoiserStateBufferSize = 0ull;
	size_t scratchBufferSize = 0ull;
	size_t pixelBufferSize = 0ull;

	for (uint32_t i = 0u; i < EII_COUNT; i++)
	{
		auto& denoiserToUse = denoisers[i];

		OptixDenoiserSizes denoiserMemReqs;

		if (optixDenoiserComputeMemoryResources(denoiserToUse.denoiser, outputDimensions[0], outputDimensions[1], &denoiserMemReqs) != OPTIX_SUCCESS)
			status = false;

		assert(status); // Failed to compute Memory Requirements!

		constexpr size_t texelByteSize = 12; // EF_R32G32B32A32_SFLOAT

		denoiserToUse.stateOffset = denoiserStateBufferSize;
		denoiserStateBufferSize += denoiserToUse.stateSize = denoiserMemReqs.stateSizeInBytes;
		scratchBufferSize = std::max<size_t>(scratchBufferSize, denoisers[i].scratchSize = denoiserMemReqs.withOverlapScratchSizeInBytes);
		pixelBufferSize = std::max<size_t>(pixelBufferSize, std::max<size_t>(texelByteSize, (i + 1u) * forcedOptiXFormatPixelStride) * maxResolution[0] * maxResolution[1]);
	}

	std::string message = "Total VRAM consumption for Denoiser algorithm: ";
	std::cout << message + std::to_string(denoiserStateBufferSize + scratchBufferSize + pixelBufferSize);

	if (pixelBufferSize == 0u)
		status = false;

	assert(status); // No input files at all!

	CUdeviceptr denoiserState;
	cuMemAlloc(&denoiserState, denoiserStateBufferSize /* + IntensityValuesSize ? */);

	CUdeviceptr temporaryPixelBuffer;
	cuMemAlloc(&temporaryPixelBuffer, pixelBufferSize);
}