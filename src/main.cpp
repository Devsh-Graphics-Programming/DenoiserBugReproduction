#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <map>
#include <cassert>

#include <cuda.h>
#include <optix.h>
#include <optix_stubs.h>

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

int main()
{
	bool status = true;

	uint32_t maxResolution[2] = { 0,0 }; // TODO load EXRs images

	/*
		...
	*/
	CUdevice device;
	cuCtxGetDevice(&device);

	// create context
	CUcontext context;
	cuCtxCreate_v2(&context,CU_CTX_SCHED_YIELD | CU_CTX_MAP_HOST | CU_CTX_LMEM_RESIZE_TO_MAX,device);
	{
		uint32_t version = 0u;

		cuCtxGetApiVersion(context, &version);

		if (version < 3020)
		{
			status = false;
			assert(status);
			return 1;
		}

		cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_L1);
	}

	optixInit();

	CUstream stream;
	cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING); // TODO: check/wrap all CUDA calls in check for CUDA_SUCCESS

	/*
		Init Optix Context
	*/

	OptixDeviceContext optixContext;
	if (optixDeviceContextCreate(context, {}, &optixContext) != OPTIX_SUCCESS)
		status = false;

	assert(status);

	optixDeviceContextSetLogCallback(optixContext, DBROptixDefaultCallback, reinterpret_cast<void*>(context), 3);

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

	constexpr uint32_t kInputBufferCount = 3u;
	OptixDenoiserOptions opts = { OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL };
	OptixDenoiser denoiser = createDenoiser(&opts);
	if (!denoiser)
		status = false;
	assert(status); // "Could not create Optix Color-Albedo-Normal Denoiser!

	/*
		Compute memory resources for denoiser
	*/

	size_t denoiserStateBufferSize = 0ull;
	size_t scratchBufferSize = 0ull;
	size_t singleInputBufferSize = 0ull;
	{
		OptixDenoiserSizes denoiserMemReqs;

		if (optixDenoiserComputeMemoryResources(denoiser, outputDimensions[0], outputDimensions[1], &denoiserMemReqs) != OPTIX_SUCCESS)
			status = false;
		assert(status); // Failed to compute Memory Requirements!

		denoiserStateBufferSize = denoiserMemReqs.stateSizeInBytes;
		scratchBufferSize = denoiserMemReqs.withOverlapScratchSizeInBytes;
		singleInputBufferSize = forcedOptiXFormatPixelStride * maxResolution[0] * maxResolution[1];
	}
	const size_t pixelBufferSize = singleInputBufferSize*kInputBufferCount;

	std::string message = "Total VRAM consumption for Denoiser algorithm: ";
	std::cout << message + std::to_string(denoiserStateBufferSize + scratchBufferSize + pixelBufferSize);

	if (pixelBufferSize == 0u)
		status = false;

	assert(status); // No input files at all!

	CUdeviceptr denoiserState;
	cuMemAlloc(&denoiserState, denoiserStateBufferSize);

	CUdeviceptr	imageIntensity;
	cuMemAlloc(&imageIntensity, sizeof(float));

	CUdeviceptr inputPixelBuffer;
	cuMemAlloc(&inputPixelBuffer, pixelBufferSize);
	CUdeviceptr inputPixelBuffers[3] =
	{
		inputPixelBuffer,
		inputPixelBuffer+singleInputBufferSize,
		inputPixelBuffer+singleInputBufferSize*2u
	};

	FILE* outputFile = fopen("output.dds","w");
	const char* hardcodedInputs[3] =
	{
		"",
		"",
		""
	};
	constexpr uint32_t ddsDataOffset = 69u;
	for (auto i=0u; i<kInputBufferCount; i++)
	{
		auto file = fopen(hardcodedInputs[i],"r");
		// copy over the header
		char header[ddsDataOffset];
		fread(header,1u,ddsDataOffset,file);
		if (i==0u)
			fwrite(header,1u,ddsDataOffset,outputFile);
		// quick and dirty load
		{
			void* tmp = malloc(singleInputBufferSize);
			fread(tmp,1u,singleInputBufferSize,file);
			cuMemcpyHtoD_v2(inputPixelBuffers[i],tmp,singleInputBufferSize);
			free(tmp);
		}
		fclose(file);
	}

	// TODO: denoise

	// TODO: write out the output with fwrite
	fclose(outputFile);
}