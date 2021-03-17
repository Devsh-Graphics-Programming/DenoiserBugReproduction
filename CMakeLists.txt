cmake_minimum_required(VERSION 3.16)

project(denoiserBugReproduction LANGUAGES CXX C)

# CUDA test
include(CheckLanguage)
check_language(CUDA)

if(NOT CMAKE_CUDA_COMPILER)
	message(WARNING "CUDA not found.")
else()
	enable_language(CUDA)
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;")

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type: Debug or Release" FORCE)
endif()

macro(DBR_adjust_flags)
	option(DBR_SANITIZE_ADDRESS OFF)

	if (DBR_SANITIZE_ADDRESS)
		set(DBR_MSVC_FLAG_SANITIZE_ADDRESS "/fsanitize=address")
		message(WARNING "Due to https://github.com/google/sanitizers/issues/328 expect not to be able to use the Debug target!")
		set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /wholearchive:clang_rt.asan_dbg-x86_64.lib /wholearchive:clang_rt.asan_cxx_dbg-x86_64.lib")
		set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} /wholearchive:clang_rt.asan-x86_64.lib /wholearchive:clang_rt.asan_cxx-x86_64.lib")
	endif()
	
	# debug
	string(REPLACE "/W3" "/W0" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zc:__cplusplus /MP /ZI /fp:fast /Zc:wchar_t /INCREMENTAL ${DBR_MSVC_SANITIZE_ADDRESS}" )
	string(REPLACE "/W3" "/W0" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
	set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MP /ZI /fp:fast /Zc:wchar_t /INCREMENTAL ${DBR_MSVC_SANITIZE_ADDRESS}")
	
	# release
	string(REPLACE "/GS" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}") # for some reason simply replacing /GS -> /GS- doesn't work... so it vanishes here and appears a few lines below!
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /GL /Zc:__cplusplus /MP /Gy- /Zc:wchar_t /sdl- /GF /GS- /fp:fast ${DBR_MSVC_SANITIZE_ADDRESS}")
	string(REPLACE "/GS" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
	set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /GL /MP /Gy- /Zc:wchar_t /sdl- /GF /GS- /fp:fast ${DBR_MSVC_SANITIZE_ADDRESS}")
	
	# relWithDebInfo
	string(REPLACE "/GS" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /GL /Zc:__cplusplus /Zc:wchar_t /MP /Gy /Zi /sdl- /Oy- /fp:fast ${DBR_MSVC_SANITIZE_ADDRESS}")
	string(REPLACE "/GS" "" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
	set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} /GL /MP /Gy /Zc:wchar_t /Zi /sdl- /Oy- /fp:fast ${DBR_MSVC_SANITIZE_ADDRESS}")
	
	#reason for INCREMENTAL:NO: https://docs.microsoft.com/en-us/cpp/build/reference/ltcg-link-time-code-generation?view=vs-2019 /LTCG is not valid for use with /INCREMENTAL.
	set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} /INCREMENTAL:NO /LTCG:incremental")
endmacro()

macro(DBR_adjust_definitions)
	if(WIN32)
		add_definitions(
			-DWIN32
			-D__GNUWIN32__
			-D_CRT_SECURE_NO_DEPRECATE
			-DNOMINMAX
		)
	endif()
endmacro()

find_package(OptiX72 REQUIRED)
find_package(CUDA 10.0 REQUIRED)

set(DBR_ROOT ${CMAKE_SOURCE_DIR})
configure_file("${CMAKE_SOURCE_DIR}/cmake/config/BuildConfigOptions.h.in" "${CMAKE_SOURCE_DIR}/src/config/BuildConfigOptions.h")
add_subdirectory(src/)