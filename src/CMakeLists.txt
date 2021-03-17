# GLI library for DDS inputs
set(_OLD_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(_OLD_BUILD_STATIC_LIBS ${BUILD_STATIC_LIBS})
set(_OLD_BUILD_TESTING ${BUILD_TESTING})
set(BUILD_SHARED_LIBS OFF)
set(BUILD_STATIC_LIBS OFF)
set(BUILD_TESTING OFF)
add_subdirectory(gli gli EXCLUDE_FROM_ALL)
set(BUILD_SHARED_LIBS ${_OLD_BUILD_SHARED_LIBS})
set(BUILD_STATIC_LIBS ${_OLD_BUILD_STATIC_LIBS})
set(BUILD_TESTING ${_OLD_BUILD_TESTING})

set(EXECUTABLE_NAME "denoiserBugReproductionApp")

set(DBR_EXTERNAL_SOURCES
	
)

set(DBR_EXTERNAL_HEADERS
	"config/BuildConfigOptions.h"
	"nvidia/CheckMacros.h"
	"nvidia/MyAssert.h"
)

set(DBR_SOURCES
	${DBR_EXTERNAL_SOURCES}
)

set(DBR_HEADERS
	${DBR_EXTERNAL_HEADERS}
)

set(DBR_GLOBAL_SOURCES
	"main.cpp"
	"${DBR_SOURCES}"
	"${DBR_HEADERS}"
)

add_executable(${EXECUTABLE_NAME} ${DBR_GLOBAL_SOURCES})

add_dependencies(${EXECUTABLE_NAME} gli)
target_include_directories(${EXECUTABLE_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/gli)

source_group( "sources" FILES ${DBR_SOURCES} )
source_group( "headers" FILES ${DBR_HEADERS} )

set_property(TARGET ${EXECUTABLE_NAME} PROPERTY
             MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
			 
if (OptiX72_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX72_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "No OptiX SDK 7.2 found.")
endif()			 
		
set(DBR_OPTIX_INCLUDE 
	"${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
	"${OPTIX_INCLUDE_DIR}"
)

set(DBR_GLOBAL_INCLUDE
	"${DBR_OPTIX_INCLUDE}"
)
			 		
target_include_directories(${EXECUTABLE_NAME}
	PRIVATE ${DBR_GLOBAL_INCLUDE}
)

message("CUDA_CUDA_LIBRARY = " "${CUDA_CUDA_LIBRARY}")

target_link_libraries(${EXECUTABLE_NAME} PUBLIC
	${CUDA_CUDA_LIBRARY}
	gli
)

DBR_adjust_flags() # macro defined in root CMakeLists
DBR_adjust_definitions() # macro defined in root CMakeLists