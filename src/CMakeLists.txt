set(EXECUTABLE_NAME "denoiserBugReproductionApp")

DBR_adjust_flags() # macro defined in root CMakeLists
DBR_adjust_definitions() # macro defined in root CMakeLists

set(DBR_SOURCES
	"dbr/Application.cpp"
)

set(INC_HEADERS
	"inc/Application.h"
	"inc/CheckMacros.h"
	"inc/MyAssert.h"
	"inc/Logger.h"
)

set(DBR_GLOBAL_SOURCES
	"dbr/main.cpp"
	"${DBR_SOURCES}"
	"${INC_HEADERS}"
)

add_executable(${EXECUTABLE_NAME} ${DBR_GLOBAL_SOURCES})

message("CUDA_CUDA_LIBRARY = " "${CUDA_CUDA_LIBRARY}")
target_link_libraries(${EXECUTABLE_NAME} ${CUDA_CUDA_LIBRARY})

source_group( "sources" FILES ${DBR_SOURCES} )
source_group( "headers" FILES ${INC_HEADERS} )

set_property(TARGET ${EXECUTABLE_NAME} PROPERTY
             MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
			 
# Make the build work with either OptiX 7.2.0, 7.1.0 or 7.0.0 releases
if (OptiX72_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX72_INCLUDE_DIR}")
elseif(OptiX71_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX71_INCLUDE_DIR}")
elseif(OptiX7_FOUND)
  set(OPTIX_INCLUDE_DIR "${OPTIX7_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "No OptiX SDK 7.x found.")
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
					 
find_package(OpenGL REQUIRED)
target_link_libraries(${EXECUTABLE_NAME} ${OPENGL_LIBRARIES})