cmake_minimum_required(VERSION 3.18)
project(cython_cmake_demo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")

execute_process(
        COMMAND ${Python_EXECUTABLE} -m cython --version
        RESULT_VARIABLE CYTHON_OK
        OUTPUT_VARIABLE CYTHON_VERSION
        ERROR_VARIABLE CYTHON_VERSION
)
message(STATUS "Using Cython: ${CYTHON_VERSION}")

set(MYMOD_PYX ${CMAKE_CURRENT_SOURCE_DIR}/src/mymod.pyx)
set(MYMOD_CPP ${CMAKE_CURRENT_BINARY_DIR}/mymod.cpp)

add_custom_command(
        OUTPUT ${MYMOD_CPP}
        COMMAND ${Python_EXECUTABLE} -m cython ${MYMOD_PYX} -3 -cplus -o ${MYMOD_CPP}
        DEPENDS ${MYMOD_PYX}
        COMMENT "Cythonizing ${MYMOD_PYX}"
)

# set(FREEIMAGE_INC_DIR /usr/include)
find_path(FREEIMAGE_INC_DIR NAMES FreeImage.h)
find_library(FREEIMAGE_LIB freeimage)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fpermissive")
link_directories(${FREEIMAGE_LIB})


set(NVOF_PUBLIC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/NvOFInterface)
set(NVOF_UTILS_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Common/Utils/NvOFUtils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Common/Utils/NvOFDataLoader.cpp
)
set(NVOF_UTILS_HDRS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Common/Utils/NvOFUtils.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Common/Utils/NvOFDataLoader.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Common/Utils/NvOFCmdParser.h
)
set(NVOF_UTILS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/Common/Utils)

set(NVOF_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Common/NvOFBase/NvOF.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Common/NvOFBase/NvOFCuda.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Common/Utils/NvOFUtilsCuda.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Common/Utils/kernel.cu
)
set(NVOF_HDRS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Common/NvOFBase/NvOF.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Common/NvOFBase/NvOFCuda.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Common/NvOFBase/NvOFDefines.h
        ${NVOF_PUBLIC_INCLUDE_DIR}/nvOpticalFlowCommon.h
        ${NVOF_PUBLIC_INCLUDE_DIR}/nvOpticalFlowCuda.h
)
source_group( "include" FILES ${NVOF_HDRS} ${NVOF_UTILS_HDRS})
source_group( "source" FILES ${MYMOD_CPP} ${NVOF_SOURCES} ${NVOF_UTILS_SOURCES})

set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_89,code=\"sm_89,compute_89\")
if ( CMAKE_COMPILER_IS_GNUCC )
    if(NOT "${CUDA_NVCC_FLAGS}" MATCHES "-std=c\\+\\+11" )
        list(APPEND CUDA_NVCC_FLAGS -std=c++11)
    endif()
endif()

find_package(CUDA)
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)

add_library(mymod MODULE
        ${MYMOD_CPP}
        ${NVOF_SOURCES}
        ${NVOF_UTILS_SOURCES}
        ${NVOF_HDRS}
        ${NVOF_HDRS}
        ${NVOF_UTILS_HDRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/mylib.cpp)
set_target_properties(mymod PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(mymod PUBLIC ${CUDA_INCLUDE_DIRS}
        ${FREEIMAGE_INC_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Common/NvOFBase
        ${NVOF_UTILS_INCLUDE_DIR}
        ${NVOF_PUBLIC_INCLUDE_DIR}
        ${Python_INCLUDE_DIRS}
        ${Python_NumPy_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(mymod
        ${CUDA_CUDA_LIBRARY}
        ${CMAKE_DL_LIBS}
        ${FREEIMAGE_LIB}
        ${Python_LIBRARIES})

#add_library(mymod MODULE ${MYMOD_CPP} ${CMAKE_CURRENT_SOURCE_DIR}/src/mylib.cpp)
#target_include_directories(mymod PRIVATE
#    ${Python_INCLUDE_DIRS}
#    ${Python_NumPy_INCLUDE_DIRS}
#    ${CMAKE_CURRENT_SOURCE_DIR}/src
#)
#
#target_link_libraries(mymod PRIVATE ${Python_LIBRARIES})
set_target_properties(mymod PROPERTIES PREFIX "")
#
