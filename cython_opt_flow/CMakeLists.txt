cmake_minimum_required(VERSION 3.18)
project(cython_cmake_demo LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)

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
    COMMAND ${Python_EXECUTABLE} -m cython ${MYMOD_PYX} -3 -o ${MYMOD_CPP}
    DEPENDS ${MYMOD_PYX}
    COMMENT "Cythonizing ${MYMOD_PYX}"
)

add_library(mymod MODULE ${MYMOD_CPP} ${CMAKE_CURRENT_SOURCE_DIR}/src/mylib.cpp)
target_include_directories(mymod PRIVATE
    ${Python_INCLUDE_DIRS}
    ${Python_NumPy_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(mymod PRIVATE ${Python_LIBRARIES})
set_target_properties(mymod PROPERTIES PREFIX "")

