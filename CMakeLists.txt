cmake_minimum_required(VERSION 3.26)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_COMPILER $ENV{CUDA_HOME}/bin/nvcc)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS} -g -G --maxrregcount=32")

project(cuda_test CXX CUDA)

add_executable(cuda_test cuda.cu)

set(TORCH_INSTALL_PREFIX $ENV{CONDA_PREFIX}/lib/python3.13/site-packages/torch/)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_INSTALL_PREFIX}/share/cmake/Torch)
find_package(Torch REQUIRED)
target_include_directories(cuda_test PRIVATE ${TORCH_INCLUDE_DIRS})
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
target_link_libraries(cuda_test "${TORCH_LIBRARIES}")
target_link_libraries(cuda_test "${TORCH_PYTHON_LIBRARY}")
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")

set(PYTHON_LIBRARIES $ENV{CONDA_PREFIX}/lib/libpython3.13.so)
set(PYTHON_INCLUDE_DIR $ENV{CONDA_PREFIX}/include/python3.13)
target_include_directories(cuda_test PRIVATE ${PYTHON_INCLUDE_DIR})
target_link_libraries(cuda_test ${PYTHON_LIBRARIES})

set_target_properties(cuda_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
