cmake_minimum_required(VERSION 3.26)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_COMPILER $ENV{CUDA_HOME}/bin/nvcc)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_pre)

project(cuda_test CXX CUDA)

add_executable(cuda_test cuda.cu)

set(TORCH_INSTALL_PREFIX $ENV{CONDA_PREFIX}/lib/python3.13/site-packages/torch/)

# Include directories
set(TORCH_INCLUDE_DIRS
        ${TORCH_INSTALL_PREFIX}/include
        ${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include
)

set(PYTHON_INCLUDE_DIR $ENV{CONDA_PREFIX}/include/python3.13)
set(INCLUDE_DIR 
        $ENV{CONDA_PREFIX}/include/
        ${PYTHON_INCLUDE_DIR}
        ${TORCH_INCLUDE_DIRS}
)
target_include_directories(cuda_test PRIVATE ${INCLUDE_DIR})

# link directories
link_directories(${LIB_DIR})
link_directories("$ENV{CONDA_PREFIX}/lib")

# Library directories
set(TORCH_LIB_DIR ${TORCH_INSTALL_PREFIX}/lib)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_INSTALL_PREFIX}/share/cmake/Torch)
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

set(Python_LIBRARIES $ENV{CONDA_PREFIX}/lib/libpython3.13.so)
target_link_libraries(cuda_test "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}" ${Python_LIBRARIES})


set_target_properties(cuda_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
