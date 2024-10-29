cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(VectorAdd CUDA)

# Specify the CUDA toolkit required
find_package(CUDA REQUIRED)

# Add the CUDA source file to the project
add_executable(vectorAdd vectorAdd.cu)

# Set the CUDA compiler
set_target_properties(vectorAdd PROPERTIES CUDA_SEPARABLE_COMPILATION ON)