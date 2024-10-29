cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(MatMul CUDA)

# Specify the CUDA toolkit required
find_package(CUDA REQUIRED)

# Add the CUDA source file to the project
add_executable(matMul matMu.cu)

# Set the CUDA compiler
set_target_properties(matMul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)