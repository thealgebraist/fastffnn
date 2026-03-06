cmake_minimum_required(VERSION 3.18)
project(FICAR10_CUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)

add_executable(cuda_curriculum src/linux_cuda_curriculum.cu)
target_link_libraries(cuda_curriculum PRIVATE CUDA::cublas CUDA::cusolver)

add_executable(sa_solver src/sa_solver.cu)
target_link_libraries(sa_solver PRIVATE CUDA::cublas CUDA::curand)

add_executable(mphc_solver src/mphc_solver.cu)
target_link_libraries(mphc_solver PRIVATE CUDA::cublas CUDA::curand)

add_executable(row_kfac_solver src/row_kfac_solver.cu)
target_link_libraries(row_kfac_solver PRIVATE CUDA::cublas CUDA::cusolver)

add_executable(full_radam_solver src/full_radam_solver.cu)
target_link_libraries(full_radam_solver PRIVATE CUDA::cublas)
