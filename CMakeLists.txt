PROJECT(latch_cuda)

FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenCV 3 REQUIRED)

INCLUDE(FindCUDA)

SET(CUDALATCHSRCS
    latch.cu
    bitMatcher.cu
)

SET(CUDALATCHAFFSRCS
    latchAff.cu
    bitMatcher.cu
)

LIST(APPEND CUDA_NVCC_FLAGS "-D_MWAITXINTRIN_H_INCLUDED -lineinfo -Xptxas -v -Xcompiler -fopenmp -use_fast_math -O3 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_60,code=sm_60 --default-stream per-thread")
message("CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

CUDA_ADD_LIBRARY(latch_cuda ${CUDALATCHSRCS} STATIC)
CUDA_ADD_LIBRARY(latch_aff_cuda ${CUDALATCHAFFSRCS} STATIC)
CUDA_ADD_EXECUTABLE(latch_min_test min.cpp)
CUDA_ADD_EXECUTABLE(latch_vo vo.cpp)
CUDA_ADD_EXECUTABLE(latch_vo2 vo2.cpp gpuFacade.cpp)
CUDA_ADD_EXECUTABLE(latch_affTest affTest.cpp)

INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

TARGET_LINK_LIBRARIES(latch_cuda ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(latch_min_test latch_cuda)
TARGET_LINK_LIBRARIES(latch_vo latch_cuda)
TARGET_LINK_LIBRARIES(latch_vo2 latch_cuda)
TARGET_LINK_LIBRARIES(latch_affTest latch_aff_cuda)

