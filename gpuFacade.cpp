#include <vector>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;
#include "latch.h"
#include "bitMatcher.h"
#include "gpuFacade.hpp"

// images
// keypoints
// descriptors
// matches

using namespace std;

#define cudaCalloc(A, B) \
    do { \
        cudaError_t __cudaCalloc_err = cudaMalloc(A, B); \
        if (__cudaCalloc_err == cudaSuccess) cudaMemset(*A, 0, B); \
    } while (0)

#define checkError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define checkLaunchError()                                            \
do {                                                                  \
    /* Check synchronous errors, i.e. pre-launch */                   \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
    /* Check asynchronous errors, i.e. kernel failed (ULF) */         \
    err = cudaThreadSynchronize();                                    \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString( err) );      \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

gpuFacade::~gpuFacade() {
    // cudaFreeArray(patchTriplets); // This crashes..?
    cudaFree(d_K);
    cudaFree(d_D1);
    cudaFree(d_D2);
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFreeHost(h_K1);
    cudaFreeHost(h_K2);
    cudaDeviceReset();
}

gpuFacade::gpuFacade(int maxKeypoints, int input_WIDTH, int input_HEIGHT, int imageSlots) {
    maxKP = maxKeypoints;
    WIDTH = input_WIDTH;
    HEIGHT = input_HEIGHT;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Sizes for device and host pointers
    sizeK = maxKP * sizeof(float) * 4; // K for keypoints
    sizeI = WIDTH * HEIGHT * sizeof(unsigned char); // I for Image
    sizeD = maxKP * (2048 / 32) * sizeof(unsigned int); // D for Descriptor
    sizeM = maxKP * sizeof(int); // M for Matches
    sizeMask = 64 * sizeof(float);

    // Host pointers
    cudaMallocHost((void **) &h_K1, sizeK);
    cudaMallocHost((void **) &h_K2, sizeK);
    h_M1 = (int*) malloc(sizeM);
    h_M2 = (int*) malloc(sizeM);
    for (int i=0; i<64; i++) { h_mask[i] = 1.0f; }

    // Device pointers
    cudaCalloc((void **) &d_K, sizeK);
    cudaCalloc((void **) &d_D1, sizeD);
    cudaCalloc((void **) &d_D2, sizeD);
    cudaCalloc((void **) &d_M1, sizeM);
    cudaCalloc((void **) &d_M2, sizeM);
    cudaCalloc((void **) &d_mask, sizeM);

    // The patch triplet locations for LATCH fits in texture memory cache.
    initPatchTriplets(patchTriplets);
    initImage(&d_I, WIDTH, HEIGHT, &pitch);
    initMask(&d_mask, h_mask);

    // Events allow asynchronous, nonblocking launch of subsequent kernels after a given event has happened,
    // such as completion of a different kernel on a different stream.
    cudaEventCreate(&latchFinished);
    // You should create a new stream for each bitMatcher kernel you want to launch at once.
    cudaStreamCreate(&streamKP1);
    cudaStreamCreate(&streamKP2);
}

void gpuFacade::LATCH(
                Mat img,
                unsigned int* d_descriptor,
                int* keypoints,
                vector<KeyPoint>* vectorKP) {
    latch( img, d_I, pitch, h_K1, d_descriptor, keypoints, maxKP, d_K, vectorKP, d_mask, latchFinished );
}

void gpuFacade::match(
                unsigned int* d_descriptorQ,
                unsigned int* d_descriptorT,
                int numKP_Q,
                int numKP_T,
                int* d_matches,
                int threshold,
                cudaStream_t stream) {
    bitMatcher( d_descriptorQ, d_descriptorT, numKP_Q, numKP_T, maxKP, d_matches, threshold, stream, latchFinished );
}

void gpuFacade::getResults(int* h_matches, int* d_matches) {
    getMatches(maxKP, h_matches, d_matches);
}
