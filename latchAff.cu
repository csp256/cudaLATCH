// Uncomment below define to use sum of squared differences instead of sum of absolute differences.
// #define use_SAD // You can keep this commented.
// Uncomment below define to use an importance mask on each patch comparison.
// #define use_mask // You can keep this commented.

#include <vector>
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;

#define _warpSize (32)
#define _warpSizef (32.0f)
#define warpsPerBlock (32)
// Region of interest
#define roiWidth (64)
#define roiHeight (64)
// Minimal amount that avoids shared memory bank conflicts
#define roiWidthPadding (4)
// The numbers loaded into the oracle assume patchSize==8. If you really want canonical LATCH, you can set the mask to ignore those pixels.
#define patchSize (8)
#define bitsPerDescriptor (512)
// With further work this should be decreased to 0 for even faster matching.
#define paddingBitsPerDescriptor (1536)
#define bitsPerUInt32 (32)
#define deg2rad (0.0174533f)
#define negDeg2rad (-0.0174533f)
#define CHECK_BORDER (0)

// Used to store the oracle of patch triplets.
texture<int, cudaTextureType2D, cudaReadModeElementType> patchTriplets;
texture<unsigned char, 2, cudaReadModeNormalizedFloat> image;

__constant__ int triplets[3*512];

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

#define checkLaunchError()                                          \
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

 // Launch as 32x32
__global__ void __launch_bounds__(1024, 2)
                latch(  const float *g_K,
                        unsigned int *g_D,
                        const int imgWidth,
                        const int imgHeight,
                        const float *g_mask/*,
                        const float *g_oriented*/,
                    float *g_out) {
    volatile __shared__ int s_kpOffset[2];
    volatile __shared__ float s_mask[64];
    volatile __shared__ float s_stride[4];
    volatile __shared__ float s_roi[roiHeight][roiWidth + roiWidthPadding][1]; // It is faster to operate on floats, even if our image is unsigned char! (we can't guarantee media instructions are available)
    volatile __shared__ unsigned int s_out[warpsPerBlock];
    {
        register float mask0, mask1;
        if (threadIdx.y == 0 && threadIdx.x < 2) { // 2 threads, 2 coordinates
            register float k;
            k = g_K[blockIdx.x*5 + threadIdx.x];
            if (CHECK_BORDER && (k < _warpSize || (threadIdx.x == 0 && imgWidth-_warpSize < k) || (threadIdx.x == 1 && imgHeight-_warpSize < k))) {
                k = -999999; // If too near boundary, make sure the kpOffset will be negative, so everyone gets the signal to bail.
            }
            s_kpOffset[threadIdx.x] = k + 0.5f;
        }
        if (threadIdx.y == 1 && threadIdx.x < 4) {
            register float l; // Just a temp variable... thread 0 has major axis, 1 has minor axis, 2 has angle.
            if (threadIdx.x < 3) {
                l = g_K[blockIdx.x*5 + 2 + threadIdx.x];
            }
            register float c, s, t; // cos and sin and theta
            t = __shfl(l, 2);
            __sincosf(negDeg2rad*t, &s, &c);
            register float a,b;
            a = __shfl(l, 0) * 0.015625f;
            b = __shfl(l, 1) * 0.015625f;

            const register float p = ((threadIdx.x & 1) == 0) ? a : b;
            const register float q = (threadIdx.x == 1 || threadIdx.x == 2) ? s : c;
            
            s_stride[threadIdx.x] = p*q;
        }
        if (threadIdx.y == 2) {
            mask0 = g_mask[threadIdx.x];
        }
        if (threadIdx.y == 3) {
            mask1 = g_mask[_warpSize + threadIdx.x];
        }
        __threadfence_block();
        __syncthreads();
        const register int x = s_kpOffset[0];
        const register int y = s_kpOffset[1];
        if (x < 0 || y < 0) {
            return; // This is the case if our keypoint is within boundary of the edge of the image. 5000 blocks returning in this way takes ~300 microseconds.
        }
        const register float r = (threadIdx.x < 4) ? s_stride[threadIdx.x] : 0.0f;

        const register float r11 =  __shfl(r, 0);
        const register float r12 =  __shfl(r, 1);
        const register float r21 = -__shfl(r, 2);
        const register float r22 =  __shfl(r, 3);
        // const register float c = s_stride[0];
        // const register float s = s_stride[1];

        // 64 by 64 region of interest means four 32 by 32 loads.
        if (threadIdx.y == 2) {
            s_mask[threadIdx.x] = mask0;
        }
        if (threadIdx.y == 3) {
            s_mask[threadIdx.x + _warpSize] = mask1;
        }

        // const register float cN = c * -_warpSizef;
        // const register float sN = s * -_warpSizef;
        //
        // const register float cu = c * threadIdx.x;
        // const register float su = s * threadIdx.x;
        // const register float cv = c * threadIdx.y;
        // const register float sv = s * threadIdx.y;
        //
        // const register float cuN = cu + cN;
        // const register float suN = su + sN;
        // const register float cvN = cv + cN;
        // const register float svN = sv + sN;

        const register float nx = threadIdx.x - _warpSizef;
        const register float ny = threadIdx.y - _warpSizef;


        s_roi[threadIdx.y            ][threadIdx.x            ][0] = /*(unsigned char)*/ ( 256.0f * tex2D(image, min((float)imgWidth-1.0f, r11*nx          + r12*ny          + x), min((float)imgHeight-1.0f, r21*nx          + r22*ny           + y)));
        s_roi[threadIdx.y            ][threadIdx.x + _warpSize][0] = /*(unsigned char)*/ ( 256.0f * tex2D(image, min((float)imgWidth-1.0f, r11*threadIdx.x + r12*ny          + x), min((float)imgHeight-1.0f, r21*threadIdx.x + r22*ny           + y)));
        s_roi[threadIdx.y + _warpSize][threadIdx.x            ][0] = /*(unsigned char)*/ ( 256.0f * tex2D(image, min((float)imgWidth-1.0f, r11*nx          + r12*threadIdx.y + x), min((float)imgHeight-1.0f, r21*nx          + r22*threadIdx.y  + y)));
        s_roi[threadIdx.y + _warpSize][threadIdx.x + _warpSize][0] = /*(unsigned char)*/ ( 256.0f * tex2D(image, min((float)imgWidth-1.0f, r11*threadIdx.x + r12*threadIdx.y + x), min((float)imgHeight-1.0f, r21*threadIdx.x + r22*threadIdx.y  + y)));
    }
    register unsigned int out = 0;
    const register int wrappedX =      threadIdx.x % patchSize; // Offset for patch, interlaced to decrease padding needed for shared memory bank conflict avoidance
    const register int wrappedY = 2 * (threadIdx.x / patchSize); // Each thread will use both wrappedY and wrappedY+1
    __syncthreads();
    __threadfence_block();
    if (blockIdx.x == 0) {
        g_out[(threadIdx.y            )*64 + (threadIdx.x            )] = s_roi[threadIdx.y            ][threadIdx.x            ][0];
        g_out[(threadIdx.y            )*64 + (threadIdx.x + _warpSize)] = s_roi[threadIdx.y            ][threadIdx.x + _warpSize][0];
        g_out[(threadIdx.y + _warpSize)*64 + (threadIdx.x + _warpSize)] = s_roi[threadIdx.y + _warpSize][threadIdx.x + _warpSize][0];
        g_out[(threadIdx.y + _warpSize)*64 + (threadIdx.x            )] = s_roi[threadIdx.y + _warpSize][threadIdx.x            ][0];
    }
    __syncthreads();
    __threadfence_block();


    const register float mask0 = s_mask[threadIdx.x];
    const register float mask1 = s_mask[threadIdx.x + _warpSize];

    register int nextCoord = tex2D(patchTriplets, threadIdx.x, threadIdx.y); // This access is hardware cached.
    // register int offset = threadIdx.y * 3 * 16;
    // register int nextAleph = triplets[offset  ];
    // register int nextTavek = triplets[offset+1];
    // register int nextBet   = triplets[offset+2];
    #pragma unroll
    for (register int i=0; i<16; ) {
        const register int coord = nextCoord;
        if (i!=16-4) nextCoord = tex2D(patchTriplets, 6*(i+4) + threadIdx.x, threadIdx.y); // This access is hardware cached.
        #pragma unroll
        for (register int j=0; j<4; j++, i++) {
            // offset += 3;
            // const register int alephIndexX = nextAleph & 255;
            // const register int alephIndexY = nextAleph >> 8;
            // nextAleph = triplets[offset];
            // const register int tavekIndexX = nextTavek & 255;
            // const register int tavekIndexY = nextTavek >> 8;
            // nextTavek = triplets[offset+1];
            // const register int betIndexX = nextBet & 255;
            // const register int betIndexY = nextBet >> 8;
            // nextBet = triplets[offset+2];

            const register int alephIndexX = __shfl(coord, 6*j  );
            const register int alephIndexY = __shfl(coord, 6*j+1);
            const register int tavekIndexX = __shfl(coord, 6*j+2);
            const register int tavekIndexY = __shfl(coord, 6*j+3);
            const register int betIndexX = __shfl(coord, 6*j+4);
            const register int betIndexY = __shfl(coord, 6*j+5);
            const register int bitIndex = 16*(threadIdx.y & 1) + i;
            const register int outThread = 0;

            // This assumes an 8x8 patch. As there are only 32 threads per warp, each thread will pull two values from each thread.
            // The access pattern is interleaved to decrease the amount of shared memory padding necessary to avoid bank conflicts:
            //      each thread pulls a verticle pair from each patch.
            const register int tavek0 = s_roi[tavekIndexY + wrappedY  ][tavekIndexX + wrappedX][0]; // Tavek means "between".
            const register int tavek1 = s_roi[tavekIndexY + wrappedY+1][tavekIndexX + wrappedX][0]; // It is our root patch.
            const register int aleph0 = s_roi[alephIndexY + wrappedY  ][alephIndexX + wrappedX][0]; // Aleph is "A"
            const register int aleph1 = s_roi[alephIndexY + wrappedY+1][alephIndexX + wrappedX][0]; // Similarity to aleph is denoted by a bit set to 0
            const register int bet0   = s_roi[betIndexY   + wrappedY  ][betIndexX   + wrappedX][0]; // Bet is "B"
            const register int bet1   = s_roi[betIndexY   + wrappedY+1][betIndexX   + wrappedX][0]; // Similarity to bet is denoted by a bit set to 1

            // Now we compute the sum of squared differences between both patch pairs.
            // First, differences:
            register int alephDiff0 = (tavek0 - aleph0);
            register int alephDiff1 = (tavek1 - aleph1);
            register int betDiff0   = (tavek0 - bet0);
            register int betDiff1   = (tavek1 - bet1);
            // Then, squared differences
            alephDiff0 *= alephDiff0;
            alephDiff1 *= alephDiff1;
            betDiff0 *= betDiff0;
            betDiff1 *= betDiff1;

            alephDiff0 *= mask0;
            alephDiff1 *= mask1;
            betDiff0 *= mask0;
            betDiff1 *= mask1;

            alephDiff0 += alephDiff1; // Merge both interleaved squared differences, to make upcoming warp reduction faster
            betDiff0   += betDiff1;

            alephDiff0 -= betDiff0; // Easiest to just take this difference now, then reduce, then compare to 0. Same as reduce then compare relative to each other.
            alephDiff0 += __shfl_xor(alephDiff0,  1);
            alephDiff0 += __shfl_xor(alephDiff0,  2);
            alephDiff0 += __shfl_xor(alephDiff0,  4);
            alephDiff0 += __shfl_xor(alephDiff0,  8);
            alephDiff0 += __shfl_xor(alephDiff0, 16); // By xor shfling, every thread has the resulting sum.

            // One thread sets a specific bit high if tavek is closer to bet.
            if (alephDiff0 < 0 && threadIdx.x == outThread) {
                out |= (1<<bitIndex);
            }
        }
    }

    if (threadIdx.x == 0) { // In this case, only thread 0 ever has important data.
        s_out[threadIdx.y] = out;
    }
    __syncthreads();
    __threadfence_block();
    if (threadIdx.y == 0) {
        out = s_out[threadIdx.x]; // Warp 0 now has all the data we need to output.
        __syncthreads();
        __threadfence_block();

        out |= __shfl_down(out,  1, _warpSize); // Each warp computed half a 32 bit word. Merge them before output.
        out = __shfl(out, 2*threadIdx.x); // Only even threads have useful data after above shfl_down.

        if (threadIdx.x < bitsPerDescriptor / bitsPerUInt32) { // 512 / 32 = 16
            g_D[((paddingBitsPerDescriptor + bitsPerDescriptor) / bitsPerUInt32)*blockIdx.x + threadIdx.x] = out; // And that's it, with this write to global memory we're done with this descriptor.
        }
    }
}


void initPatchTriplets(cudaArray* cuArray) {
    const int h_data[]= { 41, 22, 47, 47, 51, 24, 32, 44, 52, 17, 32,  7, 50, 14, 26,  8, 51, 33, 45, 18, 30, 38, 42, 10,  6, 30, 16, 40,  6, 49, 39, 34, 35, 43, 31, 17, 21, 44, 18, 14, 25, 37, 23, 29, 12, 44, 19,  7,  9, 30, 26, 19,  6, 52, 47, 40, 27,  9, 43, 19, 35, 26, 50,  5, 41, 48, 25, 37, 11, 27, 23,  9, 25, 14, 33,  7, 38, 47, 40, 19, 52, 48, 48,  8, 23, 46, 47, 39, 22, 12, 50, 35, 29, 20, 18, 34, 47, 24, 31, 36, 26, 47, 11, 38, 17, 16,  7, 11, 52, 15, 46, 14, 42,  9,  4, 13, 43, 14,  5, 17, 22, 50, 27, 17, 34, 14, 44, 46, 38,  5, 48, 32, 51, 36, 32, 35, 45,  9, 26,  7, 17, 10, 25, 35,  5, 38, 17, 33, 12, 47,  4, 32, 43, 12,  9, 23,  9, 24, 27, 33,  8, 30, 48, 40, 39,  4, 37, 50, 37, 41, 22,  5, 38, 43,  6, 20, 24, 23, 13, 48, 22, 15, 29, 44, 22, 51, 10, 25, 20, 13, 10, 33, 42, 16, 37, 41, 47, 40,  6, 44, 27, 47, 44, 16, 29, 36, 27, 24, 25, 35, 31, 43, 51,  5, 33, 19, 30, 21, 42, 15, 34, 48, 10, 39, 44, 18, 16, 32, 13, 30, 19, 49,  7, 48, 25, 33,  6, 51, 21,  6, 11, 41, 52, 14,  4,  4, 52, 43, 31,  6, 12, 35, 14,  8, 29, 21, 16, 26, 47, 45, 28, 46, 16, 21, 16, 38, 36, 33,  7, 10, 13, 37, 41, 31, 10, 11, 28, 33, 39,  6, 36, 46, 49, 30,  6, 11, 43, 31,  6, 13, 46, 51,  5, 49,  4, 44, 38, 25, 20, 27,  9, 47, 50, 51, 14, 26, 48, 43, 26,  9, 47, 13, 18, 40, 28, 19, 19, 12, 41,  6, 44, 44, 28, 14, 20, 15, 27, 48, 23,  6, 21,  5, 24, 38, 27, 48, 44, 27, 15, 12, 52, 10, 10, 40, 36, 47,  4, 14, 13, 52, 34, 30,  7,  6, 48, 26, 36, 28, 45, 18,  9, 49, 21, 48, 14, 31, 47, 45, 28, 12, 10,  9, 11,  9, 40,  5, 16, 20, 37, 38, 37,  5, 49,  4, 37, 47, 13, 10, 35,  9, 33, 31, 25, 12, 32, 26, 43, 18,  4, 44, 52, 39, 45, 44, 19, 29, 46, 13, 39, 23, 28, 52,  8, 16, 14,  9, 52, 12, 19, 22, 50, 14, 30,  6, 44, 39, 51, 27, 32, 18, 48, 50, 18, 19, 45, 41, 15, 15, 13, 41, 39, 37, 15, 37, 50, 43, 30, 46, 16, 18, 31, 51, 46, 43, 48,  4, 35, 22, 44, 39, 36, 29, 41, 44, 52,  8, 37, 24, 20, 25, 45, 52,  9, 45, 39, 34, 23, 50, 42, 18, 23, 17, 13, 18,  6, 37, 35, 46, 16, 36, 41,  4, 37, 28, 30, 31, 35, 40, 49, 42, 28, 20, 11, 30, 50, 48, 23, 44, 47,  5, 50, 10,  9, 25, 52, 13, 46, 28, 17, 44, 45, 39, 50, 43, 17, 35, 48, 19, 12, 38, 30, 29,  9, 48,  9, 24, 30, 25,  4, 45, 25, 49, 50, 16, 27, 31, 25,  8, 21, 51, 27, 19, 17, 31,  8, 23, 19, 20, 47, 11, 49, 49, 49, 15, 18, 34, 30, 26, 11,  7, 47, 52, 48, 34, 52, 17, 38,  5, 27, 19, 36, 23, 50,  8, 25, 52, 47, 33,  4, 34, 28, 15,  5, 13, 38, 48,  6, 24, 37,  8,  4, 38, 33, 13,  4,  8, 50, 34, 36, 21, 39, 50, 10, 35, 19, 47, 16, 23, 19, 49,  8, 11, 11, 50,  5, 34,  6, 16, 11, 35, 10, 31, 29, 52,  4, 48, 18, 19, 30, 43, 46, 46, 44, 15, 10, 39, 37, 22, 52, 52,  4, 50, 40, 16, 48, 35,  7, 43, 50, 23, 19, 21, 51, 15, 11,  8, 19, 22, 51, 28,  6, 41, 13, 10, 29,  6, 11, 38, 28, 32, 32, 20, 46, 20, 21, 22,  8, 46,  8, 25,  8, 14, 32, 19, 11, 20, 10, 21, 31, 36, 12, 33, 35, 16, 38, 47, 48, 49,  6, 52, 32, 36,  6, 30,  9, 10, 10, 50, 26, 41, 38, 37, 13, 43, 49, 44, 44, 39,  4, 26, 52, 49, 21, 16, 29, 42, 37, 45, 48, 45, 35, 35, 33,  4, 15, 20, 49, 46, 13, 39,  6, 36, 40, 20, 10, 51, 42, 38, 34,  4, 45, 18, 36, 41, 49, 45, 52, 25,  7,  4, 46, 39, 20, 33, 18,  5, 26, 51, 15, 33, 39, 35, 27,  7, 18, 24, 49,  6, 13, 34, 34, 24, 44, 21, 21,  5, 47, 34, 27, 49, 51, 14, 26, 11, 50, 15,  6, 32, 42, 31, 18, 31, 42, 17,  6, 36, 39, 41,  4, 38, 52, 49, 40, 30, 41, 12, 43, 29, 27, 24, 48,  6, 22,  9, 14,  8, 30, 17,  8, 52,  5, 18, 40, 29,  4, 30,  4,  5, 12, 41, 27, 17, 20, 34, 47, 15,  5, 51, 10,  4, 51, 12,  7, 44, 16, 47, 18, 34, 22, 12, 28, 13, 15, 52, 26, 37, 47, 24, 28, 49, 49, 44, 18,  4,  4,  8, 15, 23, 52, 35, 15, 35, 46, 47, 28, 50,  7, 48, 28, 46, 51, 38, 15, 14, 44, 38, 18, 16, 36, 38, 15, 52,  6, 22, 11, 42, 22, 39, 45, 45, 21, 45, 45, 16, 50, 27, 26, 25,  4, 50, 40, 28, 29, 17, 40, 12,  8, 22, 17, 45, 23,  9, 46, 35, 20, 31, 51, 17, 52, 21, 10, 52, 48, 27, 18, 32, 24,  6, 14, 20, 43, 20, 12, 48, 45, 51, 40, 43, 43,  9, 33, 32, 12, 49, 31, 25, 11, 13, 10, 42,  8,  6, 10, 40, 49, 41, 10, 28, 40, 16,  8, 51, 43, 18, 14, 12,  4, 44, 40, 23, 12, 41, 17, 15, 24, 19, 26, 10, 31, 16,  4, 28, 26, 25, 14, 14, 50, 37,  7, 45, 46, 38, 30, 51, 43, 34, 20, 10, 43, 51, 17, 51,  4, 41, 32, 44,  4, 15, 37, 28, 49,  5, 34,  4,  6, 41, 49, 47,  7, 18,  7, 47, 35, 26, 21, 29, 30,  7, 36, 48, 39, 16, 47,  9, 26, 52, 45, 29, 25, 21, 31, 45, 24, 15,  5, 23, 13, 14, 21, 39, 13,  5, 52, 50, 11, 46, 33, 21, 39,  6, 46, 23, 48, 17,  8, 28, 39, 32, 46, 46, 19, 35, 47, 45, 29, 11, 52,  4, 32, 31,  9,  5, 37, 51, 18, 37, 35, 26, 15, 33, 44, 23, 36, 15, 19,  5, 40, 41, 34,  7, 27, 28, 24, 46, 37, 11,  4,  6, 37, 45,  9, 30, 48, 14,  6, 51, 50, 39, 19, 14, 36, 24, 40,  6, 26, 41, 36, 49, 37, 20, 42, 46, 33, 19, 44, 15, 21, 21, 49, 16, 41, 16, 18, 39, 35, 39, 31, 36, 33, 22, 30, 42, 52,  6, 36, 51, 21, 18, 50, 39, 34, 48, 22, 19, 38, 23, 26, 27, 40, 43, 14, 42,  5, 34, 15, 25, 19, 30, 50, 27,  4, 18, 11, 50, 34, 19, 16, 15, 16, 29, 24, 37, 14, 26, 41, 30, 51, 26, 40, 33, 44, 14, 24,  6, 46, 45, 15, 20, 35, 23,  7, 11, 31, 27, 25, 26, 18, 47, 46, 34, 14, 52, 48, 38, 21,  8,  5, 38, 24, 20,  8, 19, 18, 44, 14,  7, 37, 45, 16, 49, 44, 52, 47,  6, 17, 16, 52,  8, 33, 13, 42, 40, 31, 15, 22, 48, 32, 50, 31,  8,  5, 16, 35, 40,  5, 44, 50, 31, 24, 46, 50, 36, 29, 29, 44, 37, 23, 46, 49, 21, 23, 13, 52, 22, 15, 42, 16, 51, 11, 46, 40, 46, 27, 28, 48, 35, 42, 21, 13, 11, 46, 26, 10, 16, 17, 42, 13,  7, 26,  7, 32, 12, 38, 26, 50,  5,  9, 11, 47, 44, 36, 50, 35,  6, 50,  6, 36, 11, 11, 28, 40, 50, 41, 36, 22, 47, 47, 22, 47, 24,  5, 18, 51, 15, 12,  9, 22, 46,  9, 51, 33, 10, 24, 16, 10,  4, 18, 37, 34, 32, 12,  4, 21, 13, 28, 31, 27, 52, 23, 40, 38,  4, 52, 50, 15, 37, 29, 46, 43, 35, 10, 25, 17,  6, 10, 23, 19, 35, 42, 52, 22, 27, 27,  4, 50, 47, 27, 41,  9, 31, 13, 12, 16, 29, 28, 40, 49, 49, 41,  6,  9, 19, 42, 40,  5, 45, 41, 17, 50, 31, 52, 14,  9, 33, 27, 48, 46, 43, 47,  9, 12, 52, 51, 35,  8, 15, 50, 49,  5, 25,  8, 47, 44, 26,  8,  9, 46, 46, 16, 12, 42, 23, 49, 12,  5, 23, 47, 36, 16, 40,  8, 23, 21, 22, 18,  4, 25, 46, 17, 23, 36, 42, 30, 25, 37, 34, 52, 30, 26, 22, 52, 16, 35, 20, 28, 36, 25, 49, 50, 29, 45, 16, 42,  5, 47, 10, 27, 51,  7, 18, 22, 27, 42,  6, 19, 19, 48,  4, 12, 17, 49, 47,  4, 37, 20, 45,  9, 21, 16, 25, 47,  4, 13, 28, 27,  7, 19, 50,  7, 24, 51, 32, 25, 39, 37, 32, 18, 38, 38, 32, 20, 35, 33, 13, 49,  5, 37, 16, 11,  7, 26, 13, 11, 13, 49, 40, 37, 51, 29, 19, 49, 48, 47, 22, 33, 27, 12,  7, 47, 25, 16, 43, 42, 31, 26, 30,  8, 11, 25, 12, 13, 15,  7, 39, 10, 49, 23, 11, 33, 39, 51, 35, 19, 45, 48, 22, 39, 14,  7, 51, 47,  7, 19, 22, 51,  4, 12, 35,  6, 49, 35, 40,  9, 16, 25, 47, 51, 38, 25, 10, 26, 50, 20, 12, 23, 51, 42, 49, 50,  9, 34, 19, 22,  4, 16, 15,  5, 37, 32,  7, 42, 24, 51, 46, 37, 35, 22, 34, 50, 28, 42, 15, 36, 52, 44, 14, 30,  8, 32, 15, 39, 17, 42, 16, 51, 35, 38, 49, 42, 24, 42, 50, 41, 14, 39, 16, 49, 47, 48, 20, 31,  8, 51, 15, 51, 51, 32, 46, 26, 38, 17, 48, 29, 49, 34, 43, 42, 25, 44, 52, 36, 17, 46, 51, 49, 25, 43,  5, 33, 23, 35, 37, 16, 24, 42, 46, 29,  4, 39, 19, 27, 38, 12, 18, 21, 50, 14, 33,  6, 46, 13,  4, 27, 36, 11, 44, 32, 28, 11, 52, 19, 20, 45, 25, 15, 49, 52, 38, 40, 40, 31, 13, 49, 34, 27, 23, 47, 47,  7, 13, 40, 42,  4, 13,  4, 38, 33, 17, 12, 22, 44, 20, 23,  8, 43, 21, 24, 48, 23, 40, 19, 48, 10, 40, 38, 14, 14, 52, 45, 16, 27, 41, 46, 47, 15, 50, 30, 37, 14, 47, 41, 16, 33, 46, 32,  4,  5, 23, 27, 17, 47, 41, 42, 17,  7, 20, 50,  6,  4, 49, 20,  7, 33, 42, 39, 24, 19, 18, 44, 30, 47, 16, 20, 42, 50,  5,  6, 15, 29, 24, 11, 32,  7, 38, 33, 31,  9, 46, 25, 10, 41, 43, 47,  5, 26, 40, 51,  9, 27, 18, 43, 21, 28,  8, 35, 28, 11, 27, 23, 43, 12,  8, 39,  7, 30, 13, 32, 30, 25, 33, 32, 26, 25, 14, 41, 50, 13, 47, 37, 11, 24, 46, 49, 35, 26, 33, 43, 50, 35,  5, 47, 42, 39, 42, 52,  5, 39, 34, 45, 49, 20, 15, 43, 39, 16,  5, 38, 36, 20, 17, 40, 23, 12,  9, 46, 22,  8,  4, 27,  6,  4, 19, 11, 16, 37, 47, 12, 52, 42, 19, 22, 35, 48,  5, 35, 47, 52, 28, 37, 51,  5, 50, 39, 35,  4, 50,  7, 28, 20, 42,  8, 51, 42, 20, 12, 13, 46, 39, 30, 22, 52, 35, 34, 52, 14, 52, 24, 31,  7, 30, 51, 38, 52,  4, 38, 38, 39, 33, 26, 43, 40, 35, 52, 39, 23, 34, 49, 40, 40, 50, 27, 41, 13, 10, 42,  5, 48, 29, 47, 51,  9,  6, 32, 26,  9, 48, 36, 30, 19, 38, 51, 49, 17, 17, 27, 43, 37, 51, 48, 29, 37, 37, 41, 49, 37,  6, 23, 12, 33, 17, 11,  5, 21, 37,  4, 51, 35, 19, 51, 30, 48, 12, 43, 10,  6, 46, 44, 42, 15, 10,  5, 20,  6, 41, 40, 19, 40, 48, 42, 16, 10, 23, 13, 31,  9, 36, 12, 15, 38, 38, 13, 45, 19, 33,  5, 44, 19, 31, 12,  9, 42, 49,  9,  6, 27, 33, 51, 41, 29,  4, 18, 47, 27,  5,  9,  5, 32,  9, 36, 32, 35, 46, 45, 40, 29, 35, 22, 46, 39,  4, 36, 46, 44, 14,  6, 39, 17, 26,  8, 42, 23, 37, 37,  5, 44, 52, 16, 20, 42, 22, 17, 33, 51, 22, 12, 23, 49, 13, 49,  6,  4, 26, 41, 20, 45, 47, 52, 24, 38, 34, 14,  7, 20, 41, 23, 27,  7, 16, 51,  4,  7, 11, 40, 39, 49, 43, 41, 51, 19, 44,  5, 26, 22, 30, 47, 32, 46,  4, 51, 34, 36,  5, 43, 26, 35, 48, 52, 38, 36, 52, 32, 25,  5, 33, 47, 25,  5, 51,  9,  8, 31, 43, 16, 34, 18, 51, 28, 31, 46,  6, 40, 36,  4, 47, 50, 30, 40, 28, 24,  4, 49, 44, 19, 25, 42, 42, 14, 32, 46, 39, 19, 14, 49,  5, 39, 50, 29, 32, 45, 25, 41,  6, 11, 51, 39, 43, 39, 14, 31, 37, 24, 16, 22, 44, 30, 33, 48, 34, 38, 27, 35, 49, 40, 35,  7, 40, 14,  7,  5, 41, 44, 52, 28, 18, 14, 12, 16, 22, 51, 36, 18, 37, 42, 10, 30, 52, 19, 23, 44, 45, 28, 27, 38, 49, 35, 28, 16, 13, 41, 17, 42,  8,  6, 15, 28, 29,  7, 13, 34,  5, 12,  8, 19, 52, 30, 11, 23, 32,  7, 46, 46,  6,  7, 22, 36, 25, 33, 45, 46, 38, 31, 28, 39, 50, 24, 16,  4, 38, 46, 48,  7,  4, 20,  9, 34,  4, 45, 35, 29, 36, 47, 36, 41,  5,  7,  4, 49, 30,  7, 13, 48, 45, 49, 25, 49, 46, 10, 18, 45, 10, 10, 38, 33, 22, 47, 38, 39, 50, 34, 52, 36, 41, 31, 20, 25, 16, 15, 32,  7, 51, 18, 33, 26,  6, 33, 19, 48, 11,  4, 44, 33, 25, 30, 33, 34,  4, 33, 49, 13, 50, 29, 35, 12, 28,  9,  7, 21, 38, 28,  5, 13, 22, 26, 10,  8, 20, 12, 47, 29, 43, 46, 32, 33, 32,  7, 14, 32, 30, 30, 47, 28, 20, 33, 35, 12, 10, 50, 30, 10, 50,  5, 30, 43,  7,  9, 18, 13, 40, 20, 14,  8, 17, 17, 31, 29, 48,  4, 48, 30, 31, 27, 52, 45, 47,  6, 30, 37,  5,  8, 25, 17, 17, 39,  8, 15,  5, 33, 27, 44, 21, 31, 37, 51, 26, 42, 51, 41, 26, 48, 16, 40, 46, 50, 29, 44,  9, 39, 36, 35, 51, 37, 37, 30,  8, 43,  5, 38, 28, 18, 51, 37, 32,  4, 10, 31, 43, 12, 21, 47, 11, 11, 29, 51, 39, 50, 21, 28, 52, 28, 25,  4,  6, 43, 37, 20, 24, 48, 14, 20, 14, 47, 37, 52, 26, 20, 24, 52, 42, 45, 27, 38,  5, 29, 43, 37, 27, 28,  4, 41, 16, 23, 38, 10, 22,  5, 39,  8, 49, 46, 32, 19, 21, 52, 43, 35, 25, 26, 39, 18,  4, 39, 30, 18, 41, 45, 11, 14, 10, 49, 14, 19, 11, 24, 19, 18, 26, 50,  7, 36, 17, 29, 51, 25, 13,  7,  8, 14, 47, 31, 18, 17, 50, 31,  7,  5, 13, 28, 37,  9, 40,  4, 25, 23, 50,  5, 43, 44, 19,  9, 10, 39, 27, 10, 34, 28,  4, 46,  5, 43, 17,  4, 32, 44, 29, 38,  7, 51, 29, 30, 18, 46, 26, 29, 33, 21,  5, 52, 12, 17,  6, 52,  9, 47, 40,  5, 30, 40, 28, 45, 37, 40, 16, 36, 39, 12, 12, 47, 28,  9,  7, 43, 48, 48,  4, 37, 31, 52, 29, 34, 49, 46,  9,  6, 49, 26, 14, 47, 50, 28, 11, 46, 16, 26, 38,  7, 20, 19, 21, 10, 13,  9, 27, 21,  7,  5, 13,  5, 23, 41, 49, 10, 29, 40, 34, 43, 16, 38,  8, 44, 15,  8, 22, 42, 41, 19, 26, 39, 14, 15, 23, 43, 24, 41, 45, 50, 25, 47, 11, 26, 39,  5, 50, 40, 44, 40, 24, 46, 37, 28, 37, 39,  8, 39, 29, 40, 17, 50, 19, 52,  5, 14, 15, 25, 33, 32, 40, 42, 40, 14, 31, 43, 45, 17,  4, 46,  5, 23, 31, 46, 37, 37, 48, 37, 31,  7, 18, 36, 27,  4,  5, 41, 32, 25,  9, 47, 29, 46, 10, 30,  7, 38, 41, 18, 11, 28, 40, 36, 47, 49, 36, 30,  5,  9, 36, 33, 24, 16, 46, 42, 16, 47,  9, 42, 33, 37, 49,  7,  7, 20, 29, 27, 42, 41, 34, 44,  4, 43, 42, 23, 49, 14, 20, 26, 39, 14,  7,  5, 47, 22, 22, 38, 18,  5, 26, 44, 44, 41, 14, 31, 13, 41,  5, 13, 15, 45, 40,  9, 47, 23, 46, 16, 38, 24, 12,  6, 13, 19, 10, 18, 44, 21, 23, 41, 10, 10,  5,  5, 50, 25,  4, 42, 48, 40, 44, 49, 17, 47, 47, 40, 10, 25, 11, 37, 14,  9, 17, 42, 15,  7, 51, 36, 22, 10, 40, 45, 29, 24, 27, 32, 47, 16, 21, 49, 31,  4, 49, 41, 36, 45, 51, 30, 43, 49, 24, 32, 44, 13,  8, 29, 34, 44,  6, 34, 39, 46, 16,  4, 27, 10, 36, 15, 26, 44, 22, 27, 21,  8,  8, 37, 18, 13, 34, 45, 44,  9, 45, 47, 28, 10, 20, 43,  5, 40, 22, 29, 39, 49, 13, 34, 47, 38,  4, 12, 51, 27, 20, 11, 14, 39, 30, 27, 35, 42, 26, 39, 48, 27, 24, 25,  5,  9, 48, 17, 26,  8,  4, 39, 16, 33,  7, 26, 15 };
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0,  	cudaChannelFormatKindSigned);
    cudaMallocArray(&cuArray, &channelDesc, 6*16, 32);
    cudaMemcpyToArrayAsync(cuArray, 0, 0, h_data, 6*16*32*sizeof(int), cudaMemcpyHostToDevice);

    patchTriplets.addressMode[0] = cudaAddressModeBorder;
    patchTriplets.addressMode[1] = cudaAddressModeBorder;
    patchTriplets.filterMode     = cudaFilterModePoint;
    patchTriplets.normalized     = false;

    cudaBindTextureToArray(patchTriplets, cuArray, channelDesc);

    // int h_data_packed[3*512];
    // for (int i=0; i<6*512; i+=2) {
    //     h_data_packed[i>>1] = h_data[i] + (h_data[i+1] << 8);
    // }
    // cudaMemcpyToSymbolAsync(triplets, h_data_packed, sizeof(float)*3*512);
}


void initImage(unsigned char ** d_I, int width, int height, size_t * pitch) {
    cudaMallocPitch((void**)d_I, pitch, width*sizeof(*d_I), height);

    image.addressMode[0] = cudaAddressModeClamp;
    image.addressMode[1] = cudaAddressModeClamp;
    image.addressMode[2] = cudaAddressModeClamp;
    image.normalized = false;
    image.filterMode = cudaFilterModeLinear;
    size_t tex_ofs;
    cudaBindTexture2D (&tex_ofs, &image, *d_I, &image.channelDesc, width, height, *pitch);
}

void initMask(float** d_mask, float* h_mask) {
    // This packs even rows together in h_mask, then odd rows.
    // It is 'run once' code.
    float t[64];
    for (int i=0; i<64; i++) {
        t[i] = h_mask[i];
    }
    for (int r=0; r<4; r++) {
        for (int c=0; c<8; c++) {
            h_mask[c + r*8] = t[c + r*16];
        }
    }
    for (int r=4; r<8; r++) {
        for (int c=0; c<8; c++) {
            h_mask[c + r*8] = t[c + (r-4)*16 + 8];
        }
    }
    size_t sizeMask = 64 * sizeof(float);
    cudaMalloc((void **) d_mask, sizeMask);
    cudaMemcpy(*d_mask, h_mask, sizeMask, cudaMemcpyHostToDevice);
}

float computeGradient(const unsigned char* img, const int width, const int x, const int y) {
    float dx = 0.0f;
    float dy = 0.0f;
    float delta = 0.0f;
    int base = x + y*width;
    int offset;

    offset = 3*width;
    delta = (img[base + offset] - img[base - offset]);
    dy += delta;

    offset = 3*width + 1;
    delta = (img[base + offset] - img[base - offset]);
    dy += delta * 3 / sqrt(10);
    dx += delta     / sqrt(10);

    offset = 2*width + 2;
    delta = (img[base + offset] - img[base - offset]);
    dy += delta     / sqrt(2);
    dx += delta     / sqrt(2);

    offset = 1*width + 3;
    delta = (img[base + offset] - img[base - offset]);
    dy += delta     / sqrt(10);
    dx += delta * 3 / sqrt(10);

    offset = 3;
    delta = (img[base + offset] - img[base - offset]);
    dx += delta;

    offset = -1*width + 3;
    delta = (img[base + offset] - img[base - offset]);
    dy -= delta     / sqrt(10);
    dx += delta * 3 / sqrt(10);

    offset = -2*width + 2;
    delta = (img[base + offset] - img[base - offset]);
    dy -= delta     / sqrt(2);
    dx += delta     / sqrt(2);

    offset = -3*width + 1;
    delta = (img[base + offset] - img[base - offset]);
    dy -= delta * 3 / sqrt(10);
    dx += delta     / sqrt(10);

    return atan2f(dy, dx);
}

void latchAff( Mat imgMat,
            unsigned char* d_I,
            size_t pitch,
            float* h_K,
            unsigned int* d_D,
            int* keypoints,
            int maxKP,
            float* d_K,
            vector<KeyPoint>* vectorKP,
            float* d_mask,
            cudaEvent_t latchFinished,
            Mat outMat,
            RotatedRect rekt) {
    const unsigned char* h_I = imgMat.data;
    const int height = imgMat.rows;
    const int width = imgMat.cols;

    // All of these calls are non blocking but serialized.
    // cudaMemsetAsync(d_K, -1, maxKP * sizeof(int) * 4); // Negative one is represented by all '1' bits in both int32 and uchar8.
    // cudaMemsetAsync(d_D,  0, maxKP * (2048 / 32) * sizeof(unsigned int));
    cudaMemcpy2DAsync(d_I, pitch, h_I, width*sizeof(unsigned char), width*sizeof(unsigned char), height, cudaMemcpyHostToDevice);

    // Only prep up to maxKP for the GPU (as that is the most we have prepared the GPU to handle)
    *keypoints = ((*vectorKP).size() < maxKP) ? (*vectorKP).size() : maxKP;
    for (int i=0; i<*keypoints; i+=1) {
        h_K[5*i  ] = (*vectorKP)[i].pt.x;
        h_K[5*i+1] = (*vectorKP)[i].pt.y;
        h_K[5*i+2] = 1.0f; // (*vectorKP)[i].size);
        // h_K[4*i+3] = (*vectorKP)[i].angle;
        h_K[5*i+3] = computeGradient(h_I, width, h_K[5*i  ], h_K[5*i+1]);
    }
    for (int i=*keypoints; i<maxKP; i++) {
        h_K[5*i  ] = -1.0f;
        h_K[5*i+1] = -1.0f;
        h_K[5*i+2] = -1.0f;
        h_K[5*i+3] = -1.0f;
    }
    h_K[0] = rekt.center.x;
    h_K[1] = rekt.center.y;
    h_K[2] = (rekt.size.width)/64;
    h_K[3] = (rekt.size.height)/64;
    h_K[4] = -3.1415926535f*rekt.angle/180.0f;
    cerr << h_K[0] << endl;
    cerr << h_K[1] << endl;
    cerr << h_K[2] << endl;
    cerr << h_K[3] << endl;
    cerr << h_K[4] << endl;

    size_t sizeK = *keypoints * sizeof(float) * 5;
    cudaMemcpyAsync(d_K, h_K, sizeK, cudaMemcpyHostToDevice);

    float *d_out, *h_out;
    size_t sizeOut = sizeof(float) * 64 * 64;
    cudaMallocHost((void **) &h_out, sizeOut);
    cudaCalloc((void **) &d_out, sizeOut);


    dim3 threadsPerBlock(_warpSize, warpsPerBlock);
    dim3 blocksPerGrid(*keypoints, 1, 1);
    checkLaunchError();
    latch<<<blocksPerGrid, threadsPerBlock>>>(d_K, d_D, width, height, d_mask, d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, sizeOut, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int j=0; j<64; j++) {
        for (int i=0; i<64; i++) {
            outMat.at<uchar>(j, 3*i) = h_out[j*64+i];
            outMat.at<uchar>(j, 3*i+1) = h_out[j*64+i];
            outMat.at<uchar>(j, 3*i+2) = h_out[j*64+i];
            // cerr << " " << h_out[j*64+i];
        }
    }
    cerr << endl;
    checkLaunchError();
    cudaEventRecord(latchFinished);
}
