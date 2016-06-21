#include <vector>
#include <iostream>
#include <time.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;
#include "latch.h"
#include "bitMatcher.h"

#define cudaCalloc(A, B) \
    do { \
        cudaError_t __cudaCalloc_err = cudaMalloc(A, B); \
        if (__cudaCalloc_err == cudaSuccess) cudaMemset(*A, 0, B); \
    } while (0)

int main( int argc, char** argv ) {
    if (argc != 3) {
        cout << "Please pass the names of two images (same size) as arguments:    ./min img1.png img2.png" << endl;
        return -1;
    }
    // maKP is the maximum number of keypoints/features you will be able to use on the GPU.
    // This _must_ be an integer multiple of 512.
    // Integers which are themselves a multiple of the number of streaming multiprocessors
    // on your GPU (or half that number) should work well.
    const int maxKP = 512 * 15;
    int matchThreshold = 8; // Second best match must be at least matchThreshold from best match.
    int fastThreshold = 12; // For keypoint detection.
    clock_t t;

    Mat img1, img2, img1g, img2g, imgMatches;
    vector<KeyPoint> keypoints1, keypoints2;
    vector<DMatch> matches;

    img1 = imread(argv[1], IMREAD_COLOR);
    img2 = imread(argv[2], IMREAD_COLOR);
    cv::cvtColor(img1, img1g, CV_BGR2GRAY);
    cv::cvtColor(img2, img2g, CV_BGR2GRAY);
    const int imgWidth = img1.cols; // Assumes both images are the same size.
    const int imgHeight = img1.rows;

    // Sizes for host and device arrays both.
    size_t sizeK = maxKP * sizeof(int) * 2; // K for Keypoint
    size_t sizeI = imgWidth * imgHeight * sizeof(unsigned char); // I for Image
    size_t sizeD = maxKP * (2048 / 32) * sizeof(unsigned int); // D for Descriptor. 32 bits per uint32. 2048 bits per descriptor.
    size_t sizeM = maxKP * sizeof(int); // M for Matches

    // Host (CPU) arrays
    int *h_K1, *h_K2;
    cudaMallocHost((void **) &h_K1, sizeK); // Page locked memory is faster to transfer to-and-from the GPU
    cudaMallocHost((void **) &h_K2, sizeK); // (but that isnt really our bottleneck)
    int h_M1[maxKP];
    int h_M2[maxKP]; // For reasons unknown to me, if I use cudaMallocHost for h_M2 everything breaks...? Would love to know why.
    int numKP1, numKP2; // Minimum of the vector of keypoints.size() and maxKP (the max number of keypoints the GPU is prepared to handle)

    // Device (GPU) pointers. You can not directly look at device memory (without transfering it back to the host, aka CPU)
    unsigned char *d_I;
    unsigned int *d_D1, *d_D2;
    int *d_K, *d_M1, *d_M2;
    float *d_K, *d_mask;
    cudaCalloc((void **) &d_K, sizeK);
    cudaCalloc((void **) &d_I, sizeI);
    cudaCalloc((void **) &d_D1, sizeD);
    cudaCalloc((void **) &d_D2, sizeD);
    cudaCalloc((void **) &d_M1, sizeM);
    cudaCalloc((void **) &d_M2, sizeM);
    cudaCalloc((void **) &d_mask, sizeM);

    // The patch triplet locations for LATCH fits in texture memory cache.
    cudaArray* triplets;
    initPatchTriplets(triplets);

    size_t pitch;
    initImage(&d_I, imgWidth, imgHeight, &pitch);
    initMask(&d_mask, h_mask);

    // Events allow asynchronous, nonblocking launch of subsequent kernels after a given event has happened.
    cudaEvent_t latchFinishedEvent;
    cudaEventCreate(&latchFinishedEvent);
    // You should create a new stream for each bitMatcher kernel you want to launch at once.
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Normal OpenCV CPU code.
    FAST(img1g, keypoints1, fastThreshold);
    FAST(img2g, keypoints2, fastThreshold); // If we were clever, we would put this after the first LATCH call, so both the CPU and GPU would be working at the same time.

    t = clock(); // Begin timing kernel launches.
    // LATCH runs on the default stream and will block until it is finished.
    latch( img1g, d_I, pitch, h_K1, d_D1, &numKP1, maxKP, d_K, &keypoints1, d_mask, latchFinishedEvent);
    latch( img2g, d_I, pitch, h_K2, d_D2, &numKP2, maxKP, d_K, &keypoints2, d_mask, latchFinishedEvent);

    // latch( img1g, h_K1, d_D1, &numKP1, maxKP, d_K, d_I, &keypoints1, imgWidth, imgHeight, latchFinishedEvent ); // The latchFinishedEvent will be overridden by the next LATCH launch.  (this one will be ignored)
    // latch( img2g, h_K2, d_D2, &numKP2, maxKP, d_K, d_I, &keypoints2, imgWidth, imgHeight, latchFinishedEvent ); // This call will only begin after the above has completed. (but is still non blocking)
    bitMatcher( d_D1, d_D2, numKP1, numKP2, maxKP, d_M1, matchThreshold, stream1, latchFinishedEvent ); // Each concurrent bitMatcher launch should get its own d_M# pointer and its own stream#
    bitMatcher( d_D2, d_D1, numKP2, numKP1, maxKP, d_M2, matchThreshold, stream2, latchFinishedEvent ); // Both bitMatcher launches will start in parallel when the most recent call to LATCH completes.
    cout << "Launching kernels took " << 1000*(clock() - t)/(float)CLOCKS_PER_SEC << " milliseconds." << endl;

    // Put as much CPU code as possible here.
    // The CPU can continue to do useful work while the GPU is thinking.
    // If you put no code here, the CPU will stall until the GPU is done.

    t = clock(); // Begin timing wasted CPU time.
    getMatches(maxKP, h_M1, d_M1);
    getMatches(maxKP, h_M2, d_M2);
    cout << "Gathering results took " << 1000*(clock() - t)/(float)CLOCKS_PER_SEC << " milliseconds." << endl;
    for (int i=0; i<numKP1; i++) {
        if (h_M1[i] >= 0 && h_M1[i] < numKP2 && h_M2[h_M1[i]] == i) {
            matches.push_back( DMatch(i, h_M1[i], 0));
        }
    }
    cout << "Between " << keypoints1.size() << " and " << keypoints2.size() << " keypoints found " << matches.size() << " matches." << endl;

    drawMatches( img1, keypoints1, img2, keypoints2,
        matches, imgMatches, Scalar::all(-1), Scalar::all(-1),
        vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    resize(imgMatches, imgMatches, Size(1280, 480));
    imshow( "Matches", imgMatches );
    waitKey(0);

    cudaFreeArray(triplets);
    cudaFree(d_K);
    cudaFree(d_I);
    cudaFree(d_D1);
    cudaFree(d_D2);
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFreeHost(h_K1);
    cudaFreeHost(h_K2);
    return 0;
}
