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


// Sometimes the recovered pose is 180 degrees off...? I thought cheirality test would handle that, but apparently not always.
double dist2(Mat a, Mat b) {
    double s = 0.0;
    for (int i=0; i<3; i++) {
        const double t = a.at<double>(i) - b.at<double>(i);
        s += t*t;
    }
    return s;
}

// In general a suffix of 1 means previous frame, and 2 means current frame.
// However, we start processing the next frame while the GPU is working on current...
// So at a certain point frame 1 shifts down to 0, 2 shifts down to 1, and the new 2 is loaded.
int main( int argc, char** argv ) {
    // This must be an integer multiple of 512.
    // Specifically, half-multiples of the number of SM's for your GPU are sensible.
    // I have 10 streaming multiprocessors, so I chose 15*512 = 7680.
    const int maxKP = 512 * 15;
    const bool showMatches = true;
    // Shows every Nth processed frame's matches.
    const int showMatchesInterval = 10;
    const bool showVideo = true;
    // Shows every Nth processed frame.
    const int showVideoInterval = 1;
    int WIDTH, HEIGHT, totalMatches = 0, totalInliers = 0;
    const int matchThreshold = 12;
    // Discard this many frames for each one processed. Change with +/- keys while running.
    int skipFrames = 0;
    // Threshold for FAST detector
    int threshold = 90;
    int targetKP = 3000;
    int tolerance = 200;
    int maxLoops = 150;
    const bool gnuplot = true;
    double defect = 0.0;

    VideoCapture cap;
    // if (argc == 1) {
    //     cap = VideoCapture(0);
    //     WIDTH  = cap.get(CAP_PROP_FRAME_WIDTH);
    //     HEIGHT = cap.get(CAP_PROP_FRAME_HEIGHT);
    // }
    if (argc == 2 || argc == 3) {
        cap = VideoCapture(argv[1]);
        WIDTH  = cap.get(CAP_PROP_FRAME_WIDTH);
        HEIGHT = cap.get(CAP_PROP_FRAME_HEIGHT);
        if (argc == 3) {
            for (int i=0; i<atoi(argv[2]); i++) {
                cap.grab();
            }
        }
    }
    // if (argc == 4) {
    //     cap = VideoCapture(0);
    //     WIDTH  = atoi(argv[2]);
    //     HEIGHT = atoi(argv[3]);
    //     cap.set(CAP_PROP_FRAME_WIDTH,  WIDTH);
    //     cap.set(CAP_PROP_FRAME_HEIGHT, HEIGHT);
    // }

    double f = 0.4;
    double data[]= {f*WIDTH,  0.0,  WIDTH*0.5,  0.0, f*HEIGHT, HEIGHT*0.5, 0.0, 0.0, 1.0};
    Mat K(3, 3, CV_64F, data);
    Mat F, R, T, rod, mask;
    Mat img0, img1, img2, img1g, img2g, imgMatches, E, rodOld;

    cap >> img1;
    cap >> img2;
    cv::cvtColor(img1, img1g, CV_BGR2GRAY);
    cv::cvtColor(img2, img2g, CV_BGR2GRAY);
    if (showMatches) {
        namedWindow("Matches", WINDOW_NORMAL);
    }
    waitKey(1);
    if (showVideo) {
        namedWindow("Video", WINDOW_NORMAL);
    }
    waitKey(1);
    resizeWindow("Matches", 1920/2, 540/2);
    resizeWindow("Video", 960, 540);
    moveWindow("Matches", 0, 540+55);
    moveWindow("Video", 0, 0);
    waitKey(1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    vector<KeyPoint> keypoints0, keypoints1, keypoints2;
    vector<DMatch> goodMatches;
    vector<Point2f> p1, p2; // Point correspondences for recovering pose.
    int key = -1;
    clock_t timer, timer2;
    float time;

    gpuFacade gpu(maxKP, WIDTH, HEIGHT);
    FAST(img1g, keypoints1, threshold);
    gpu.LATCH(img1g, gpu.d_D1, &(gpu.numKP1), &keypoints1);
    FAST(img2g, keypoints2, threshold); // This call to fast is concurrent with above execution.
    gpu.LATCH(img2g, gpu.d_D2, &(gpu.numKP2), &keypoints2);
    gpu.match(gpu.d_D1, gpu.d_D2, gpu.numKP1, gpu.numKP2, gpu.d_M1, matchThreshold, gpu.streamKP1);
    gpu.match(gpu.d_D2, gpu.d_D1, gpu.numKP2, gpu.numKP1, gpu.d_M2, matchThreshold, gpu.streamKP2);
    gpu.getResults(gpu.h_M1, gpu.d_M1);
    gpu.getResults(gpu.h_M2, gpu.d_M2);
    for (int i=0; i<gpu.numKP1; i++) {
        if (gpu.h_M1[i] >= 0 && gpu.h_M1[i] < gpu.numKP2 && gpu.h_M2[gpu.h_M1[i]] == i) {
            goodMatches.push_back( DMatch(i, gpu.h_M1[i], 0)); // For drawing.
            p1.push_back(keypoints1[i].pt); // For recovering pose.
            p2.push_back(keypoints2[gpu.h_M1[i]].pt);
        }
    }

    img1.copyTo(img0);
    img2.copyTo(img1);
    cap.read(img2);
    cvtColor(img2, img2g, CV_BGR2GRAY);

    keypoints0 = keypoints1;
    keypoints1 = keypoints2;

    gpu.uIntSwapPointer = gpu.d_D1;
    gpu.d_D1 = gpu.d_D2;
    gpu.d_D2 = gpu.uIntSwapPointer;

    gpu.numKP0 = gpu.numKP1;
    gpu.numKP1 = gpu.numKP2;

    FAST(img2g, keypoints2, threshold);
    int loopIteration = 0;
    for (; loopIteration < maxLoops || maxLoops == -1; loopIteration++) { // Main Loop.
        { // GPU code for descriptors and matching.
            cudaEventRecord(gpu.start, 0);
            gpu.LATCH(img2g, gpu.d_D2, &(gpu.numKP2), &keypoints2);
            gpu.match(gpu.d_D1, gpu.d_D2, gpu.numKP1, gpu.numKP2, gpu.d_M1, matchThreshold, gpu.streamKP1);
            gpu.match(gpu.d_D2, gpu.d_D1, gpu.numKP2, gpu.numKP1, gpu.d_M2, matchThreshold, gpu.streamKP2);
            cudaEventRecord(gpu.stop, 0);
        }
        timer = clock();
        { // Put as much CPU code here as possible.
            { // Display matches and/or video to user.
                bool needToDraw = false;
                if (showMatches && loopIteration % showMatchesInterval == 0) { // Draw matches.
                    drawMatches( img0, keypoints0, img1, keypoints1,
                        goodMatches, imgMatches, Scalar::all(-1), Scalar::all(-1),
                        vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
                    imshow( "Matches", imgMatches );
                    needToDraw = true;
                }
                if (showVideo && loopIteration % showVideoInterval == 0) {
                    imshow("Video", img1);
                    needToDraw = true;
                }
                if (needToDraw) {
                    key = waitKey(1);
                }
            }
            { // Handle user input.
                switch (key) {
                    case (-1):
                    break;
                    case (1048689): // q
                    case (113): // also q
                        return 0;
                    break;
                    case (1048695): // w
                        waitKey(0);
                    break;
                    case (1114027): // +
                        skipFrames++;
                        cerr << "For each processed frame we are now skipping " << skipFrames << endl;
                    break;
                    case (1114029): // -
                        skipFrames = max(1, --skipFrames);
                        cerr << "For each processed frame we are now skipping " << skipFrames << endl;
                    break;
                    default:
                        cerr << "Currently pressed key is:   " << key << endl;
                    break;
                }
                key = -1;
            }
            { // Iterate the "logical" loop (get ready to process next frame)
                img1.copyTo(img0);
                img2.copyTo(img1);
                for (int i=0; i<skipFrames; i++) {
                    cap.grab();
                }
                cap.read(img2);
                if (img2.cols == 0) break;
                cvtColor(img2, img2g, CV_BGR2GRAY);

                keypoints0 = keypoints1;
                keypoints1 = keypoints2;

                gpu.uIntSwapPointer = gpu.d_D1;
                gpu.d_D1 = gpu.d_D2;
                gpu.d_D2 = gpu.uIntSwapPointer;

                gpu.numKP0 = gpu.numKP1;
                gpu.numKP1 = gpu.numKP2;
            }
            { // Solve for and output rotation vector (this gets piped to feedgnuplot).
                if (10 < p1.size() && 10 < p2.size()) {
                    E = findEssentialMat(p1, p2, f*WIDTH, Point2d(WIDTH*0.5f, HEIGHT*0.5f), RANSAC, 0.999, 3.0, mask);
                    int inliers = 0;
                    for (int i=0; i<mask.rows; i++) {
                        inliers += mask.data[i];
                    }
                    totalInliers += inliers;
                    double size = p1.size();
                    double r = inliers/max((double)size, 150.0);
                    r = 1.0 - min(r + 0.05, 1.0);
                    defect += r*r;
                    if (gnuplot) {
                        cout << "11:" << r*r << endl;
                    }

                    // recoverPose(E, p1, p2, R, T, f*WIDTH, Point2d(WIDTH*0.5f, HEIGHT*0.5f), mask);
                    // Rodrigues(R, rod);
                    // if (loopIteration==0) {
                    //     rod.copyTo(rodOld);
                    // }
                    // if (dist2(rod, rodOld) < 1.0) {
                    //     rod.copyTo(rodOld);
                    // } else {
                    //     cerr << "Rejecting the recovered pose: " << rod.t() * 57.2957795 << endl;
                    //     // This commented out chunk of code is good for webcams. If you initialize with a bad value it will recover.
                    //     // const double alpha = 0.1; // Move our region of acceptable responses (only a little) closer to the observed (but presumed erroneous) value.
                    //     // for (int i=0; i<3; i++) {
                    //     //     rodOld.at<double>(i) = rodOld.at<double>(i)*(1.0-alpha) + rod.at<double>(i)*alpha;
                    //     // }
                    //     rodOld.copyTo(rod);
                    // }
                } else {
                    defect += 1.0;
                    cout << "11:" << 1.0 << endl;
                    cerr << "Too few matches! Not going to try to recover pose this frame." << endl;
                }
                // To prevent the graphs from desynchronizing from each other, we have to output this unconditionally.
                if (gnuplot) {
                    for (int i=0; i<3; i++) {
                        // cout << i << ":" << rod.at<double>(i) * 57.2957795 << endl; // Output Rodrigues vector, rescaled to degrees
                    }
                    // T is unit norm (scale-less) and often erroneously sign-reversed.
                    // if (T.at<double>(2) < 0) T = -T; // Assume dominate motion is forward... (this is not an elegant assumption)
                    // double theta = atan2(T.at<double>(0), T.at<double>(2));
                    // double phi = atan2(T.at<double>(1), T.at<double>(2));
                    // cout << 3 << ":" << theta * 57.2957795 << endl; // Plot polar translation angle
                    // cout << 4 << ":" << phi * 57.2957795 << endl; // Plot azimuthal translation angle
                }
            }
            { // run FAST detector on the CPU for next frame (get ready for next loop iteration).
                FAST(img2g, keypoints2, threshold);
                // Apply proportional control to threshold to drive it towards targetKP.
                int control = (int)(((float)keypoints2.size() - (float)targetKP) / (float)tolerance);
                threshold += min(100, control);
                if (threshold < 1) threshold = 1;
            }
        }
        if (gnuplot) {
            time = (1000*(clock() - timer)/(double)CLOCKS_PER_SEC);
            cout << "9:" << time << endl; // Plot CPU time.
            timer = clock();
        }
        { // Get new GPU results
            p1.clear();
            p2.clear();
            goodMatches.clear();
            gpu.getResults(gpu.h_M1, gpu.d_M1);
            gpu.getResults(gpu.h_M2, gpu.d_M2);
            cudaEventElapsedTime(&time, gpu.start, gpu.stop);
            if (gnuplot) {
                cout << "10:" << (time+(1000*(clock() - timer)/(double)CLOCKS_PER_SEC)) << endl; // Plot total asynchronous GPU time.
            }
            checkLaunchError();
            for (int i=0; i<gpu.numKP0; i++) {
                if (gpu.h_M1[i] >= 0 && gpu.h_M1[i] < gpu.numKP1 && gpu.h_M2[gpu.h_M1[i]] == i) {
                    goodMatches.push_back( DMatch(i, gpu.h_M1[i], 0)); // For drawing matches.
                    p1.push_back(keypoints0[i].pt); // For recovering pose.
                    p2.push_back(keypoints1[gpu.h_M1[i]].pt);
                }
            }
        }
        if (gnuplot) {
            cout << "6:" << gpu.numKP1 << endl; // Plot number of keypoints.
            cout << "7:" << p1.size() << endl; // Plot number of matches.
            cout << "8:" << 100*threshold << endl; // Plot current threshold for FAST.
        }
        totalMatches += p1.size();
    }
    cerr << "Total matches: " << totalMatches << endl;
    cerr << "Total inliers: " << totalInliers << endl;
    cerr << "Defect: " << defect << endl;
    cerr << "Loop iteration: " << loopIteration << endl;
    return 0;
}
