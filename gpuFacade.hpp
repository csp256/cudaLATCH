class gpuFacade {
    public:
        int WIDTH, HEIGHT;
        void set_values (int,int);
        int area();

        int maxKP;
        cudaEvent_t start, stop;
        vector<KeyPoint> keypoints0, keypoints1, keypoints2;
        vector<DMatch> goodMatches;
        vector<Point2f> p1, p2; // Point correspondences for recovering pose.
        int numKP0, numKP1, numKP2; // The actual number of keypoints we are dealing with: just keypoints#.size(), but capped at maxKP.
        size_t sizeK; // K for keypoints
        size_t sizeI; // I for Image
        size_t sizeD; // D for Descriptor
        size_t sizeM; // M for Matches
        size_t sizeMask;
        float *h_K1, *h_K2;
        // For reasons opaque to me, allocating both (but not either) h_M1 or h_M2
        // with cudaMallocHost segfaults, apparently after graceful exit? So neither of them are pinned.
        int* h_M1;
        int* h_M2;
        float h_mask[64];
        unsigned char *d_I;
        unsigned int *d_D1, *d_D2, *uIntSwapPointer;
        int *d_M1, *d_M2;
        float *d_K, *d_mask;
        cudaArray* patchTriplets;
        size_t pitch;
        cudaEvent_t latchFinished;
        cudaStream_t streamKP1, streamKP2;

        void LATCH(cv::Mat, unsigned int*, int*, std::vector<cv::KeyPoint>*);
        void match( unsigned int*,
                    unsigned int*,
                    int,
                    int,
                    int*,
                    int,
                    cudaStream_t);
        void getResults(int* h_matches, int* d_matches);
        gpuFacade(int, int, int, int);
        ~gpuFacade();
};
