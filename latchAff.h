void latchAff( Mat,
            unsigned char *,
            size_t,
            float *,
            unsigned int *,
            int *,
            int,
            float *,
            vector<KeyPoint>*,
            float*,
            cudaEvent_t,
            Mat,
            RotatedRect);

void initPatchTriplets(cudaArray*);

void initImage(    unsigned char**,
                    int,
                    int,
                    size_t *
                );

void initMask(      float **,
                    float *);
