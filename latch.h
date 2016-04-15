void latch( unsigned char *,
            unsigned char *,
            size_t,
            float *,
            unsigned int *,
            int *,
            int,
            float *,
            vector<KeyPoint>*,
            const int,
            const int,
            float*,
            cudaEvent_t);

void loadPatchTriplets(cudaArray*);

void initImage(    unsigned char**,
                    int,
                    int,
                    size_t *
                );

void initMask(      float **,
                    float *);
