/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include <shrUtils.h>
#include "MersenneTwister.h"

__global__ void init(float *matrix, int size)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

	// Здесь можно сделать инициализацию
	matrix[tx*size + ty] = 1;
}

__device__ void step(float *src_v, float *src_w, float *dst_v, float *dst_w, float *stats, int size, float c1, float c2, float dt, float D, float M, float R1, float R2, int i)
{
    int x = threadIdx.x;
    int y = threadIdx.y;
	
	dst_v[x*size+y] =
		(src_v[x*size+y]
			+ c1 * dt * (src_v[((x - 1 + size) % size)*size + y] + src_v[((x + 1 + size) % size)*size + y])
			+ c2 * dt * (src_v[x*size + ((y - 1 + size) % size)] + src_v[x*size + ((y + 1 + size) % size)])
			+ src_w[x*size+y] * dt)
				/ (1 + src_w[x*size+y] * src_w[x*size+y] + D * dt)
		
		+ R1 * (__powf(dt, 0.5)) * M;

	dst_w[x*size+y] = (src_w[x*size+y] + src_v[x*size+y] * dt) / (1 + src_v[x*size+y] * src_v[x*size+y] * dt) + R2 * (__powf(dt, 0.5)) * M;
	if(x == 0 && y == 0)
		stats[i] = dst_v[x*size+y];
}

////Скопировали необходимый функионал для генерации случайных чсел с нормальным распеределением на GPU из Mersentwisterkernel

__device__ static mt_struct_stripped ds_MT[MT_RNG_COUNT];
static mt_struct_stripped h_MT[MT_RNG_COUNT];

//Load twister configurations
void loadMTGPU(const char *fname){
    FILE *fd = fopen(fname, "rb");
    if(!fd){
        shrLog("initMTGPU(): failed to open %s\n", fname);
        shrLog("FAILED\n");
        exit(0);
    }
    if( !fread(h_MT, sizeof(h_MT), 1, fd) ){
        shrLog("initMTGPU(): failed to load %s\n", fname);
        shrLog("FAILED\n");
        exit(0);
    }
    fclose(fd);
}

//Initialize/seed twister for current GPU context
void seedMTGPU(unsigned int seed){
    int i;
    //Need to be thread-safe
    mt_struct_stripped *MT = (mt_struct_stripped *)malloc(MT_RNG_COUNT * sizeof(mt_struct_stripped));

    for(i = 0; i < MT_RNG_COUNT; i++){
        MT[i]      = h_MT[i];
        MT[i].seed = seed;
    }
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(ds_MT, MT, sizeof(h_MT)) );

    free(MT);
}

#define PI 3.14159265358979f
__device__ inline void BoxMuller(float& u1, float& u2){
    float   r = sqrtf(-2.0f * logf(u1));
    float phi = 2 * PI * u2;
    u1 = r * __cosf(phi);
    u2 = r * __sinf(phi);
}

__global__ void RandomGPU(int nPerRng, float *src_v, float *src_w, float *dst_v, float *dst_w, float *stats, int size, float c1, float c2, float dt, float D, float M)
{
    const int tid = threadIdx.x * size + threadIdx.y;

    int iState, iState1, iStateM, iOut;
    unsigned int mti, mti1, mtiM, x;
    unsigned int mt[MT_NN], matrix_a, mask_b, mask_c; 
	float x1, x2;

    //Load bit-vector Mersenne Twister parameters
    matrix_a = ds_MT[tid].matrix_a;
    mask_b = ds_MT[tid].mask_b;
    mask_c = ds_MT[tid].mask_c;

    //Initialize current state
    mt[0] = ds_MT[tid].seed;
    for (iState = 1; iState < MT_NN; iState++)
        mt[iState] = (1812433253U * (mt[iState - 1] ^ (mt[iState - 1] >> 30)) + iState) & MT_WMASK;

    iState = 0;
    mti1 = mt[0];
    for (iOut = 0; iOut < nPerRng; iOut++) {
        iState1 = iState + 1;
        iStateM = iState + MT_MM;
        if(iState1 >= MT_NN) iState1 -= MT_NN;
        if(iStateM >= MT_NN) iStateM -= MT_NN;
        mti  = mti1;
        mti1 = mt[iState1];
        mtiM = mt[iStateM];

        // MT recurrence
        x    = (mti & MT_UMASK) | (mti1 & MT_LMASK);
        x    =  mtiM ^ (x >> 1) ^ ((x & 1) ? matrix_a : 0);

        mt[iState] = x;
        iState = iState1;

        //Tempering transformation
        x ^= (x >> MT_SHIFT0);
        x ^= (x << MT_SHIFTB) & mask_b;
        x ^= (x << MT_SHIFTC) & mask_c;
        x ^= (x >> MT_SHIFT1);

        //Convert to (0, 1] float and write to global memory
		// d_Random[tid + iOut * MT_RNG_COUNT] = ((float)x + 1.0f) / 4294967296.0f;

		if (iOut % 2 == 0)
			x1 = ((float)x + 1.0f) / 4294967296.0f;
		else
		{
			x2 = ((float)x + 1.0f) / 4294967296.0f;
			BoxMuller(x1, x2);

			if(iOut % 4 == 1)
				step(src_v, src_w, dst_v, dst_w, stats, size, c1, c2, dt, D, M, x1, x2, iOut / 2);
			else
				step(dst_v, dst_w, src_v, src_w, stats, size, c1, c2, dt, D, M, x1, x2, iOut / 2);
		}
    }
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
