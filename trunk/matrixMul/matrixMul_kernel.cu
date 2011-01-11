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
#include "matrixMul.h"

#define CHECK_BANK_CONFLICTS 0
#if CHECK_BANK_CONFLICTS
#define AS(i, j) cutilBankChecker(((float*)&As[0][0]), (BLOCK_SIZE * i + j))
#define BS(i, j) cutilBankChecker(((float*)&Bs[0][0]), (BLOCK_SIZE * i + j))
#else
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
#endif

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
////////////////////////////////////////////////////////////////////////////////
__global__ void
matrixMul( float* C, float* A, float* B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        AS(ty, tx) = A[a + wA * ty + tx];
        BS(ty, tx) = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += AS(ty, k) * BS(k, tx);

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_


////Скопировали необходимый функионал для генерации случайных чсел с нормальным распеределением на GPU из Mersentwisterkernel
//
////#include <shrUtils.h>
//#include "MersenneTwister.h"
//
//
//
//__device__ static mt_struct_stripped ds_MT[MT_RNG_COUNT];
//static mt_struct_stripped h_MT[MT_RNG_COUNT];
//
//
////Load twister configurations
//void loadMTGPU(const char *fname){
//    FILE *fd = fopen(fname, "rb");
//    if(!fd){
//        shrLog("initMTGPU(): failed to open %s\n", fname);
//        shrLog("FAILED\n");
//        exit(0);
//    }
//    if( !fread(h_MT, sizeof(h_MT), 1, fd) ){
//        shrLog("initMTGPU(): failed to load %s\n", fname);
//        shrLog("FAILED\n");
//        exit(0);
//    }
//    fclose(fd);
//}
//
////Initialize/seed twister for current GPU context
//void seedMTGPU(unsigned int seed){
//    int i;
//    //Need to be thread-safe
//    mt_struct_stripped *MT = (mt_struct_stripped *)malloc(MT_RNG_COUNT * sizeof(mt_struct_stripped));
//
//    for(i = 0; i < MT_RNG_COUNT; i++){
//        MT[i]      = h_MT[i];
//        MT[i].seed = seed;
//    }
//    CUDA_SAFE_CALL( cudaMemcpyToSymbol(ds_MT, MT, sizeof(h_MT)) );
//
//    free(MT);
//}
//
//
//////////////////////////////////////////////////////////////////////////////////
//// Write MT_RNG_COUNT vertical lanes of nPerRng random numbers to *d_Random.
//// For coalesced global writes MT_RNG_COUNT should be a multiple of warp size.
//// Initial states for each generator are the same, since the states are
//// initialized from the global seed. In order to improve distribution properties
//// on small NPerRng supply dedicated (local) seed to each twister.
//// The local seeds, in their turn, can be extracted from global seed
//// by means of any simple random number generator, like LCG.
//////////////////////////////////////////////////////////////////////////////////
//__global__ void RandomGPU(
//    float *d_Random,
//    int nPerRng)
//{
//    const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
//
//    int iState, iState1, iStateM, iOut;
//    unsigned int mti, mti1, mtiM, x;
//    unsigned int mt[MT_NN], matrix_a, mask_b, mask_c; 
//
//    //Load bit-vector Mersenne Twister parameters
//    matrix_a = ds_MT[tid].matrix_a;
//    mask_b = ds_MT[tid].mask_b;
//    mask_c = ds_MT[tid].mask_c;
//
//    //Initialize current state
//    mt[0] = ds_MT[tid].seed;
//    for (iState = 1; iState < MT_NN; iState++)
//        mt[iState] = (1812433253U * (mt[iState - 1] ^ (mt[iState - 1] >> 30)) + iState) & MT_WMASK;
//
//    iState = 0;
//    mti1 = mt[0];
//    for (iOut = 0; iOut < nPerRng; iOut++) {
//        iState1 = iState + 1;
//        iStateM = iState + MT_MM;
//        if(iState1 >= MT_NN) iState1 -= MT_NN;
//        if(iStateM >= MT_NN) iStateM -= MT_NN;
//        mti  = mti1;
//        mti1 = mt[iState1];
//        mtiM = mt[iStateM];
//
//        // MT recurrence
//        x    = (mti & MT_UMASK) | (mti1 & MT_LMASK);
//        x    =  mtiM ^ (x >> 1) ^ ((x & 1) ? matrix_a : 0);
//
//        mt[iState] = x;
//        iState = iState1;
//
//        //Tempering transformation
//        x ^= (x >> MT_SHIFT0);
//        x ^= (x << MT_SHIFTB) & mask_b;
//        x ^= (x << MT_SHIFTC) & mask_c;
//        x ^= (x >> MT_SHIFT1);
//
//        //Convert to (0, 1] float and write to global memory
//        d_Random[tid + iOut * MT_RNG_COUNT] = ((float)x + 1.0f) / 4294967296.0f;
//    }
//}
//
//
//
//////////////////////////////////////////////////////////////////////////////////
//// Transform each of MT_RNG_COUNT lanes of nPerRng uniformly distributed 
//// random samples, produced by RandomGPU(), to normally distributed lanes
//// using Cartesian form of Box-Muller transformation.
//// nPerRng must be even.
//////////////////////////////////////////////////////////////////////////////////
//#define PI 3.14159265358979f
//__device__ inline void BoxMuller(float& u1, float& u2){
//    float   r = sqrtf(-2.0f * logf(u1));
//    float phi = 2 * PI * u2;
//    u1 = r * __cosf(phi);
//    u2 = r * __sinf(phi);
//}
//
//__global__ void BoxMullerGPU(float *d_Random, int nPerRng){
//    const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
//
//    for (int iOut = 0; iOut < nPerRng; iOut += 2)
//        BoxMuller(
//                d_Random[tid + (iOut + 0) * MT_RNG_COUNT],
//                d_Random[tid + (iOut + 1) * MT_RNG_COUNT]
//                );
//}
