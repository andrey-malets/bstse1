
#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include <shrUtils.h>
#include "MersenneTwister.h"

#include <stdint.h>

/////////////////////////////////////////////////////////////////////////////////////
// Public constants

const unsigned WarpStandard_K=32;
const unsigned WarpStandard_REG_COUNT=3;
const unsigned WarpStandard_STATE_WORDS=32;

const uint32_t WarpStandard_TEST_DATA[WarpStandard_STATE_WORDS]={
	0x8cf35fea, 0xe1dd819e, 0x4a7d0a8e, 0xe0c05911, 0xfd053b8d, 0x30643089, 0x6f6ac111, 0xc4869595,
	0x9416b7be, 0xe6d329e8, 0x5af0f5bf, 0xc5c742b5, 0x7197e922, 0x71aa35b4, 0x2070b9d1, 0x2bb34804,
	0x7754a517, 0xe725315e, 0x7f9dd497, 0x043b58bf, 0x83ffa33d, 0x2532905a, 0xbdfe0c8a, 0x16f68671,
	0x0d14da2e, 0x847efd5f, 0x1edeec64, 0x1bebdf9b, 0xf74d4ff3, 0xd404774b, 0x8ee32599, 0xefe0c405
};

//////////////////////////////////////////////////////////////////////////////////////
// Private constants

const char *WarpStandard_name="WarpRNG[CorrelatedU32Rng;k=32;g=16;rs=0;w=32;n=1024;hash=deac2e12ec6e615]";
const char *WarpStandard_post_processing="addtaps";
const unsigned WarpStandard_N=1024;
const unsigned WarpStandard_W=32;
const unsigned WarpStandard_G=16;
const unsigned WarpStandard_SR=0;
__device__ const unsigned WarpStandard_Q[2][32]={
  {29,24,5,23,14,26,11,31,9,3,1,28,0,2,22,20,18,15,27,13,10,16,8,17,25,12,19,30,7,6,4,21},
  {5,14,28,24,19,13,0,17,11,20,7,10,6,15,2,9,8,23,4,30,12,25,3,21,26,27,31,18,22,16,29,1}
};
const unsigned WarpStandard_Z0=2;
__device__ const unsigned WarpStandard_Z1[32]={
  0,1,0,1,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,1,1,0,1};

const unsigned WarpStandard_SHMEM_WORDS=32;
const unsigned WarpStandard_GMEM_WORDS=0;

////////////////////////////////////////////////////////////////////////////////////////
// Public functions

__device__ void WarpStandard_LoadState(const unsigned *seed, unsigned *regs, unsigned *shmem)
{
  unsigned offset=threadIdx.x % 32;  unsigned base=threadIdx.x-offset;
  // setup constants
  regs[0]=WarpStandard_Z1[offset];
  regs[1]=base + WarpStandard_Q[0][offset];
  regs[2]=base + WarpStandard_Q[1][offset];
  // Setup state
  unsigned stateOff=blockDim.x * blockIdx.x * 1 + threadIdx.x * 1;
  shmem[threadIdx.x]=seed[stateOff];
}

__device__ void WarpStandard_SaveState(const unsigned *regs, const unsigned *shmem, unsigned *seed)
{
  unsigned stateOff=blockDim.x * blockIdx.x * 1 + threadIdx.x * 1;
  seed[stateOff] = shmem[threadIdx.x];
}

__device__ unsigned WarpStandard_Generate(unsigned *regs, unsigned *shmem)
{
#if __DEVICE_EMULATION__
    __syncthreads();
#endif
  unsigned t0=shmem[regs[1]], t1=shmem[regs[2]];
  unsigned res=(t0<<WarpStandard_Z0) ^ (t1>>regs[0]);
#if __DEVICE_EMULATION__
    __syncthreads();
#endif
  shmem[threadIdx.x]=res;
  return t0+t1;
};












__global__ void init(float *matrix, int size)
{
	// Здесь можно сделать инициализацию
	matrix[threadIdx.x + threadIdx.y*size] = 1;
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

__device__ void step1(float *src_v, float *src_w, float *dst_v, float *dst_w, float *stats, int size, float c1, float dt, float D, float M, float R1, float R2, int i)
{
    int x = threadIdx.x;

	dst_v[x] =
		(src_v[x]
			+ c1 * dt * (src_v[(x - 1 + size) % size] + src_v[(x + 1 + size) % size])
			+ src_w[x] * dt)
				/ (1 + src_w[x] * src_w[x] + D * dt)
		
		+ R1 * (__powf(dt, 0.5)) * M;

	dst_w[x] = (src_w[x] + src_v[x] * dt) / (1 + src_v[x] * src_v[x] * dt) + R2 * (__powf(dt, 0.5)) * M;

	stats[size*x + i] = dst_v[x];
	
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
        MT[i].seed = seed+i;
    }
	int z= 1;
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



__global__ void RandomGPU2(unsigned *state, int nPerRng, float *src_v, float *src_w, float *dst_v, float *dst_w, float *stats, int size, float c1, float c2, float dt, float D, float M)
	{	

       extern __shared__ unsigned sharedMemory[];

		
		 int iOut;
		 float x1, x2;
		 unsigned *rngShmem = sharedMemory;
				
		 unsigned rngRegs[WarpStandard_REG_COUNT];
		 WarpStandard_LoadState(state, rngRegs, rngShmem);
		  for (iOut = 0; iOut < nPerRng; iOut++) 
		  
		  {
			 unsigned u=WarpStandard_Generate(rngRegs, rngShmem);
			 if (iOut % 2 == 0)
				x1 = ((float)u + 1.0f) / 4294967296.0f;
			 else
			 {
				x2 = ((float)u + 1.0f) / 4294967296.0f;
				BoxMuller(x1, x2);

				if(iOut % 4 == 1)
					step1(src_v, src_w, dst_v, dst_w, stats, size, c1, dt, D, M, x1, x2, iOut / 2);
				else
					step1(dst_v, dst_w, src_v, src_w, stats, size, c1, dt, D, M, x1, x2, iOut / 2);
			 }
			 __syncthreads();
		  }


}
__global__ void RandomGPU(int nPerRng, float *src_v, float *src_w, float *dst_v, float *dst_w, float *stats, int size, float c1, float c2, float dt, float D, float M)
{
   // const int tid = threadIdx.x * size + threadIdx.y;
    const int tid = threadIdx.x;
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
				step1(src_v, src_w, dst_v, dst_w, stats, size, c1, dt, D, M, x1, x2, iOut / 2);
			else
				step1(dst_v, dst_w, src_v, src_w, stats, size, c1, dt, D, M, x1, x2, iOut / 2);
		}
		 __syncthreads();
    }
}


#endif // #ifndef _MATRIXMUL_KERNEL_H_
