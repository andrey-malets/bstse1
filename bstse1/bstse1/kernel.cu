
#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include <shrUtils.h>

/////////////////////////////////////////////////////////////////////////////////////
// Public constants

const unsigned WarpStandard_K=32;
const unsigned WarpStandard_REG_COUNT=3;
const unsigned WarpStandard_STATE_WORDS=32;

//////////////////////////////////////////////////////////////////////////////////////
// Private constants

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
	unsigned offset = (threadIdx.x + blockDim.x * threadIdx.y) % 32, base = threadIdx.x + blockDim.x * threadIdx.y - offset;
	// setup constants
	regs[0]=WarpStandard_Z1[offset];
	regs[1]=base + WarpStandard_Q[0][offset];
	regs[2]=base + WarpStandard_Q[1][offset];
	// Setup state
	unsigned stateOff =
		blockDim.x * blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y)
		+ threadIdx.x + blockDim.x * threadIdx.y;
	shmem[threadIdx.x + blockDim.x * threadIdx.y] = seed[stateOff];
}

__device__ void WarpStandard_SaveState(const unsigned *regs, const unsigned *shmem, unsigned *seed)
{
	unsigned stateOff =
		blockDim.x * blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y)
		+ threadIdx.x + blockDim.x * threadIdx.y;
	seed[stateOff] = shmem[threadIdx.x*blockDim.x + threadIdx.y];
}

__device__ unsigned WarpStandard_Generate(unsigned *regs, unsigned *shmem)
{
	__syncthreads();
	unsigned t0=shmem[regs[1]], t1=shmem[regs[2]];
	unsigned res=(t0<<WarpStandard_Z0) ^ (t1>>regs[0]);
	__syncthreads();
	shmem[threadIdx.x + blockDim.x * threadIdx.y] = res;
	return t0+t1;
};

__global__ void init(float *matrix)
{
	// Здесь можно сделать инициализацию
	matrix[blockDim.y * blockDim.x * blockIdx.x + threadIdx.y * blockDim.x + threadIdx.x] = 0.2;
}

__device__ void step(float *stats, int count,int i, float *src_v, float *src_w, float *dst_v, float *dst_w,  float c1, float c2, float dt, float D, float M, float R1, float R2)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x, size = blockDim.x * gridDim.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	dst_v[x+y*size] =
		(src_v[x+y*size]
			+ c1 * dt * (src_v[((x - 1 + size) % size) + y*size] + src_v[((x + 1 + size) % size) + y*size])
			+ c2 * dt * (src_v[x + ((y - 1 + size) % size)*size] + src_v[x + ((y + 1 + size) % size)*size])
			+ src_w[x+y*size] * dt)
				/ (1 + src_w[x+y*size] * src_w[x+y*size] + D * dt)
		
		+ R1 * (__powf(dt, 0.5)) * M;

	dst_w[x+y*size] = (src_w[x+y*size] + src_v[x+y*size] * dt) / (1 + src_v[x+y*size] * src_v[x+y*size] * dt) + R2 * (__powf(dt, 0.5)) * M;
	if(y == 11)
		stats[count * x + i] =  dst_v[x+y*size];
}

__device__ void step1(float *stats, int count, int i, float *src_v, float *src_w, float *dst_v, float *dst_w, float c1, float dt, float D, float M, float R1, float R2)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x, size = blockDim.x * gridDim.x;

	dst_v[x] =
		(src_v[x]
			+ c1 * dt * (src_v[(x - 1 + size) % size] + src_v[(x + 1 + size) % size])
			+ src_w[x] * dt)
				/ (1 + src_w[x] * src_w[x] + D * dt)
		
		+ R1 * (__powf(dt, 0.5)) * M;

	dst_w[x] = (src_w[x] + src_v[x] * dt) / (1 + src_v[x] * src_v[x] * dt) + R2 * (__powf(dt, 0.5)) * M;

	stats[count * x + i] = dst_v[x];
}

#define PI 3.14159265358979f
__device__ inline void BoxMuller(float& u1, float& u2){
    float   r = sqrtf(-2.0f * logf(u1));
    float phi = 2 * PI * u2;
    u1 = r * __cosf(phi);
    u2 = r * __sinf(phi);
}

__global__ void RandomGPU2(unsigned *state, int count, float *stats, float *src_v, float *src_w, float *dst_v, float *dst_w, float c1, float c2, float dt, float D, float M)
{	
	__shared__ unsigned sharedMemory[1024];

	unsigned rngRegs[WarpStandard_REG_COUNT];
	WarpStandard_LoadState(state, rngRegs, sharedMemory);

	for (int iOut = 0; iOut < count; iOut++) 
	{
		unsigned
			n1 = WarpStandard_Generate(rngRegs, sharedMemory),
			n2 = WarpStandard_Generate(rngRegs, sharedMemory);

		float 
			x1 = ((float)n1 + 1.0f) / 4294967296.0f,
			x2 = ((float)n2 + 1.0f) / 4294967296.0f;

		BoxMuller(x1, x2);

		if(iOut % 2 == 0)
				step(stats, count, iOut, src_v, src_w, dst_v, dst_w, c1, c2, dt, D, M, x1, x2);
		else
				step(stats, count, iOut, dst_v, dst_w, src_v, src_w, c1, c2, dt, D, M, x1, x2);

		__syncthreads();
	}
}
__global__ void ComplexAbs (cufftComplex *idata, float *odata, unsigned s)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	
		odata[i] = __powf(__powf(((float)idata[i].x*(float)idata[i].x + (float)idata[i].y*(float)idata[i].y), 0.5),2);
	
		//odata[i] = idata[i].x;
	

}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
