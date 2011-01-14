// Utilities and system includes
#include <shrUtils.h>
#include <iostream>
#include <fstream>
#include "cutil_inline.h"

// includes, kernels
#include <matrixMul_kernel.cu>

static char *sSDKsample = "matrixMul";

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);
void randomInit(float*, int);
void printDiff(float*, float*, int, int, int, float);

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

//__global__ void
//matrixMul( float* C, float* A, float* B, int wA, int wB)

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
	printf("[ %s ]\n", sSDKsample);

    shrSetLogFileName ("matrixMul.txt");
    shrLog("%s Starting...\n\n", argv[0]);

    runTest(argc, argv);

    shrEXIT(argc, (const char**)argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char** argv)
{
	cudaSetDevice(cutGetMaxGflopsDeviceId());

	size_t size = 32, size2 = size*size, bsize = size2 * sizeof(float), count = 2<<15, bcount = count * sizeof(float);
	
	float *h_v = new float[size2], *h_w = new float[size2], *h_stats = new float[count];

	float *d_v, *d_w, *d_v2, *d_w2, *d_stats;
	cutilSafeCall(cudaMalloc((void**) &d_v, bsize));
	cutilSafeCall(cudaMalloc((void**) &d_w, bsize));
	cutilSafeCall(cudaMalloc((void**) &d_v2, bsize));
	cutilSafeCall(cudaMalloc((void**) &d_w2, bsize));
	cutilSafeCall(cudaMalloc((void**) &d_stats, bcount));

	//cutilSafeCall(cudaMemcpy(d_v, h_v, bsize, cudaMemcpyHostToDevice));
	//cutilSafeCall(cudaMemcpy(d_w, h_w, bsize, cudaMemcpyHostToDevice));

	dim3 threadsPerBlock(size);
	int numBlocks = 1;
	init <<<numBlocks, threadsPerBlock>>>(d_v, size);
	init <<<numBlocks, threadsPerBlock>>>(d_w, size);

	const char *dat_path = shrFindFilePath("MersenneTwister.dat", argv[0]);
	loadMTGPU(dat_path);

	seedMTGPU(1001);
    // __device__ void step(кол во итераций, float *src_v, float *src_w, float *dst_v, float *dst_w, float *stats, int size, float c1, float c2, float dt, float D, float M, float R1, float R2, int i)

	RandomGPU<<<numBlocks, threadsPerBlock>>>(2*count, d_v, d_w, d_v2, d_w2, d_stats, size, 1, 1, 0.06, 2, 0.88);
	cutilSafeCall(cudaMemcpy(h_stats, d_stats, bcount, cudaMemcpyDeviceToHost));

	{
		std::ofstream output("c:\\output2.txt");

		for(int i = 0; i != count; ++i)
			output << h_stats[i] << "\t";
	}

    cudaThreadExit();
}

