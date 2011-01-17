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


int main(int argc, char** argv)
{
	printf("[ %s ]\n", sSDKsample);
/*
    shrSetLogFileName ("matrixMul.txt");
    shrLog("%s Starting...\n\n", argv[0]);*/

    runTest(argc, argv);

   /* shrEXIT(argc, (const char**)argv);*/
}


void runTest(int argc, char** argv)
{
	cudaSetDevice(cutGetMaxGflopsDeviceId());

//	unsigned int timer = 0;
 //   cutilCheckError(cutCreateTimer(&timer));
//    cutilCheckError(cutStartTimer(timer));


	size_t
		// Размер строки в одномерном случае (в элементах)
		size = 1024,
		// Размер матрицы в двумерном случае (в элементах)
		size2 = size*size,
		// Размер матрицы в одномерном случае (в байтах)
		bsize = size * sizeof(float),
		// Размер матрицы в двумерном случае (в байтах)
		bsize2 = size2 * sizeof(float),
		// Количество итераций по времени
		count = 2<<15;



	float *h_v = new float[size], *h_w = new float[size], *h_stats = new float[count*size];

	float *d_v, *d_w, *d_v2, *d_w2, *d_stats;
	cutilSafeCall(cudaMalloc((void**) &d_v, bsize));
	cutilSafeCall(cudaMalloc((void**) &d_w, bsize));
	cutilSafeCall(cudaMalloc((void**) &d_v2, bsize));
	cutilSafeCall(cudaMalloc((void**) &d_w2, bsize));

	cutilSafeCall(cudaMalloc((void**) &d_stats, count*bsize));

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
	cutilSafeCall(cudaMemcpy(h_stats, d_stats, count*bsize, cudaMemcpyDeviceToHost));











 

 //  cutilCheckError(cutStopTimer(timer));
//   double dSeconds = cutGetTimerValue(timer)/((double)count * 1000.0);
 ////   double dNumOps = 2.0 * (double)uiWA * (double)uiHA * (double)uiWB;
 ////   double gflops = 1.0e-9 * dNumOps/dSeconds;

 ////   //Log througput, etc
 ////   shrLogEx(LOGBOTH | MASTER, 0, "matrixMul, Throughput = %.4f GFlop/s, Time = %.5f s, Size = %.0f Ops, NumDevsUsed = %d, Workgroup = %u\n", 
 ////           gflops, dSeconds, dNumOps, 1, threads.x * threads.y);
  //   cutilCheckError(cutDeleteTimer(timer));
	
// printf("[ %s ]\n", dSeconds);





	/*{
		std::ofstream output("c:\\output2.txt");

		for(int j = 0; j != size; ++j)
		{
			for(int i = 0; i != count; ++i)
				output << h_stats[j*size+i] << "\t";
			output << std::endl;
		}
		

		


	}*/

    cudaThreadExit();
}

