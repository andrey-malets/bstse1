// Utilities and system includes
#include <shrUtils.h>
#include <iostream>
#include <fstream>
#include "cutil_inline.h"

// includes, kernels
#include <matrixMul_kernel.cu>

static char *sSDKsample = "Starting...";

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
    // таймер для оценки времени работы программы
	unsigned int timer = 0;
    cutilCheckError(cutCreateTimer(&timer));
    cutilCheckError(cutStartTimer(timer));

    // Параметры системы
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

	dim3 threadsPerBlock(size);
	int numBlocks = 1;
	init <<<numBlocks, threadsPerBlock>>>(d_v, size);
	init <<<numBlocks, threadsPerBlock>>>(d_w, size);

	const char *dat_path = shrFindFilePath("MersenneTwister.dat", argv[0]);
	loadMTGPU(dat_path);
	seedMTGPU(1001);

	RandomGPU<<<numBlocks, threadsPerBlock>>>(2*count, d_v, d_w, d_v2, d_w2, d_stats, size, 1, 1, 0.06, 2, 0.88);
	
	(cutStopTimer(timer));
     printf("Processing time: %f (ms)\n", cutGetTimerValue( timer));

	cutilCheckError( cutDeleteTimer( timer));
	cutilCheckError(cutCreateTimer(&timer));
    cutilCheckError(cutStartTimer(timer));
	
	
	//cutilSafeCall(cudaMemcpy(h_stats, d_stats, count*bsize, cudaMemcpyDeviceToHost));
	
	int count2 = count;
	for (int i=0; i != count2; ++i)
	{
    
	RandomGPU<<<numBlocks, threadsPerBlock>>>(2, d_v, d_w, d_v2, d_w2, d_stats, size, 1, 1, 0.06, 2, 0.88);
	
	}
    cutilSafeCall( cudaThreadSynchronize() );
    (cutStopTimer(timer));
    printf("Copy device to host: %f (ms)\n", cutGetTimerValue( timer));
    cutilCheckError( cutDeleteTimer( timer));
    cutilCheckError(cutCreateTimer(&timer));
    cutilCheckError(cutStartTimer(timer));



	{
		std::ofstream output("c:\\output2.txt");

		for(int j = 0; j != size; ++j)
		{
			for(int i = 0; i != count; ++i)
				output << h_stats[j*size+i] << "\t";
			output << std::endl;
		}
		

		


	}
    printf("Time of extracting data: %f (ms)\n", cutGetTimerValue( timer));
    cudaThreadExit();
}

