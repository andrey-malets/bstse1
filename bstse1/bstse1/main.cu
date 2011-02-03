// Utilities and system includes
#include <shrUtils.h>
#include <iostream>
#include <fstream>
#include <cutil_inline.h>
#include <stdio.h>
#include <cufft.h>
#include <time.h>


// includes, kernels
#include "kernel.cu"

static char *sSDKsample = "Starting...";

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);
void randomInit(float*, int);
void printDiff(float*, float*, int, int, int, float);

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

// генератор массива случайных чисел для начального seed'a warp_standart'a

#define znew   (z=36969*(z&65535)+(z>>16))
#define wnew   (w=18000*(w&65535)+(w>>16))
#define MWC    ((znew<<16)+wnew )
#define SHR3  (jsr^=(jsr<<17), jsr^=(jsr>>13), jsr^=(jsr<<5))
#define CONG  (jcong=69069*jcong+1234567)
#define FIB   ((b=a+b),(a=b-a))
#define KISS  ((MWC^CONG)+SHR3)
#define LFIB4 (c++,t[c]=t[c]+t[UC(c+58)]+t[UC(c+119)]+t[UC(c+178)])
#define SWB   (c++,bro=(x<y),t[c]=(x=t[UC(c+34)])-(y=t[UC(c+19)]+bro))
#define UNI   (KISS*2.328306e-10)
#define VNI   ((long) KISS)*4.656613e-10
#define UC    (unsigned char)  /*a cast operation*/
typedef unsigned long UL;

/*  Global static variables: */
 static UL z=362436069, w=521288629, jsr=123456789, jcong=380116160;
 static UL a=224466889, b=7584631, t[256];
/* Use random seeds to reset z,w,jsr,jcong,a,b, and the table t[256]*/

 static UL x=0,y=0,bro; static unsigned char c=0;

/* Example procedure to set the table, using KISS: */
void settable(UL i1,UL i2,UL i3,UL i4,UL i5, UL i6)
{
	int i; z=i1;w=i2,jsr=i3; jcong=i4; a=i5; b=i6;
	for(i=0;i<256;i=i+1)  t[i]=KISS;
}

void main2(unsigned *res, size_t num)
{
	srand(time(0));
	int a1 = rand();
	int a2 = rand();
	int a3 = rand();
	int a4 = rand();
	int a5 = rand();
	int a6 = rand();

   size_t i;
 //    settable(a1,a2,a3,a4,a5,a6);
   settable(1345,6542,3221,123453,651,9118);
	for(i=1; i<num; i++)
	{
		res[i]=KISS;
	} 
}

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
		size = 32,
		// Размер матрицы в двумерном случае (в элементах)
		size2 = size * size,
		// Размер матрицы в одномерном случае (в байтах)
		bsize = size * sizeof(float),
		// Размер матрицы в двумерном случае (в байтах)
		bsize2 = size2 * sizeof(float),
		// Количество итераций по времени
		count = 1024;
   
	float *h_v = new float[size2], *h_w = new float[size2], *h_stats = new float[count*size2];

	float *d_v, *d_w, *d_v2, *d_w2, *d_stats;
	unsigned *d_seed;
	cufftComplex *d_f1;
//	int fftInputSize = count*sizeof(float2);
	cutilSafeCall(cudaMalloc((void**) &d_v, bsize2));
	cutilSafeCall(cudaMalloc((void**) &d_w, bsize2));
	cutilSafeCall(cudaMalloc((void**) &d_v2, bsize2));
	cutilSafeCall(cudaMalloc((void**) &d_w2, bsize2));	
	cutilSafeCall(cudaMalloc((void**) &d_stats, count * bsize2));
//	cutilSafeCall(cudaMalloc((void**) &d_f1, fftInputSize));   // для фурье преобразования временной реализации одной точки, 2*count тк надо добавить комплексную часть = 0
	
	//int numBlocks = 4;
	dim3 blockDim(16,16);
	dim3 numBlocks(size/blockDim.x,size/blockDim.y);
	unsigned *h_seed = new unsigned[size2];
	main2(h_seed, size2);

	cutilSafeCall(cudaMalloc((void**) &d_seed, size2 * sizeof(unsigned)));
    cutilSafeCall(cudaMemcpy(d_seed, h_seed, size2 * sizeof(unsigned), cudaMemcpyHostToDevice));

	init <<<numBlocks, blockDim>>>(d_v);
	init <<<numBlocks, blockDim>>>(d_w);

//	RandomGPU<<<numBlocks, threadsPerBlock>>>(2*count, d_v, d_w, d_v2, d_w2, d_stats, size, 1, 1, 0.06, 2, 0.88);
	RandomGPU2<<<numBlocks, blockDim>>>(d_seed, count, d_stats, d_v, d_w, d_v2, d_w2, 0.8, 0.7, 0.06, 1.66, 0.88);
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	
	(cutStopTimer(timer));
    printf("Processing time: %f (ms)\n", cutGetTimerValue( timer));

	cutilCheckError( cutDeleteTimer( timer));
	cutilCheckError(cutCreateTimer(&timer));
    cutilCheckError(cutStartTimer(timer));
	
	cutilSafeCall(cudaMemcpy(h_stats, d_stats, count * bsize2, cudaMemcpyDeviceToHost));
	// <---------------   тут надо вызывать функцию которая будет забивать нулями комплексную часть исходного массива для фурье
	//фурье 
	//cufftHandle fftPlan;  
	  
//	cufftPlan1d(&fftPlan, count, CUFFT_C2R, 1); 

	cutilSafeCall( cudaThreadSynchronize() );
    (cutStopTimer(timer));
    printf("Copy device to host: %f (ms)\n", cutGetTimerValue( timer));
    cutilCheckError(cutDeleteTimer( timer));
    cutilCheckError(cutCreateTimer(&timer));
    cutilCheckError(cutStartTimer(timer));

	{
		std::ofstream output("G:\\output2.txt");

		for(int j = 0; j != 2; ++j)
		{
			for(int i = 0; i != count; ++i)
				output << h_stats[j*count+i] << "\t";
			output << std::endl;
		}
	}

    printf("Time of extracting data: %f (ms)\n", cutGetTimerValue( timer));
	
    cudaThreadExit();
}

