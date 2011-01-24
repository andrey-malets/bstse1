// Utilities and system includes
#include <shrUtils.h>
#include <iostream>
#include <fstream>
#include "cutil_inline.h"
#include <stdio.h>

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
 { int i; z=i1;w=i2,jsr=i3; jcong=i4; a=i5; b=i6;
 for(i=0;i<256;i=i+1)  t[i]=KISS;
 }


void main2(unsigned *res, int num)
{
	int i;
    settable(12345,65435,34221,12345,9983651,95746118);
	for(i=1;i<num;i++)
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
		size = 1024,
		// Размер матрицы в двумерном случае (в элементах)
		size2 = size*size,
		// Размер матрицы в одномерном случае (в байтах)
		bsize = size * sizeof(float),
		// Размер матрицы в двумерном случае (в байтах)
		bsize2 = size2 * sizeof(float),
		// Количество итераций по времени
		count = 2<<15;

	
   
    unsigned *devState=0;
	unsigned *cpu_seed = new unsigned [size]; 
	float *h_v = new float[size], *h_w = new float[size], *h_stats = new float[count*size],  *h_seed = new float[32];

	float *d_v, *d_w, *d_v2, *d_w2, *d_stats;
	unsigned *d_seed; 
	cutilSafeCall(cudaMalloc((void**) &d_v, bsize));
	cutilSafeCall(cudaMalloc((void**) &d_w, bsize));
	cutilSafeCall(cudaMalloc((void**) &d_v2, bsize));
	cutilSafeCall(cudaMalloc((void**) &d_w2, bsize));	
	cutilSafeCall(cudaMalloc((void**) &d_stats, count*bsize));



   // cutilSafeCall(cudaMalloc((void**) &h_seed, 32*sizeof(float)));
  
   // На 32 потока требутся одно 32 битное слово состояния
   // Вычисляем необходимое количество элементов для начального сидирования генератора
	
	int threadsPerBlock(size);
	int numBlocks = 1;
	int blockDim = 512;
	//unsigned localStateN = numBlocks*blockDim/32;
	//unsigned *seed = new unsigned[localStateN];
//	main2(seed,localStateN);
//	unsigned  totalStateBytes=4*blockDim/WarpStandard_K*WarpStandard_STATE_WORDS;
//	cutilSafeCall(cudaMalloc((void**) &d_seed, localStateN * sizeof(unsigned) ));
    
 
    
	


	init <<<numBlocks, threadsPerBlock>>>(d_v, size);
	init <<<numBlocks, threadsPerBlock>>>(d_w, size);




	const char *dat_path = shrFindFilePath("MersenneTwister.dat", argv[0]);
	loadMTGPU(dat_path);
	seedMTGPU(1001);

	RandomGPU<<<numBlocks, threadsPerBlock>>>(2*count, d_v, d_w, d_v2, d_w2, d_stats, size, 1, 1, 0.06, 2, 0.88);
//	RandomGPU2<<<numBlocks, threadsPerBlock>>>(WarpStandard_TEST_DATA,2*count, d_v, d_w, d_v2, d_w2, d_stats, size, 1, 1, 0.06, 2, 0.88);
	
	(cutStopTimer(timer));
     printf("Processing time: %f (ms)\n", cutGetTimerValue( timer));

	cutilCheckError( cutDeleteTimer( timer));
	cutilCheckError(cutCreateTimer(&timer));
    cutilCheckError(cutStartTimer(timer));
	
	
	cutilSafeCall(cudaMemcpy(h_stats, d_stats, count*bsize, cudaMemcpyDeviceToHost));
	
	/*int count2 = count;
	for (int i=0; i != count2; ++i)
	{
    
	RandomGPU<<<numBlocks, threadsPerBlock>>>(2, d_v, d_w, d_v2, d_w2, d_stats, size, 1, 1, 0.06, 2, 0.88);
	
	}*/
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

