// Utilities and system includes
#include <shrUtils.h>
#include <iostream>
#include <fstream>
#include <cutil_inline.h>
#include <stdio.h>
#include <cufft.h>
#include <time.h>
using namespace std;



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

// генератор массива случайных чисел дл€ начального seed'a warp_standart'a

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
   settable(a1,a2,a3,a4,a5,a6);
  // settable(1345,6542,3221,123453,651,9118);
	for(i=1; i<num; i++)
	{
		res[i]=KISS;
	} 
}

int main(int argc, char** argv)
{

	printf("[ %s ]\n", sSDKsample);
	char ans;
		do 
		{ 
			runTest(argc, argv);
			printf("Do you want to run again ? Y/N \n");
			cin>>ans;
		}  
		while (ans != 'n');

}


void runTest(int argc, char** argv)
{
	

    // ѕараметры системы
	size_t
		// –азмер строки в одномерном случае (в элементах)
		size = 256,
		// –азмер матрицы в двумерном случае (в элементах)
		size2 = size * size,
		// –азмер матрицы в одномерном случае (в байтах)
		bsize = size * sizeof(float),
		// –азмер матрицы в двумерном случае (в байтах)
		bsize2 = size2 * sizeof(float),
		//  оличество итераций по времени
		count = 1<<15;
		// параметры счЄта
		float dt = 0.005, 
		// шаг по времени
		m = 0.075,
		// интенсивность шума
		d1 = 1;
		// количество сохран€емых данных
		int sn = 1;
		//массив с номерами строк матрицы дл€ сохранени€ и последующей обработки
		int cif;
		//X или Y строки сохран€ем.
		int XY;
	/*	printf("Enter parametrs of system: Time(dt), Noise(m), Diffusion(d1), Number of saved lines(sn) \n");
		printf("default values 0.005, 0.075, 1, 1 \n");
		scanf("%f%f%f%i", &dt, &m, &d1, &sn);
		*/
		/*cout<<"Enter number of save lines";
		cin>>sn;
		unsigned* h_nsavedata= new unsigned[sn];
		  for(int i=0; i<sn; i++)
			 {
				 cout<<"Enter coordinate for " <<i<<" line ";
				 cin>>cif;
				 h_nsavedata[i]=cif;
			 }*/
		
	  



	    double dt1 = (double) dt;
	    float c1 = (float)0.5*m/pow(dt1,0.5),c2 = (float)0.5*m/pow(dt1,0.5), D = (float) 2*m/pow(dt1, 0.5)*d1;
	
	cudaSetDevice(cutGetMaxGflopsDeviceId());
    // таймер дл€ оценки времени работы программы
	unsigned int timer = 0;
    cutilCheckError(cutCreateTimer(&timer));
    cutilCheckError(cutStartTimer(timer));



	float *h_v = new float[size2], *h_w = new float[size2], *h_stats = new float[count*size*sn], *h_fft = new float[(count/2)*size];

	float *d_v, *d_w, *d_v2, *d_w2, *d_stats;
	unsigned *d_seed, *d_nsavedata;
	cufftComplex *d_f1;

	cutilSafeCall(cudaMalloc((void**) &d_v, bsize2));
	cutilSafeCall(cudaMalloc((void**) &d_w, bsize2));
	cutilSafeCall(cudaMalloc((void**) &d_v2, bsize2));
	cutilSafeCall(cudaMalloc((void**) &d_w2, bsize2));	
	cutilSafeCall(cudaMalloc((void**) &d_stats, count * bsize*sn));
//	cutilSafeCall(cudaMalloc((void**) &d_nsavedata, sn*sizeof(unsigned)));
	dim3 blockDim(32,32);
	dim3 numBlocks(size/blockDim.x,size/blockDim.y);
	unsigned *h_seed = new unsigned[size2];
	main2(h_seed, size2);

	cutilSafeCall(cudaMalloc((void**) &d_seed, size2 * sizeof(unsigned)));
    cutilSafeCall(cudaMemcpy(d_seed, h_seed, size2 * sizeof(unsigned), cudaMemcpyHostToDevice));
//	cutilSafeCall(cudaMemcpy(d_nsavedata, h_nsavedata, sn * sizeof(unsigned), cudaMemcpyHostToDevice));
	init <<<numBlocks, blockDim>>>(d_v);
	init <<<numBlocks, blockDim>>>(d_w);

//	RandomGPU<<<numBlocks, threadsPerBlock>>>(2*count, d_v, d_w, d_v2, d_w2, d_stats, size, 1, 1, 0.06, 2, 0.88);
	RandomGPU2<<<numBlocks, blockDim>>>(d_seed, count, d_stats, d_v, d_w, d_v2, d_w2, c1, c2, dt, D, m);
	cudaFree(d_seed);
	cudaFree(d_v);
    cudaFree(d_w);
	cudaFree(d_v2);
	cudaFree(d_w2);
	
	(cutStopTimer(timer));
    printf("Processing time: %f (ms)\n", cutGetTimerValue( timer));




	cutilCheckError( cutDeleteTimer( timer));
	cutilCheckError(cutCreateTimer(&timer));
    cutilCheckError(cutStartTimer(timer));
	
  // cutilSafeCall(cudaMemcpy(h_stats, d_stats, count * bsize, cudaMemcpyDeviceToHost));
	
	//фурье 
	cufftHandle fftPlan;  
	cufftComplex *d_fft;
	int numf = size;
	cufftReal *h = new cufftReal[(count/2+1)*numf]; 
	float *d_ffta; // дл€ модулей фурье преобразовани€
	cudaMalloc((void**)&d_fft,sizeof(cufftComplex)*(count/2+1)*numf);
	cudaMalloc((void**)&d_ffta,sizeof(cufftReal)*(count/2+1)*numf);
	cufftPlan1d(&fftPlan, count, CUFFT_R2C, numf);
	printf("1");
	cufftExecR2C(fftPlan, d_stats, d_fft);
	printf("2");
	cutilSafeCall( cudaThreadSynchronize() );
	printf("3");
	ComplexAbs <<<count*size/1024,512>>>(d_fft, d_ffta,(count/2+1)*numf);
	cutilSafeCall( cudaThreadSynchronize() );
	printf("4");
	cutilSafeCall(cudaMemcpy(h, d_ffta, (count/2+1) * numf*sizeof(float), cudaMemcpyDeviceToHost));
	cutilSafeCall( cudaThreadSynchronize() );
    (cutStopTimer(timer));
    printf("Copy device to host: %f (ms)\n", cutGetTimerValue( timer));
    cutilCheckError(cutDeleteTimer( timer));
    cutilCheckError(cutCreateTimer(&timer));
    cutilCheckError(cutStartTimer(timer));

	{
		char s[1000];
        sprintf(s, "C:\\2D s = %i count = %i dt = %f M = %f d1 = %f.txt", size, count, dt, m, d1);
		std::ofstream output(s);

		//for(int j = 0; j != numf; ++j)
		//{
		//	for(int i = 0; i != count/2+1; ++i)
		//	{
		//		output << h[j*(count/2+1)+i] << "\t";
		//		//output << h[j*(count/2+1)+i]<< "\t";
		//	}
		//	output << std::endl;
		//}
		//
	
		
		cufftReal *h_fftsum = new cufftReal[count/2+1];
		for(int i = 0; i != count/2+1; ++i)
			h_fftsum[i]=0;
		

		for(int j = 0; j != numf; ++j)
		{
			for(int i = 0; i != count/2+1; ++i)
				h_fftsum[i] += h[j*(count/2+1)+i];
		}
	/*	output << "size = " << size << "n= " << count << "dt= " << dt << "M= " << m << "d1= " << d1;
		output << std::endl;*/
		for(int i = 0; i != count/2+1; ++i)
		{
		output << h_fftsum[i]/numf << "\t";
			
		}
		output << std::endl;
	}

    printf("Time of extracting data: %f (ms)\n", cutGetTimerValue( timer));
	
    cudaThreadExit();
	
}

