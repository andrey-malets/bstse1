// Utilities and system includes
#include <shrUtils.h>
#include <iostream>
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

__global__ void init(float *matrix, int size)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

	// Здесь можно сделать инициализацию
	matrix[tx*size + ty] = 1;
}

__global__ void step(float *src_v, float *src_w, float *dst_v, float *dst_w, int size, float c1, float c2, float dt, float D, float M, R1,R2)
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

	dst_w[x*size+y] = (src_w[x*size+y] + src_v[x*size+y] * dt) / (1 + src_v[x*size+y] * src_v[x*size+y] * dt) +  R2  * (__powf(dt, 0.5)) * M;
}


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

	size_t size = 5, size2 = size*size, bsize = size2 * sizeof(float);
	float *h_v = new float[size2], *h_w = new float[size2];

	float *d_v, *d_w, *d_v2, *d_w2;
	cutilSafeCall(cudaMalloc((void**) &d_v, bsize));
	cutilSafeCall(cudaMalloc((void**) &d_w, bsize));
	cutilSafeCall(cudaMalloc((void**) &d_v2, bsize));
	cutilSafeCall(cudaMalloc((void**) &d_w2, bsize));

	//cutilSafeCall(cudaMemcpy(d_v, h_v, bsize, cudaMemcpyHostToDevice));
	//cutilSafeCall(cudaMemcpy(d_w, h_w, bsize, cudaMemcpyHostToDevice));

	dim3 threadsPerBlock(size, size);
	int numBlocks = 1;
	init <<<numBlocks, threadsPerBlock>>>(d_v, size);
	init <<<numBlocks, threadsPerBlock>>>(d_w, size);

	step <<<numBlocks, threadsPerBlock>>>(d_v, d_w, d_v2, d_w2, size, 2, 2, 0.05, 4, 0.8);
	step <<<numBlocks, threadsPerBlock>>>(d_v2, d_w2, d_v, d_w, size, 2, 2, 0.05, 4, 0.8);
	step <<<numBlocks, threadsPerBlock>>>(d_v, d_w, d_v2, d_w2, size, 2, 2, 0.05, 4, 0.8);
	step <<<numBlocks, threadsPerBlock>>>(d_v2, d_w2, d_v, d_w, size, 2, 2, 0.05, 4, 0.8);
	step <<<numBlocks, threadsPerBlock>>>(d_v, d_w, d_v2, d_w2, size, 2, 2, 0.05, 4, 0.8);
	step <<<numBlocks, threadsPerBlock>>>(d_v2, d_w2, d_v, d_w, size, 2, 2, 0.05, 4, 0.8);
	step <<<numBlocks, threadsPerBlock>>>(d_v, d_w, d_v2, d_w2, size, 2, 2, 0.05, 4, 0.8);
	step <<<numBlocks, threadsPerBlock>>>(d_v2, d_w2, d_v, d_w, size, 2, 2, 0.05, 4, 0.8);

	cutilSafeCall(cudaMemcpy(h_v, d_v2, bsize, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(h_w, d_w2, bsize, cudaMemcpyDeviceToHost));

	for(int i = 0; i != size2; ++i)
		std::cout << h_v[i] << " ";

	//unsigned int size_A = uiWA * uiHA;
 //   unsigned int mem_size_A = sizeof(float) * size_A;
 //   float* h_A = (float*)malloc(mem_size_A);
 //   unsigned int size_B = uiWB * uiHB;
 //   unsigned int mem_size_B = sizeof(float) * size_B;
 //   float* h_B = (float*)malloc(mem_size_B);

 //   // initialize host memory
 //   randomInit(h_A, size_A);
 //   randomInit(h_B, size_B);

 //   // allocate device memory
 //   float* d_A;
 //   cutilSafeCall(cudaMalloc((void**) &d_A, mem_size_A));
 //   float* d_B;
 //   cutilSafeCall(cudaMalloc((void**) &d_B, mem_size_B));

 //   // copy host memory to device
 //   cutilSafeCall(cudaMemcpy(d_A, h_A, mem_size_A,
 //                             cudaMemcpyHostToDevice) );
 //   cutilSafeCall(cudaMemcpy(d_B, h_B, mem_size_B,
 //                             cudaMemcpyHostToDevice) );

 //   // allocate device memory for result
 //   unsigned int size_C = uiWC * uiHC;
 //   unsigned int mem_size_C = sizeof(float) * size_C;
 //   float* d_C;
 //   cutilSafeCall(cudaMalloc((void**) &d_C, mem_size_C));

 //   // allocate host memory for the result
 //   float* h_C = (float*) malloc(mem_size_C);
 //   

 //   // setup execution parameters
 //   dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
 //   dim3 grid(uiWC / threads.x, uiHC / threads.y);

 //   // kernel warmup
 //   matrixMul<<< grid, threads >>>(d_C, d_A, d_B, uiWA, uiWB);
 //   cudaThreadSynchronize();
 //   
 //   // create and start timer
 //   shrLog("Run Kernels...\n\n");
 //   unsigned int timer = 0;
 //   cutilCheckError(cutCreateTimer(&timer));
 //   cutilCheckError(cutStartTimer(timer));

 //   // execute the kernel
 //   int nIter = 30;
 //   for (int j = 0; j < nIter; j++) 
	//	{
 //           matrixMul<<< grid, threads >>>(d_C, d_A, d_B, uiWA, uiWB);
 //       }

 //   // check if kernel execution generated and error
 //   cutilCheckMsg("Kernel execution failed");

 //   cudaThreadSynchronize();
 //   // stop and destroy timer
 //   cutilCheckError(cutStopTimer(timer));
 //   double dSeconds = cutGetTimerValue(timer)/((double)nIter * 1000.0);
 //   double dNumOps = 2.0 * (double)uiWA * (double)uiHA * (double)uiWB;
 //   double gflops = 1.0e-9 * dNumOps/dSeconds;

 //   //Log througput, etc
 //   shrLogEx(LOGBOTH | MASTER, 0, "matrixMul, Throughput = %.4f GFlop/s, Time = %.5f s, Size = %.0f Ops, NumDevsUsed = %d, Workgroup = %u\n", 
 //           gflops, dSeconds, dNumOps, 1, threads.x * threads.y);
 //   cutilCheckError(cutDeleteTimer(timer));

 //   // copy result from device to host
 //   cutilSafeCall(cudaMemcpy(h_C, d_C, mem_size_C,
 //                             cudaMemcpyDeviceToHost) );

 //   // compute reference solution
 //   shrLog("\nCheck against Host computation...\n\n");    
 //   float* reference = (float*)malloc(mem_size_C);
 //   computeGold(reference, h_A, h_B, uiHA, uiWA, uiWB);

 //   // check result
 //   shrBOOL res = shrCompareL2fe(reference, h_C, size_C, 1.0e-6f);
 //   if (res != shrTRUE) 
 //   {
 //       printDiff(reference, h_C, uiWC, uiHC, 100, 1.0e-5f);
 //   }
 //   shrLog("%s \n\n", (shrTRUE == res) ? "PASSED" : "FAILED");

 //   // clean up memory
 //   free(h_A);
 //   free(h_B);
 //   free(h_C);
 //   free(reference);
 //   cutilSafeCall(cudaFree(d_A));
 //   cutilSafeCall(cudaFree(d_B));
 //   cutilSafeCall(cudaFree(d_C));

    cudaThreadExit();
}

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
    shrLog("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int i,j,k;
    int error_count=0;
    for (j = 0; j < height; j++) 
    {
        if (error_count < iListLength)
        {
            shrLog("\n  Row %d:\n", j);
        }
        for (i = 0; i < width; i++) 
        {
            k = j * width + i;
            float fDiff = fabs(data1[k] - data2[k]);
            if (fDiff > fListTol) 
            {                
                if (error_count < iListLength)
                {
                    shrLog("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }
                error_count++;
            }
        }
    }
    shrLog(" \n  Total Errors = %d\n\n", error_count);
}
