#include "hip/hip_runtime.h"
#include <stdio.h>

// Macro for checking errors in HIP API calls
#define hipErrorCheck(call)                                                              \
do{                                                                                       \
    hipError_t hipErr = call;                                                             \
    if(hipSuccess != hipErr){                                                             \
      printf("HIP Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(hipErr));\
      exit(0);                                                                            \
    }                                                                                     \
}while(0)

// Size of array
#define N 1048576

// Kernel
__global__ void add_vectors_hip(double *a, double *b, double *c)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < N) c[id] = a[id] + b[id];
}

// Main program
int main()
{
	// Number of bytes to allocate for N doubles
	size_t bytes = N*sizeof(double);

	// Allocate memory for arrays A, B, and C on host
	double *A = (double*)malloc(bytes);
	double *B = (double*)malloc(bytes);
	double *C = (double*)malloc(bytes);

	// Allocate memory for arrays d_A, d_B, and d_C on device
	double *d_A, *d_B, *d_C;
	hipErrorCheck( hipMalloc(&d_A, bytes) );	
	hipErrorCheck( hipMalloc(&d_B, bytes) );
	hipErrorCheck( hipMalloc(&d_C, bytes) );

	// Fill host arrays A, B, and C
	for(int i=0; i<N; i++)
	{
		A[i] = 1.0;
		B[i] = 2.0;
        C[i] = 0.0;
	}

	// Copy data from host arrays A and B to device arrays d_A and d_B
	hipErrorCheck( hipMemcpy(d_A, A, bytes, hipMemcpyHostToDevice) );
	hipErrorCheck( hipMemcpy(d_B, B, bytes, hipMemcpyHostToDevice) );

	// Set execution configuration parameters
	//		thr_per_blk: number of HIP threads per grid block
	//		blk_in_grid: number of blocks in grid
	int thr_per_blk = 256;
	int blk_in_grid = ceil( float(N) / thr_per_blk );

	// Launch kernel
	hipLaunchKernelGGL(add_vectors_hip, blk_in_grid, thr_per_blk , 0, 0, d_A, d_B, d_C);

  	// Check for errors in kernel launch (e.g. invalid execution configuration paramters)
	hipError_t hipErrSync  = hipGetLastError();

  	// Check for errors on the GPU after control is returned to CPU
	hipError_t hipErrAsync = hipDeviceSynchronize();

	if (hipErrSync != hipSuccess) { printf("HIP Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(hipErrSync)); exit(0); }
	if (hipErrAsync != hipSuccess) { printf("HIP Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(hipErrAsync)); exit(0); }

	// Copy data from device array d_C to host array C
	hipErrorCheck( hipMemcpy(C, d_C, bytes, hipMemcpyDeviceToHost) );

	// Verify results
    double tolerance = 1.0e-14;
	for(int i=0; i<N; i++)
	{
            if( fabs(C[i] - 3.0) > tolerance )
		{ 
			printf("Error: value of C[%d] = %f instead of 3.0\n", i, C[i]);
			exit(-1);
		}
	}	

	// Free CPU memory
	free(A);
	free(B);
	free(C);

	// Free GPU memory
	hipErrorCheck( hipFree(d_A) );
	hipErrorCheck( hipFree(d_B) );
	hipErrorCheck( hipFree(d_C) );

	printf("\n---------------------------\n");
	printf("__SUCCESS__\n");
	printf("---------------------------\n");
	printf("N                 = %d\n", N);
	printf("Threads Per Block = %d\n", thr_per_blk);
	printf("Blocks In Grid    = %d\n", blk_in_grid);
	printf("---------------------------\n\n");

	return 0;
}
