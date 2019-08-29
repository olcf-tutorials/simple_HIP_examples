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
__global__ void add_vectors_hip(int *a, int *b, int *c)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < N) c[id] = a[id] + b[id];
}

// Main program
int main()
{
	// Number of bytes to allocate for N integers
	size_t bytes = N*sizeof(int);

	// Allocate memory for arrays A, B, and C on host
	int *A = (int*)malloc(bytes);
	int *B = (int*)malloc(bytes);
	int *C = (int*)malloc(bytes);

	// Allocate memory for arrays d_A, d_B, and d_C on device
	int *d_A, *d_B, *d_C;
	hipErrorCheck( hipMalloc(&d_A, bytes) );	
	hipErrorCheck( hipMalloc(&d_B, bytes) );
	hipErrorCheck( hipMalloc(&d_C, bytes) );

	// Fill host arrays A and B
	for(int i=0; i<N; i++)
	{
		A[i] = 1;
		B[i] = 2;
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
	hipLaunchKernelGGL((add_vectors_hip), dim3(blk_in_grid), dim3(thr_per_blk ), 0, 0, d_A, d_B, d_C);

  	// Check for errors in kernel launch (e.g. invalid execution configuration paramters)
	hipError_t hipErrSync  = hipGetLastError();

  	// Check for errors on the GPU after control is returned to CPU
	hipError_t hipErrAsync = hipDeviceSynchronize();

	if (hipErrSync != hipSuccess) { printf("HIP Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(hipErrSync)); exit(0); }
	if (hipErrAsync != hipSuccess) { printf("HIP Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(hipErrAsync)); exit(0); }

	// Copy data from device array d_C to host array C
	hipErrorCheck( hipMemcpy(C, d_C, bytes, hipMemcpyDeviceToHost) );

	// Verify results
	for(int i=0; i<N; i++)
	{
		if(C[i] != 3)
		{ 
			printf("Error: value of C[%d] = %d instead of 3\n", i, C[i]);
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
