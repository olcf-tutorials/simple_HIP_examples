/* =========================================================================================
This program fills an NxN matrix (where N is even), A, with alternating sin(index) and 
cos(index) down its diagonal, performs the matrix multiply A*A on the GPU, then checks if 
the sum of the diagonal of the resulting matrix equals N/2 since 

    sin(index)*sin(index) + cos(index)*cos(index) = 1

Written by Tom Papatheodore
========================================================================================= */

#include <stdio.h>
#include <math.h>
#include <cublas_v2.h>

// Macro for checking errors in CUDA API calls
#define cudaErrorCheck(call)                                                              \
do{                                                                                       \
    cudaError_t cuErr = call;                                                             \
    if(cudaSuccess != cuErr){                                                             \
      printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));\
      exit(0);                                                                            \
    }                                                                                     \
}while(0)

// Size of matrices
#define N 512


int main(int argc, char *argv[])
{

	// Set device to GPU 0
	cudaErrorCheck( cudaSetDevice(0) );


	/* Allocate memory for A, B on CPU -------------------------------------------------*/

	double *A = (double*)malloc(N*N*sizeof(double));
	double *B = (double*)malloc(N*N*sizeof(double));


	/* Set Values for A, B on CPU ------------------------------------------------------*/

	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){

            int index = i*N + j;

            if(i == j){
                if(i % 2 == 0){
                    A[index] = sin((double)index);
                }
                else{
                    A[index] = cos((double)index);
                }
            }

            B[index] = 0.0;
        }
	}


	/* Allocate memory for d_A, d_B on GPU ---------------------------------------------*/

	double *d_A, *d_B;
	cudaErrorCheck( cudaMalloc(&d_A, N*N*sizeof(double)) );
	cudaErrorCheck( cudaMalloc(&d_B, N*N*sizeof(double)) );


	/* Copy host arrays (A,B) to device arrays (d_A,d_B) -------------------------------*/

	cudaErrorCheck( cudaMemcpy(d_A, A, N*N*sizeof(double), cudaMemcpyHostToDevice) );
	cudaErrorCheck( cudaMemcpy(d_B, B, N*N*sizeof(double), cudaMemcpyHostToDevice) );


	/* Perform Matrix Multiply on GPU --------------------------------------------------*/

    const double alpha = 1.0;
    const double beta = 0.0;

    cublasHandle_t handle;
    cublasCreate_v2(&handle);

	cublasStatus_t status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_A, N, &beta, d_B, N);

	if (status != CUBLAS_STATUS_SUCCESS){
		printf("cublasDgemm failed with code %d\n", status);
		return EXIT_FAILURE;
	}


	/* Copy values of d_B (computed on GPU) into host array B --------------------------*/
	cudaErrorCheck( cudaMemcpy(B, d_B, N*N*sizeof(double), cudaMemcpyDeviceToHost) );

    /* Check if result is "roughly" N/2 ------------------------------------------------*/

    double sum       = 0.0;

    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){

            if(i == j){ sum += B[i*N + j]; }
        }
    }

    double difference = sum - N/2;
    if(difference > 0){
        sum = floor(sum); 
    }
    else if(difference < 0){
        sum = ceil(sum);
    }

    if(sum != N/2){
        printf("sum = %f instead of %f\n", sum, (double)(N/2));
        exit(-1);
    }

	/* Clean up and output --------------------------------------------------------------*/

	cublasDestroy(handle);

    cudaErrorCheck( cudaFree(d_A) );
    cudaErrorCheck( cudaFree(d_B) );

    free(A);
    free(B);

    printf("__SUCCESS__\n");

    return 0;
}
