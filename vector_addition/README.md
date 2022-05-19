# hipify Vector Addition (CUDA to HIP)

In this tutorial, we will use a simple vector addition code to understand the steps necessary to translate a CUDA code to HIP using the `hipify-perl` tool. 

Consider the CUDA code below, called `vector_addition.cu`.

> NOTE: Although we will assume a basic familiarity with CUDA programming for this tutorial, it's worth pointing out that the macro defined above the `main` program is used to wrap CUDA API calls to check for errors and report meaningful error messages.

```c
#include <stdio.h>

// Macro for checking errors in CUDA API calls
#define cudaErrorCheck(call)                                                              \
do{                                                                                       \
    cudaError_t cuErr = call;                                                             \
    if(cudaSuccess != cuErr){                                                             \
      printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));\
      exit(0);                                                                            \
    }                                                                                     \
}while(0)

// Size of array
#define N 1048576

// Kernel
__global__ void add_vectors_cuda(double *a, double *b, double *c)
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
    cudaErrorCheck( cudaMalloc(&d_A, bytes) );
    cudaErrorCheck( cudaMalloc(&d_B, bytes) );
    cudaErrorCheck( cudaMalloc(&d_C, bytes) );

    // Fill host arrays A, B, and C
    for(int i=0; i<N; i++)
    {
        A[i] = 1.0;
        B[i] = 2.0;
        C[i] = 0.0;
    }

    // Copy data from host arrays A and B to device arrays d_A and d_B
    cudaErrorCheck( cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice) );

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(N) / thr_per_blk );

    // Launch kernel
    add_vectors_cuda<<< blk_in_grid, thr_per_blk >>>(d_A, d_B, d_C);

    // Check for errors in kernel launch (e.g. invalid execution configuration paramters)
    cudaError_t cuErrSync  = cudaGetLastError();

    // Check for errors on the GPU after control is returned to CPU
    cudaError_t cuErrAsync = cudaDeviceSynchronize();

    if (cuErrSync != cudaSuccess) { printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErrSync)); exit(0); }
    if (cuErrAsync != cudaSuccess) { printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErrAsync)); exit(0); }

    // Copy data from device array d_C to host array C
    cudaErrorCheck( cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost) );

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
    cudaErrorCheck( cudaFree(d_A) );
    cudaErrorCheck( cudaFree(d_B) );
    cudaErrorCheck( cudaFree(d_C) );

    printf("\n---------------------------\n");
    printf("__SUCCESS__\n");
    printf("---------------------------\n");
    printf("N                 = %d\n", N);
    printf("Threads Per Block = %d\n", thr_per_blk);
    printf("Blocks In Grid    = %d\n", blk_in_grid);
    printf("---------------------------\n\n");

    return 0;
}
```


## Compiling and Running on Summit

Before `hipify`-ing the code, let's run it as-is on Summit just to see what the expected output looks like. To do so, you must first load the CUDA module:

```
$ module load cuda/11.5.2
```

Now compile the program with the `nvcc` compiler:

```
$ nvcc vector_addition.cu -o run_cuda
```

Now submit the job using the `submit_cuda.lsf` batch script (make sure to change `PROJ123` to a project you are associated with):

```
$ bsub submit_cuda.lsf
```

You can check the status of your job with the `jobstat -u <username>` command. Once your job has completed, you can view the results in the output file named `add_vec_cuda.JOBID`

```
$ cat add_vec_cuda.JOBID

---------------------------
__SUCCESS__
---------------------------
N                 = 1048576
Threads Per Block = 256
Blocks In Grid    = 4096
---------------------------

...

```

## hipifying the Code

To translate the code from CUDA to HIP, you can use the `hipify-perl` translation tool: 

> NOTE: There are 2 versions of the `hipify` tool; a perl-based version and a clang-based version. For this tutorial we used the perl-based version only. For more information on the difference between the 2 versions, please see the <a href="https://www.exascaleproject.org/wp-content/uploads/2017/05/ORNL_HIP_webinar_20190606_final.pdf">slides</a> and <a href="https://youtu.be/3ZXbRJVvgJs">recording</a> from a webinar delivered by AMD.

```c
$ hipify-perl vector_addition.cu > vector_addition.cpp
  warning: vector_addition.cu:#4 : #define cudaErrorCheck(call)                                                              \
  warning: vector_addition.cu:#36 : 	cudaErrorCheck( hipMalloc(&d_A, bytes) );
  warning: vector_addition.cu:#37 : 	cudaErrorCheck( hipMalloc(&d_B, bytes) );
  warning: vector_addition.cu:#38 : 	cudaErrorCheck( hipMalloc(&d_C, bytes) );
  warning: vector_addition.cu:#49 : 	cudaErrorCheck( hipMemcpy(d_A, A, bytes, hipMemcpyHostToDevice) );
  warning: vector_addition.cu:#50 : 	cudaErrorCheck( hipMemcpy(d_B, B, bytes, hipMemcpyHostToDevice) );
  warning: vector_addition.cu:#71 : 	cudaErrorCheck( hipMemcpy(C, d_C, bytes, hipMemcpyDeviceToHost) );
  warning: vector_addition.cu:#90 : 	cudaErrorCheck( hipFree(d_A) );
  warning: vector_addition.cu:#91 : 	cudaErrorCheck( hipFree(d_B) );
  warning: vector_addition.cu:#92 : 	cudaErrorCheck( hipFree(d_C) );
```

> NOTE: The original source file `vector_addition.cu` was not altered by using the `hipify-perl` tool. In fact, if you do not redirect the output into a new file (as we've done here with `> vector_addition.cpp`), the reulting output would simply be printed to stdout without changes to the code. 

Before looking at the new file that was created, notice that there were several warnings printed. These warnings are there to notify you that some lines of code that refers to CUDA, but are not actually part of the CUDA API, CUDA built-ins, or CUDA variable types that `hipify-perl` recognizes, were found (in this case, they are all related to the `cudaErrorCheck` macro).

Ok, now let's look at the new file, `vector_addition.cpp`:

```c
#include "hip/hip_runtime.h"
#include <stdio.h>

// Macro for checking errors in CUDA API calls
#define cudaErrorCheck(call)                                                              \
do{                                                                                       \
    hipError_t cuErr = call;                                                             \
    if(hipSuccess != cuErr){                                                             \
      printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(cuErr));\
      exit(0);                                                                            \
    }                                                                                     \
}while(0)

// Size of array
#define N 1048576

// Kernel
__global__ void add_vectors_cuda(double *a, double *b, double *c)
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
    cudaErrorCheck( hipMalloc(&d_A, bytes) );
    cudaErrorCheck( hipMalloc(&d_B, bytes) );
    cudaErrorCheck( hipMalloc(&d_C, bytes) );

    // Fill host arrays A, B, and C
    for(int i=0; i<N; i++)
    {
        A[i] = 1.0;
        B[i] = 2.0;
        C[i] = 0.0;
    }

    // Copy data from host arrays A and B to device arrays d_A and d_B
    cudaErrorCheck( hipMemcpy(d_A, A, bytes, hipMemcpyHostToDevice) );
    cudaErrorCheck( hipMemcpy(d_B, B, bytes, hipMemcpyHostToDevice) );

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(N) / thr_per_blk );

    // Launch kernel
    hipLaunchKernelGGL(add_vectors_cuda, blk_in_grid, thr_per_blk , 0, 0, d_A, d_B, d_C);

    // Check for errors in kernel launch (e.g. invalid execution configuration paramters)
    hipError_t cuErrSync  = hipGetLastError();

    // Check for errors on the GPU after control is returned to CPU
    hipError_t cuErrAsync = hipDeviceSynchronize();

    if (cuErrSync != hipSuccess) { printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(cuErrSync)); exit(0); }
    if (cuErrAsync != hipSuccess) { printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(cuErrAsync)); exit(0); }

    // Copy data from device array d_C to host array C
    cudaErrorCheck( hipMemcpy(C, d_C, bytes, hipMemcpyDeviceToHost) );

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
    cudaErrorCheck( hipFree(d_A) );
    cudaErrorCheck( hipFree(d_B) );
    cudaErrorCheck( hipFree(d_C) );

    printf("\n---------------------------\n");
    printf("__SUCCESS__\n");
    printf("---------------------------\n");
    printf("N                 = %d\n", N);
    printf("Threads Per Block = %d\n", thr_per_blk);
    printf("Blocks In Grid    = %d\n", blk_in_grid);
    printf("---------------------------\n\n");

    return 0;
```

Looking at this code, we can see the following:

* The `hip/hip_runtime.h` header file has been added.

* The CUDA API calls (`cudaMalloc`, `cudaMemcpy`, `cudaFree`, `cudaGetErrorString`, `cudaGetLastError`, `cudaDeviceSynchronize`), CUDA built-ins (`cudaSuccess`, `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`), and CUDA types (`cudaError_t`) have all been correctly translated to HIP.

* The CUDA kernel call (`add_vectors_cuda<<< blk_in_grid, thr_per_blk >>>(d_A, d_B, d_C);`) has been translated to correctly use the HIP kernel launch syntax (`hipLaunchKernelGGL((add_vectors_cuda), dim3(blk_in_grid), dim3(thr_per_blk ), 0, 0, d_A, d_B, d_C);`), but the kernel's name still reads `add_vectors_cuda`. 

* The user-defined macro (`cudaErrorCheck`), kernel function (`add_vectors_cuda`), variables (`cuErr`, `cuErrSync`, `cuErrAsync`), and comments have not been translated. These will require manual intervention.

> NOTE: The warnings that were produced when we used the `hipify-perl` tool only found the error checking macro (`cudaErrorCheck`) because it contained a commonly-used block of text ("`cudaError`"), but did not catch the other user-defined functions, variables, and text.

> NOTE: This code will still compile and run correctly without making the additional changes.

For this simple code, we can manually translate all occurrences of `CUDA` and `cuda` to `HIP` and `hip`, respectively, using the following `sed` commands:

```c
$ sed -i 's/CUDA/HIP/g' vector_addition.cpp
$ sed -i 's/cuda/hip/g' vector_addition.cpp 
```

And all occurrences of `cuErr*` to `hipErr*` using this command:

```c
$ sed -i 's/cuErr/hipErr/g' vector_addition.cpp
```

> NOTE: By removing the `-i` flag from `sed`, you can see the results first instead of editing the file in place.

> NOTE: This will obviously be application-specific so please use appropriate commands for your own application.

Now the code is fully ported to HIP - including both the necessary steps as well as the manual steps that were (in this case) really just to change the user-defined function/variable names. In the future, we should probably use terms like "device" instead of platform-specific terms such as "CUDA" and "HIP".

## Compiling and Running hipify Version on Summit

To run this program on Summit, you must first load the HIP module:

```
$ module load hip-cuda/5.1.0
```

> NOTE: This version of `hip-cuda` requires a `cuda` module >= version 11.4

Now compile the program using the `hipcc` compiler:

```
$ hipcc vector_addition.cpp -o run_hip
```

Now submit the job using the `submit_hip.lsf` batch script (make sure to change `PROJ123` to a project you are associated with):

```
$ bsub submit_hip.lsf
```

You can check the status of your job with the `jobstat` command. Once your job has completed, you can view the results in the output file named `add_vec_hip.JOBID`

```
$ cat add_vec_hip.JOBID

---------------------------
__SUCCESS__
---------------------------
N                 = 1048576
Threads Per Block = 256
Blocks In Grid    = 4096
---------------------------

...

```

So we have successfully translated our CUDA version of the vector addition code to HIP. For this simple code, the `hipify-perl` tool was sufficient to produce a HIP version of the code that gave correct results, but we needed to manually change most of our user-defined functions, variables, and text.

## Helpful (external) Links

For more information on HIP programming and porting, please visit <a href="https://docs.amd.com/">AMD's documentation</a>

## Problems?
If you see a problem with the code or have suggestions to improve it, feel free to open an issue.
