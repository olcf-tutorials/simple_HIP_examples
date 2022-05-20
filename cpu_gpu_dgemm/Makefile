GPUCOMP  = nvcc
GPUFLAGS = -arch=sm_70

INCLUDES  = -I${OLCF_ESSL_ROOT}/include
LIBRARIES = -L${CUDA_DIR}/lib64 -L${OLCF_ESSL_ROOT}/lib64 -lcublas -lessl

cpu_gpu_dgemm: cpu_gpu_dgemm.o
	${GPUCOMP} ${GPUFLAGS} ${LIBRARIES} cpu_gpu_dgemm.o -o cpu_gpu_dgemm

cpu_gpu_dgemm.o: cpu_gpu_dgemm.cu
	${GPUCOMP} ${GPUFLAGS} ${INCLUDES} -c cpu_gpu_dgemm.cu

.PHONY: clean

clean:
	rm -f cpu_gpu_dgemm *.o
