GPUCOMP  = nvcc
GPUFLAGS = -arch=sm_70

INCLUDES  = 
LIBRARIES = -L${CUDA_DIR}/lib64 -lcublas

gpu_dgemm: gpu_dgemm.o
	${GPUCOMP} ${GPUFLAGS} ${LIBRARIES} gpu_dgemm.o -o gpu_dgemm

gpu_dgemm.o: gpu_dgemm.cu
	${GPUCOMP} ${GPUFLAGS} ${INCLUDES} -c gpu_dgemm.cu

.PHONY: clean

clean:
	rm -f gpu_dgemm *.o
