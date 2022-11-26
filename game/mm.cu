#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>

__global__ void reset(unsigned *matrix, unsigned matrixsize) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned jj = 0; jj < matrixsize; ++jj) {
		matrix[id * matrixsize + jj] = 0;
	}
}
__global__ void init(unsigned *matrix, unsigned matrixsize) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned jj = 0; jj < matrixsize; ++jj) {
		matrix[id * matrixsize + jj] = id * matrixsize + jj;
	}
}
__global__ void square(unsigned *matrix, unsigned *result, unsigned matrixsize) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned jj = 0; jj < matrixsize; ++jj) {
		for (unsigned kk = 0; kk < matrixsize; ++kk) {
			result[id * matrixsize + jj] += matrix[id * matrixsize + kk] * matrix[kk * matrixsize + jj];
		}
	}
}
void printmatrix(unsigned *matrix, unsigned matrixsize) {
	for (unsigned ii = 0; ii < matrixsize; ++ii) {
		for (unsigned jj = 0; jj < matrixsize; ++jj) {
			printf("%8d ", matrix[ii * matrixsize + jj]);
		}
		printf("\n");
	}
}
void initmatrix(unsigned *matrix, unsigned matrixsize) {
	for (unsigned ii = 0; ii < matrixsize; ++ii) {
		for (unsigned jj = 0; jj < matrixsize; ++jj) {
			matrix[ii * matrixsize + jj] = ii * matrixsize + jj;
		}
	}
}
void compareresults(unsigned *one, unsigned *two, unsigned matrixsize) {
	for (unsigned ii = 0; ii < matrixsize; ++ii) {
		for (unsigned jj = 0; jj < matrixsize; ++jj) {
			if (one[ii * matrixsize + jj] != two[ii * matrixsize + jj]) {
				printf("ERROR: mismatch at [%d][%d]: %d versus %d.\n", ii, jj, one[ii * matrixsize + jj], two[ii * matrixsize + jj]);
				return;
			}
		}
	}
	printf("Same output.\n");
}
void squarecpu(unsigned *matrix, unsigned *result, unsigned matrixsize) {
	for (unsigned ii = 0; ii < matrixsize; ++ii) {
	for (unsigned jj = 0; jj < matrixsize; ++jj) {
		for (unsigned kk = 0; kk < matrixsize; ++kk) {
			result[ii * matrixsize + jj] += matrix[ii * matrixsize + kk] * matrix[kk * matrixsize + jj];
		}
	}
	}

}
double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d", stat);
  return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime) {
	printf("%s%3f seconds\n", str, endtime - starttime);
}


#define N		64
#define BLOCKSIZE	N
int main() {
	unsigned *matrix, *result, *hmatrix;
	cudaMalloc(&matrix, N * N * sizeof(unsigned));
	cudaMalloc(&result, N * N * sizeof(unsigned));
	hmatrix = (unsigned *)malloc(N * N * sizeof(unsigned));

	unsigned nblocks = ceil((float)N / BLOCKSIZE);
	printf("nblocks = %d\n", nblocks);

    	init<<<nblocks, BLOCKSIZE>>>(matrix, N);
    	reset<<<nblocks, BLOCKSIZE>>>(result, N);
	double starttime = rtclock();
    	square<<<nblocks, BLOCKSIZE>>>(matrix, result, N);
	cudaThreadSynchronize();
	double endtime = rtclock();

	printtime("GPU time: ", starttime, endtime);
	cudaMemcpy(hmatrix, result, N * N * sizeof(unsigned), cudaMemcpyDeviceToHost);

	//printmatrix(hmatrix, N);

	unsigned *secondmatrix, *secondresult;
	secondmatrix = (unsigned *)malloc(N * N * sizeof(unsigned));
	initmatrix(secondmatrix, N);

	secondresult = (unsigned *)calloc(N* N, sizeof(unsigned));
	starttime = rtclock();
	squarecpu(secondmatrix, secondresult, N);
	endtime = rtclock();
	printtime("CPU time: ", starttime, endtime);

	compareresults(secondresult, hmatrix, N);
    	return 0;
}
