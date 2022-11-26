#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>

#define N		64
#define BLOCKSIZE	N
int main() {
	unsigned *matrix, *result, *hmatrix;
	cudaMalloc(&matrix, N * N * sizeof(unsigned));
	cudaMalloc(&result, N * N * sizeof(unsigned));
	hmatrix = (unsigned *)malloc(N * N * sizeof(unsigned));
}