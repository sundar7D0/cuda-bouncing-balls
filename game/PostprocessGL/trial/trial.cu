#include <iostream>

__global__ void kernel()
{
//	volatile __shared__ int s;
	__shared__ int i;
	i=threadIdx.x;
//	__threadfence();
//	__syncthreads();
	if(i!=threadIdx.x)
		printf("thread_idx:%d;i:%d\n",threadIdx.x,i);
}

int main(int argc, char* argv[])
{
	for(int i=0;i<1000;i++)
		kernel<<<1000,1000>>>();
	cudaDeviceSynchronize();
}