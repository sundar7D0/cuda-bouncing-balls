#include <iostream>
int *counter;

__global__ void printk(int *counter)
{ 
do {
	while (*counter % 2);
			++*counter;
		printf("\t%d\n", *counter);
	}while (*counter<10);
}

void fun(int* counter)
{
	if (*counter % 2 != 0)
	++*counter;
}

int main()
{
//	int *counter;
	cudaHostAlloc(&counter, sizeof(int), 0);
		printf("%d\n", *counter);	
	*counter=0;
	printk <<<1, 1>>>(counter);
	while(*counter<10)
		fun(counter);
/*
	do {
		printf("%d\n", *counter);
		while (*counter % 2 == 0);
			++*counter;
		}while(*counter < 10);
*/
	cudaFreeHost(counter);
	return 0;
}
