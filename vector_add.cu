// 2014 1 14 by lichao
// CUDA实战 27页
// 矢量求和
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <Windows.h>

#define N 10
using namespace std;

__global__ void add( int *a, int *b, int *c)
{
	int tid = blockIdx.x; //计算该索引处的数据
	if ( tid < N)
	{
		c[tid] = a[tid] + b[tid];
	}
}

int main()
{
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;
	cudaError_t cudaStatus;
	
	// assign numbers to 'a' and 'b' on the cpu
	for (int i=0; i<N; i++)
	{
		a[i] = -i;
		b[i] = i * i;
	}

	// malloc memory on the GPU
	cudaStatus = cudaMalloc( (void**)&dev_a, N * sizeof(int) );
	if (cudaStatus != cudaSuccess)
	{
		cout << "cudaMalloc failed!" << endl;
		goto Error;
	}
	cudaStatus = cudaMalloc( (void**)&dev_b, N * sizeof(int) );
	if (cudaStatus != cudaSuccess)
	{
		cout << "cudaMalloc failed!" << endl;
		goto Error;
	}
	cudaStatus = cudaMalloc( (void**)&dev_c, N * sizeof(int) );
	if (cudaStatus != cudaSuccess)
	{
		cout << "cudaMalloc failed!" << endl;
		goto Error;
	}

	// copy memory from host to device
	cudaStatus = cudaMemcpy( dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cout << "cudaMemcpy failed!" << endl;
		goto Error;
	}
	cudaStatus = cudaMemcpy( dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cout << "cudaMemcpy failed!" << endl;
		goto Error;
	}

	add<<<N,1>>>( dev_a, dev_b, dev_c );

	cudaStatus = cudaMemcpy( c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		cout << "cudaMemcpy failed!" << endl;
		goto Error;
	}	
	for (int i=0; i<N; i++)
	{
		cout << "c:" << c[i] << endl;
	}
	
Error:
	cudaFree( dev_a );
	cudaFree( dev_b );
	cudaFree( dev_c );

	Sleep( 20000 );
	return 0;
}
