#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <Windows.h>

using namespace std;

/*
__global__ void AplusB(int *ret, int a, int b) 
{
	ret[threadIdx.x] = a + b + threadIdx.x;
}
*/
/*
// 统一寻址
// 带有错误判断
int main() 
{
	cudaError_t cudaStatus;

	int *ret;

	cudaStatus = cudaMallocManaged(&ret, 10 * sizeof(int));
	
    if (cudaStatus != cudaSuccess) 
	{
        cout << "error!" << endl;
    }

	AplusB<<< 1, 10 >>>(ret, 10, 100);

	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) 
	{
        cout << "error2!" << endl;
    }

	for(int i=0; i<10; i++)
	{
		cout<< "A+B = " << ret[i] << endl;
	}

	cudaFree(ret);
	Sleep(20000);
	return 0;
} 
*/

/* 统一寻址
int main() 
{
	int *ret;

	cudaMallocManaged(&ret, 10 * sizeof(int));
	AplusB<<< 1, 10 >>>(ret, 10, 100);
	cudaDeviceSynchronize();

	for(int i=0; i<10; i++)
	{
		cout<< "A+B = " << ret[i] << endl;
	}

	cudaFree(ret);
	Sleep(20000);
	return 0;
} 

*/

/* 非统一寻址
int main() 
{
	int *ret;

	cudaMalloc(&ret, 10 * sizeof(int));
	AplusB<<< 1, 10 >>>(ret, 10, 100);

	int *host_ret = (int *)malloc(10 * sizeof(int));
	cudaMemcpy(host_ret, ret, 10 * sizeof(int), cudaMemcpyDeviceToHost);

	for(int i=0; i<10; i++)
	{
		cout<< "A+B = " << host_ret[i] << endl;
	}

	cudaFree(ret);
	free(host_ret);
	Sleep(20000);
	return 0;
}
*/


/* 
// 使用 __managed__
// 要求sm_30
__device__ __managed__ int ret[10];
__global__ void AplusB(int a, int b) 
{
	ret[threadIdx.x] = a + b + threadIdx.x;
}
int main() 
{
	AplusB<<< 1, 10 >>>(10, 100);
	cudaDeviceSynchronize();
	for(int i=0; i<10; i++)
	{
		cout<< "A+B = " << ret[i] << endl;
	}

	Sleep(20000);
	return 0;
}
*/

 
// 使用 __managed__
// 要求sm_30
__device__ __managed__ int ret[10];
__global__ void AplusB(int a, int b) 
{
	ret[threadIdx.x] = a + b + 2 * ret[threadIdx.x];
}
int main() 
{
	for(int i=0; i<10; i++)
	{
		ret[i] = i;
	}

	AplusB<<< 1, 10 >>>(10, 100);
	cudaDeviceSynchronize();
	for(int i=0; i<10; i++)
	{
		cout<< "A+B = " << ret[i] << endl;
	}

	Sleep(20000);
	return 0;
}



/*
__global__ void AplusB( int *ret, int a, int b) 
{
	ret[threadIdx.x] = a + b + ret[threadIdx.x];
}
int main() 
{
	//int *ret;
	//cudaMalloc(&ret, 1000 * sizeof(int));
	int ret[1000];
	for(int i=0; i<1000; i++)
	{
		ret[i] = i;
	}

	int *dev_ret = 0;
	cudaMalloc((void**)&dev_ret, 1000 * sizeof(int));
	cudaMemcpy(dev_ret, ret, 1000 * sizeof(int), cudaMemcpyHostToDevice);

	AplusB<<< 1, 1000 >>>(dev_ret, 10, 100);

	int *host_ret = (int *)malloc(1000 * sizeof(int));
	//cudaMemcpy(host_ret, ret, 1000 * sizeof(int), cudaMemcpyDefault);
	cudaMemcpy(host_ret, dev_ret, 1000 * sizeof(int), cudaMemcpyDeviceToHost);

	for(int i=0; i<1000; i++)
	{
		cout<< "A+B = " << host_ret[i] << endl;
	}
	Sleep(20000);
	free(host_ret);
	cudaFree(ret);
	return 0;
}
*/
