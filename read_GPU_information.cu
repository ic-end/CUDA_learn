
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <Windows.h>
#include <iostream>

using namespace std;

int main()
{
	cudaDeviceProp  prop;
	cudaError_t cudaStatus;

    int count;
    cudaStatus = cudaGetDeviceCount( &count );
	if (cudaStatus != cudaSuccess)
	{
		cout << "cudaMalloc failed!" << endl;
	}
    for (int i=0; i< count; i++) 
	{
        cudaStatus = cudaGetDeviceProperties( &prop, i );
		if (cudaStatus != cudaSuccess)
		{
			cout << "cudaGetDeviceProperties failed!" << endl;
		}

		cout << "   --- 第" << i << "个设备信息 ---" << "\n" << endl;
		cout << "显卡名字：" << prop.name << "\n" << endl;
		cout << "计算能力：" << prop.major << "." << prop.minor << "\n" << endl;
		cout << "时钟频率：" << prop.clockRate << "\n" << endl;
        if (prop.deviceOverlap)
		{
			cout << "是否可以同时执行cudaMemory()调用和一个核函数调用：是!" << "\n" << endl;
		}
        else
		{
			cout << "是否可以同时执行cudaMemory()调用和一个核函数调用：否!" << "\n" << endl;
		}

        if (prop.kernelExecTimeoutEnabled)
		{
			cout << "设备上执行的核函数是否存在运行时限制：是!" << "\n" << endl;
		}
        else
		{
			cout << "设备上执行的核函数是否存在运行时限制：否!" << "\n" << endl;
		}

		cout << "\n" << "   ---设备内存信息---" << "\n" << endl;
		cout << "全局内存总量（字节）：" << prop.totalGlobalMem << "\n" << endl;
		cout << "常量内存总量（字节）：" << prop.totalConstMem << "\n" << endl;
		cout << "内存复制中的最大修正值（字节）：" << prop.memPitch << "\n" << endl;
		cout << "设备的纹理对齐要求：" << prop.textureAlignment << "\n" << endl;

		cout << "\n" << "   ---设备的多处理器信息---" << "\n" << endl;

		cout << "流处理器数量：" << prop.multiProcessorCount << "\n" << endl;
		cout << "一个线程块可使用的最大共享内存数量(字节):" << prop.sharedMemPerBlock << "\n" << endl;
		cout << "一个线程块可使用的32位寄存器数量：" << prop.regsPerBlock << "\n" << endl;
		cout << "一个线程束中包含的线程个数：" << prop.warpSize << "\n" << endl;
		cout << "一个线程块中包含的最大线程数量：" << prop.maxThreadsPerBlock << "\n" << endl;
		cout << "线程块(Block)维数：" << "(" << prop.maxThreadsDim[0] << "," << prop.maxThreadsDim[1] << "," << prop.maxThreadsDim[2] << ")" << "\n" << endl;
		cout << "线程格(Grid)维数：" << "(" << prop.maxGridSize[0] << "," << prop.maxGridSize[1] << "," << prop.maxGridSize[2] << ")" << "\n" << endl;

    }
	Sleep(200000);
	return 0;
}
