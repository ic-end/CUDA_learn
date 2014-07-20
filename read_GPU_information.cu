
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

		cout << "   --- ��" << i << "���豸��Ϣ ---" << "\n" << endl;
		cout << "�Կ����֣�" << prop.name << "\n" << endl;
		cout << "����������" << prop.major << "." << prop.minor << "\n" << endl;
		cout << "ʱ��Ƶ�ʣ�" << prop.clockRate << "\n" << endl;
        if (prop.deviceOverlap)
		{
			cout << "�Ƿ����ͬʱִ��cudaMemory()���ú�һ���˺������ã���!" << "\n" << endl;
		}
        else
		{
			cout << "�Ƿ����ͬʱִ��cudaMemory()���ú�һ���˺������ã���!" << "\n" << endl;
		}

        if (prop.kernelExecTimeoutEnabled)
		{
			cout << "�豸��ִ�еĺ˺����Ƿ��������ʱ���ƣ���!" << "\n" << endl;
		}
        else
		{
			cout << "�豸��ִ�еĺ˺����Ƿ��������ʱ���ƣ���!" << "\n" << endl;
		}

		cout << "\n" << "   ---�豸�ڴ���Ϣ---" << "\n" << endl;
		cout << "ȫ���ڴ��������ֽڣ���" << prop.totalGlobalMem << "\n" << endl;
		cout << "�����ڴ��������ֽڣ���" << prop.totalConstMem << "\n" << endl;
		cout << "�ڴ渴���е��������ֵ���ֽڣ���" << prop.memPitch << "\n" << endl;
		cout << "�豸���������Ҫ��" << prop.textureAlignment << "\n" << endl;

		cout << "\n" << "   ---�豸�Ķദ������Ϣ---" << "\n" << endl;

		cout << "��������������" << prop.multiProcessorCount << "\n" << endl;
		cout << "һ���߳̿��ʹ�õ�������ڴ�����(�ֽ�):" << prop.sharedMemPerBlock << "\n" << endl;
		cout << "һ���߳̿��ʹ�õ�32λ�Ĵ���������" << prop.regsPerBlock << "\n" << endl;
		cout << "һ���߳����а������̸߳�����" << prop.warpSize << "\n" << endl;
		cout << "һ���߳̿��а���������߳�������" << prop.maxThreadsPerBlock << "\n" << endl;
		cout << "�߳̿�(Block)ά����" << "(" << prop.maxThreadsDim[0] << "," << prop.maxThreadsDim[1] << "," << prop.maxThreadsDim[2] << ")" << "\n" << endl;
		cout << "�̸߳�(Grid)ά����" << "(" << prop.maxGridSize[0] << "," << prop.maxGridSize[1] << "," << prop.maxGridSize[2] << ")" << "\n" << endl;

    }
	Sleep(200000);
	return 0;
}
