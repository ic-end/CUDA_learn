#ifndef	HANDLERROR_H
#define HANDLERROR_H

#include "cuda_runtime.h"
#include <iostream>

using namespace std;
// cuda���������ж����
// ȷ��cuda������ȷִ��
static void HandleError( cudaError_t err, const char *file, int line ) 
{
    if (err != cudaSuccess) 
	{
		cout << "��" << file << "�ĵ�" << line << "�г��ִ���" << endl;
		cout << "�������Ϊ��" << cudaGetErrorString( err ) << endl;
    }
}
// �궨��
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#endif