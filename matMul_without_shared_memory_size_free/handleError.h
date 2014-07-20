#ifndef	HANDLERROR_H
#define HANDLERROR_H

#include "cuda_runtime.h"
#include <iostream>

using namespace std;
// cuda函数错误判断语句
// 确保cuda程序正确执行
static void HandleError( cudaError_t err, const char *file, int line ) 
{
    if (err != cudaSuccess) 
	{
		cout << "在" << file << "的第" << line << "行出现错误！" << endl;
		cout << "错误代码为：" << cudaGetErrorString( err ) << endl;
    }
}
// 宏定义
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#endif