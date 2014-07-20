#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <Windows.h>
#include <time.h>

#include "createMatrix.h"
#include "handleError.h"
#include "matCompute_CPU.h"

using namespace std;

// 线程块的大小
#define BLOCK_SIZE 32

// 获得矩阵mat中(row, col)处的元素
// 进行判断是为了防止越界
// 比如：矩阵大小不是线程块整数倍时，开辟的线程是大于矩阵元素数的
__device__ float GetElement(const Matrix mat, int row, int col)
{
	if(row < mat.height && col < mat.width)
		return mat.elements[row * mat.stride + col];
	else
		return 0;
}

// 设置矩阵mat中(row, col)处的元素
// 进行判断是为了防止越界
// 比如：矩阵大小不是线程块整数倍时，开辟的线程是大于矩阵元素数的
__device__ void SetElement(Matrix mat, int row, int col, float value)
{
	if(row < mat.height && col < mat.width)
		mat.elements[row * mat.stride + col] = value;
}


// 矩阵相乘的核函数
// 每个线程计算C中的一个点
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
	// 坐标索引
	int x_id = blockDim.x * blockIdx.x + threadIdx.x;
	int y_id = blockDim.y * blockIdx.y + threadIdx.y;

	float Cvalue = 0;
	for(int i = 0; i < A.width; i++)
	{
		Cvalue += GetElement(A, y_id, i) * GetElement(B, i, x_id);
	}
	//if(Cvalue < 512)
	//	printf("x_id = %d,y_id = %d\n",x_id, y_id);
	//printf("Cvalue = %f\n",Cvalue);

	// 将计算结果写入C矩阵中
	SetElement(C, y_id, x_id, Cvalue);

}

// 矩阵相乘--主机代码
// 默认矩阵的行和列都是BLOCK_SIZE的整数倍
void MatMul(Matrix A, Matrix B, Matrix C)
{
	// 将矩阵A拷贝到显存中
	Matrix d_A;
	d_A = matParameterCopy(A);
	HANDLE_ERROR( cudaMalloc((void**)&d_A.elements, d_A.size), );
	HANDLE_ERROR( cudaMemcpy(d_A.elements, A.elements, d_A.size, cudaMemcpyHostToDevice) );

	// 将矩阵B拷贝到显存中
	Matrix d_B;
	d_B = matParameterCopy(B);
	HANDLE_ERROR( cudaMalloc((void**)&d_B.elements, d_B.size) );
	HANDLE_ERROR( cudaMemcpy(d_B.elements, B.elements, d_B.size, cudaMemcpyHostToDevice) );

	// 在显存中为C开辟空间
	Matrix d_C;
	d_C = matParameterCopy(C);
	HANDLE_ERROR( cudaMalloc((void**)&d_C.elements, d_C.size) );

	// 核函数
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((B.width + BLOCK_SIZE -1) / dimBlock.x, 
				 (A.height + BLOCK_SIZE -1) / dimBlock.y);
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

	// 将C矩阵从显存中读到主机内存中
	HANDLE_ERROR( cudaMemcpy(C.elements, d_C.elements, d_C.size, cudaMemcpyDeviceToHost) );

	// 释放显存空间
	HANDLE_ERROR( cudaFree(d_A.elements) );
	HANDLE_ERROR( cudaFree(d_B.elements) );
	HANDLE_ERROR( cudaFree(d_C.elements) );
}

int main()
{
	// 声明矩阵A、B、C并分配空间
	Matrix A, B, C_GPU, C_CPU;
	A = createMat(512, 512);
	B = createMat(512, 512);
	C_GPU = createMat(512, 512);
	C_CPU = createMat(512, 512);

	// 生成随机矩阵A、B
	matGen(A);
	matGen(B);

	// 记录起始时间
	cudaEvent_t start_GPU, end_GPU;
	cudaEventCreate(&start_GPU);
	cudaEventCreate(&end_GPU);
	cudaEventRecord(start_GPU, 0);

	// 矩阵相乘：C_GPU=A*B
	MatMul(A, B, C_GPU);
	
	cout << C_GPU.elements[0] << endl;
	cout << C_GPU.elements[512*512-1] << endl;

	// 计时结束
	cudaEventRecord(end_GPU, 0);
	cudaEventSynchronize(end_GPU);
	float elaspsedTime;
	cudaEventElapsedTime(&elaspsedTime, start_GPU, end_GPU);

	cout << "GPU的运行时间为：" << elaspsedTime << "ms." << endl;

	cudaEventDestroy(start_GPU);
	cudaEventDestroy(end_GPU);

	// 统计CPU运行时间
	clock_t start_CPU, end_CPU;

	start_CPU = clock();
	C_CPU = matMul_CPU(A, B);
	end_CPU = clock();

	cout << C_CPU.elements[0] << endl;
	cout << C_CPU.elements[512*512-1] << endl;

	double duration;
	duration = (double)(end_CPU - start_CPU)*512 / CLOCKS_PER_SEC; 
	cout<<"CPU的运行时间为："<< duration <<" ms."<<endl;
	
	// 计算GPU与CPU计算误差
	double error_CPU_GPU = 0;
	error_CPU_GPU = matSum_CPU(matSub_CPU(C_CPU, C_GPU));
	//error_CPU_GPU = matSum_CPU(C_GPU);
	cout << "GPU与CPU计算误差为：" << error_CPU_GPU << endl;

	//Sleep(512000);
	
	cudaThreadExit();
	return 0;
}
