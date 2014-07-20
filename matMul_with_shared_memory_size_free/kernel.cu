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

// 获得mat的一个子矩阵matSub，大小为：BLOCK_SIZE * BLOCK_SIZE 
// 子矩阵块的坐标为（row， col）
__device__ Matrix GetSubMatrix(Matrix mat, int row, int col)
{
	Matrix matSub;
	matSub.width = BLOCK_SIZE;
	matSub.height = BLOCK_SIZE;
	matSub.stride = mat.stride; // 子矩阵与原矩阵的行偏移相同
	matSub.elements = &mat.elements[mat.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
	return matSub;
}

// 矩阵相乘的核函数
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
	// 块对应的行和列
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	// 得到C对应的一个子块Csub
	Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

	// 每一个线程计算Csub中的一个元素
	// 将结果存在Cvalue
	float Cvalue = 0;

	// 子块内线程的坐标
	int row = threadIdx.y;
	int col = threadIdx.x;

	// A的行子块 * B的列子块 = 对应C的子块Csub
	// 一个循环： A的行子块中的一个子块 * B的列子块中的一个子块 （相加）
	// 子块的循环
	for (int m = 0; m < ((A.width + BLOCK_SIZE -1) / BLOCK_SIZE); ++m) 
	{
		// 得到A中的一个子块Asub
		Matrix Asub = GetSubMatrix(A, blockRow, m);

		// 得到B中的一个子块Bsub
		Matrix Bsub = GetSubMatrix(B, m, blockCol);

		// 分配共享内存空间，用来存放Asub和Bsub
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		// 将Asub和Bsub拷贝到共享内存中
		// 每一个线程拷贝子块中的一个元素
		// 因为最后一行子块或最后一列子块可能未满
		// 因此进行判断，防止越界
		if((m * BLOCK_SIZE + col < A.width) &&
			(blockRow * BLOCK_SIZE + row < A.height))
		{
			As[row][col] = GetElement(Asub, row, col);
		}
		else
		{
			As[row][col] = 0;
		}

		if((blockCol * BLOCK_SIZE + col < B.width) &&
			(m * BLOCK_SIZE + row < B.height))
		{
			Bs[row][col] = GetElement(Bsub, row, col);
		}
		else
		{
			Bs[row][col] = 0;
		}

		// 对线程块中的线程进行同步，确保线程块中的每个线程都执行完
		// 此处同步是为了确保子矩阵都已经拷贝到共享内存中
		__syncthreads();

		// A子块的行*B子块的列
		// 子块内的循环
		for (int e = 0; e < BLOCK_SIZE; ++e)
		{
			Cvalue += As[row][e] * Bs[e][col];
		}

		// 同步,确保当前A子块与B子块的计算完成
		// 同步完成才将下个A子块与B子块拷贝的共享内存
		__syncthreads();
	}

	// C子块Csub的计算已完成
	// 每个线程写一个元素
	//SetElement(Csub, row, col, Cvalue);
	SetElement(C, blockRow * blockDim.y + row, blockCol * blockDim.x + col, Cvalue);
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
	//float* d_C.elements;
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
	A = createMat(1000, 1000);
	B = createMat(1000, 1000);
	C_GPU = createMat(1000, 1000);
	C_CPU = createMat(1000, 1000);

	// 生成随机矩阵A、B
	matGen(A);
	matGen(B);

	// 记录起始时间
	cudaEvent_t start_GPU, end_GPU;
	cudaEventCreate(&start_GPU);
	cudaEventCreate(&end_GPU);
	cudaEventRecord(start_GPU, 0);

	// 矩阵相乘：C=A*B
	MatMul(A, B, C_GPU);

	cout << C_GPU.elements[0] << endl;
	cout << C_GPU.elements[1000*1000-1] << endl;

	// 计时结束
	cudaEventRecord(end_GPU, 0);
	cudaEventSynchronize(end_GPU);
	float elaspsedTime;
	cudaEventElapsedTime(&elaspsedTime, start_GPU, end_GPU);

	cout << "GPU的运行时间为：" << elaspsedTime << "ms" << endl;

	cudaEventDestroy(start_GPU);
	cudaEventDestroy(end_GPU);
	
	// 统计CPU运行时间
	clock_t start_CPU, end_CPU;

	start_CPU = clock();
	C_CPU = matMul_CPU(A, B);
	end_CPU = clock();

	double duration;
	duration = (double)(end_CPU - start_CPU)*1000 / CLOCKS_PER_SEC; 
	cout<<"CPU的运行时间为："<<duration<<" ms."<<endl;

	// 计算GPU与CPU计算误差
	double error_CPU_GPU = 0;
	error_CPU_GPU = matSum_CPU(matSub_CPU(C_CPU, C_GPU));
	//error_CPU_GPU = matSum_CPU(C_GPU);
	cout << "GPU与CPU计算误差为：" << error_CPU_GPU << endl;

	Sleep(200000);
	return 0;
}
