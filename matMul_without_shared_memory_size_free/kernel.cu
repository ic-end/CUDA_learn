#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <Windows.h>
#include <time.h>

#include "createMatrix.h"
#include "handleError.h"
#include "matCompute_CPU.h"

using namespace std;

// �߳̿�Ĵ�С
#define BLOCK_SIZE 32

// ��þ���mat��(row, col)����Ԫ��
// �����ж���Ϊ�˷�ֹԽ��
// ���磺�����С�����߳̿�������ʱ�����ٵ��߳��Ǵ��ھ���Ԫ������
__device__ float GetElement(const Matrix mat, int row, int col)
{
	if(row < mat.height && col < mat.width)
		return mat.elements[row * mat.stride + col];
	else
		return 0;
}

// ���þ���mat��(row, col)����Ԫ��
// �����ж���Ϊ�˷�ֹԽ��
// ���磺�����С�����߳̿�������ʱ�����ٵ��߳��Ǵ��ھ���Ԫ������
__device__ void SetElement(Matrix mat, int row, int col, float value)
{
	if(row < mat.height && col < mat.width)
		mat.elements[row * mat.stride + col] = value;
}


// ������˵ĺ˺���
// ÿ���̼߳���C�е�һ����
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
	// ��������
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

	// ��������д��C������
	SetElement(C, y_id, x_id, Cvalue);

}

// �������--��������
// Ĭ�Ͼ�����к��ж���BLOCK_SIZE��������
void MatMul(Matrix A, Matrix B, Matrix C)
{
	// ������A�������Դ���
	Matrix d_A;
	d_A = matParameterCopy(A);
	HANDLE_ERROR( cudaMalloc((void**)&d_A.elements, d_A.size), );
	HANDLE_ERROR( cudaMemcpy(d_A.elements, A.elements, d_A.size, cudaMemcpyHostToDevice) );

	// ������B�������Դ���
	Matrix d_B;
	d_B = matParameterCopy(B);
	HANDLE_ERROR( cudaMalloc((void**)&d_B.elements, d_B.size) );
	HANDLE_ERROR( cudaMemcpy(d_B.elements, B.elements, d_B.size, cudaMemcpyHostToDevice) );

	// ���Դ���ΪC���ٿռ�
	Matrix d_C;
	d_C = matParameterCopy(C);
	HANDLE_ERROR( cudaMalloc((void**)&d_C.elements, d_C.size) );

	// �˺���
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((B.width + BLOCK_SIZE -1) / dimBlock.x, 
				 (A.height + BLOCK_SIZE -1) / dimBlock.y);
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

	// ��C������Դ��ж��������ڴ���
	HANDLE_ERROR( cudaMemcpy(C.elements, d_C.elements, d_C.size, cudaMemcpyDeviceToHost) );

	// �ͷ��Դ�ռ�
	HANDLE_ERROR( cudaFree(d_A.elements) );
	HANDLE_ERROR( cudaFree(d_B.elements) );
	HANDLE_ERROR( cudaFree(d_C.elements) );
}

int main()
{
	// ��������A��B��C������ռ�
	Matrix A, B, C_GPU, C_CPU;
	A = createMat(512, 512);
	B = createMat(512, 512);
	C_GPU = createMat(512, 512);
	C_CPU = createMat(512, 512);

	// �����������A��B
	matGen(A);
	matGen(B);

	// ��¼��ʼʱ��
	cudaEvent_t start_GPU, end_GPU;
	cudaEventCreate(&start_GPU);
	cudaEventCreate(&end_GPU);
	cudaEventRecord(start_GPU, 0);

	// ������ˣ�C_GPU=A*B
	MatMul(A, B, C_GPU);
	
	cout << C_GPU.elements[0] << endl;
	cout << C_GPU.elements[512*512-1] << endl;

	// ��ʱ����
	cudaEventRecord(end_GPU, 0);
	cudaEventSynchronize(end_GPU);
	float elaspsedTime;
	cudaEventElapsedTime(&elaspsedTime, start_GPU, end_GPU);

	cout << "GPU������ʱ��Ϊ��" << elaspsedTime << "ms." << endl;

	cudaEventDestroy(start_GPU);
	cudaEventDestroy(end_GPU);

	// ͳ��CPU����ʱ��
	clock_t start_CPU, end_CPU;

	start_CPU = clock();
	C_CPU = matMul_CPU(A, B);
	end_CPU = clock();

	cout << C_CPU.elements[0] << endl;
	cout << C_CPU.elements[512*512-1] << endl;

	double duration;
	duration = (double)(end_CPU - start_CPU)*512 / CLOCKS_PER_SEC; 
	cout<<"CPU������ʱ��Ϊ��"<< duration <<" ms."<<endl;
	
	// ����GPU��CPU�������
	double error_CPU_GPU = 0;
	error_CPU_GPU = matSum_CPU(matSub_CPU(C_CPU, C_GPU));
	//error_CPU_GPU = matSum_CPU(C_GPU);
	cout << "GPU��CPU�������Ϊ��" << error_CPU_GPU << endl;

	//Sleep(512000);
	
	cudaThreadExit();
	return 0;
}
