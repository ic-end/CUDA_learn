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

// ���mat��һ���Ӿ���matSub����СΪ��BLOCK_SIZE * BLOCK_SIZE 
// �Ӿ���������Ϊ��row�� col��
__device__ Matrix GetSubMatrix(Matrix mat, int row, int col)
{
	Matrix matSub;
	matSub.width = BLOCK_SIZE;
	matSub.height = BLOCK_SIZE;
	matSub.stride = mat.stride; // �Ӿ�����ԭ�������ƫ����ͬ
	matSub.elements = &mat.elements[mat.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
	return matSub;
}

// ������˵ĺ˺���
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
	// ���Ӧ���к���
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	// �õ�C��Ӧ��һ���ӿ�Csub
	Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

	// ÿһ���̼߳���Csub�е�һ��Ԫ��
	// ���������Cvalue
	float Cvalue = 0;

	// �ӿ����̵߳�����
	int row = threadIdx.y;
	int col = threadIdx.x;

	// A�����ӿ� * B�����ӿ� = ��ӦC���ӿ�Csub
	// һ��ѭ���� A�����ӿ��е�һ���ӿ� * B�����ӿ��е�һ���ӿ� ����ӣ�
	// �ӿ��ѭ��
	for (int m = 0; m < ((A.width + BLOCK_SIZE -1) / BLOCK_SIZE); ++m) 
	{
		// �õ�A�е�һ���ӿ�Asub
		Matrix Asub = GetSubMatrix(A, blockRow, m);

		// �õ�B�е�һ���ӿ�Bsub
		Matrix Bsub = GetSubMatrix(B, m, blockCol);

		// ���乲���ڴ�ռ䣬�������Asub��Bsub
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		// ��Asub��Bsub�����������ڴ���
		// ÿһ���߳̿����ӿ��е�һ��Ԫ��
		// ��Ϊ���һ���ӿ�����һ���ӿ����δ��
		// ��˽����жϣ���ֹԽ��
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

		// ���߳̿��е��߳̽���ͬ����ȷ���߳̿��е�ÿ���̶߳�ִ����
		// �˴�ͬ����Ϊ��ȷ���Ӿ����Ѿ������������ڴ���
		__syncthreads();

		// A�ӿ����*B�ӿ����
		// �ӿ��ڵ�ѭ��
		for (int e = 0; e < BLOCK_SIZE; ++e)
		{
			Cvalue += As[row][e] * Bs[e][col];
		}

		// ͬ��,ȷ����ǰA�ӿ���B�ӿ�ļ������
		// ͬ����ɲŽ��¸�A�ӿ���B�ӿ鿽���Ĺ����ڴ�
		__syncthreads();
	}

	// C�ӿ�Csub�ļ��������
	// ÿ���߳�дһ��Ԫ��
	//SetElement(Csub, row, col, Cvalue);
	SetElement(C, blockRow * blockDim.y + row, blockCol * blockDim.x + col, Cvalue);
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
	//float* d_C.elements;
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
	A = createMat(1000, 1000);
	B = createMat(1000, 1000);
	C_GPU = createMat(1000, 1000);
	C_CPU = createMat(1000, 1000);

	// �����������A��B
	matGen(A);
	matGen(B);

	// ��¼��ʼʱ��
	cudaEvent_t start_GPU, end_GPU;
	cudaEventCreate(&start_GPU);
	cudaEventCreate(&end_GPU);
	cudaEventRecord(start_GPU, 0);

	// ������ˣ�C=A*B
	MatMul(A, B, C_GPU);

	cout << C_GPU.elements[0] << endl;
	cout << C_GPU.elements[1000*1000-1] << endl;

	// ��ʱ����
	cudaEventRecord(end_GPU, 0);
	cudaEventSynchronize(end_GPU);
	float elaspsedTime;
	cudaEventElapsedTime(&elaspsedTime, start_GPU, end_GPU);

	cout << "GPU������ʱ��Ϊ��" << elaspsedTime << "ms" << endl;

	cudaEventDestroy(start_GPU);
	cudaEventDestroy(end_GPU);
	
	// ͳ��CPU����ʱ��
	clock_t start_CPU, end_CPU;

	start_CPU = clock();
	C_CPU = matMul_CPU(A, B);
	end_CPU = clock();

	double duration;
	duration = (double)(end_CPU - start_CPU)*1000 / CLOCKS_PER_SEC; 
	cout<<"CPU������ʱ��Ϊ��"<<duration<<" ms."<<endl;

	// ����GPU��CPU�������
	double error_CPU_GPU = 0;
	error_CPU_GPU = matSum_CPU(matSub_CPU(C_CPU, C_GPU));
	//error_CPU_GPU = matSum_CPU(C_GPU);
	cout << "GPU��CPU�������Ϊ��" << error_CPU_GPU << endl;

	Sleep(200000);
	return 0;
}
