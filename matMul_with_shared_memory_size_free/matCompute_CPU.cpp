#include "matCompute_CPU.h"
#include <iostream>

using namespace std;

// 获得矩阵mat中(row,col)处的元素
float getMatElement(Matrix mat, const int row, const int col)
{
	float matValue = 0;
	if(mat.height < 1 || mat.width < 1 || row < 0 || col <0)
	{
		cout << "请出入正确的参数！" << endl;
		exit(0);
	}
	else
	{
		matValue = mat.elements[row * mat.stride + col];
	}

	return matValue;
}

// 在CPU上执行矩阵的乘法
// C = A * B
// 三层循环
Matrix matMul_CPU(Matrix A, Matrix B)
{
	Matrix C;
	C = createMat(A.height, B.width);

	for(int i = 0; i < C.height; i++)
	{		
		for(int j = 0; j< C.width; j++)
		{
			float Cvalue = 0;
			for(int k = 0; k < A.width; k++)
			{
				Cvalue += getMatElement(A, i, k) * getMatElement(B, k, j);
			}
			C.elements[i * C.stride + j] = Cvalue;
		}
	}
	return C;
}

// 在CPU上执行矩阵的减法
// C = A - B
// 二层循环
Matrix matSub_CPU(Matrix A, Matrix B)
{
	Matrix C;
	C = createMat(A.height, A.width);

	for(int i = 0; i < C.height; i++)
	{
		for(int j = 0; j< C.width; j++)
		{
			C.elements[i * C.stride + j] = getMatElement(A, i, j)
											- getMatElement(B, i, j);
		}
	}
	return C;
}

// 在CPU上执行矩阵的求和
// value = sum(mat)
// 二层循环
double matSum_CPU(Matrix mat)
{
	double valueSum = 0;
	for(int i = 0; i < mat.height; i++)
	{
		for(int j = 0; j< mat.width; j++)
		{
			valueSum += abs(double(getMatElement(mat, i, j)));
		}
	}
	return valueSum;
}
