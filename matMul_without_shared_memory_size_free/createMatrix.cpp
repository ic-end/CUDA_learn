#include "createMatrix.h"
#include <iostream>
using namespace std;

// 根据输入的行和列生成矩阵结构体
Matrix createMat(const int height, const int width)
{
	if( width<1 || height<1 )
	{
		cout << "请输入正确的矩阵行和列！" << endl;
		exit(0);
	}
	Matrix mat;
	mat.height = height;
	mat.width = width;
	mat.stride = width;
	mat.size = height * width * sizeof(float);
	mat.elements = (float*)malloc(mat.size);
	return mat;
}

// 将srcMat矩阵的参数复制给desMat
// 为复制矩阵首地址
// 是因为只是为了生成一个与原矩阵大小相同的矩阵
Matrix matParameterCopy(Matrix srcMat)
{
	Matrix desMat;
	desMat.height = srcMat.height;
	desMat.width = srcMat.width;
	desMat.size = srcMat.size;
	desMat.stride = srcMat.stride;
	return desMat;
}

// 产生浮点型(0-1)随机矩阵 
// width 列数 宽
// height 行数 高
bool matGen(Matrix mat)
{
	if( mat.width<1 || mat.height<1 )
	{
		cout << "请输入正确的矩阵行和列！" << endl;
		return false;
	}

    for( int i = 0; i < mat.height; i++ ) 
	{
        for( int j = 0; j < mat.width; j++ ) 
		{
			mat.elements[i * mat.width + j] = (float) rand() / RAND_MAX + 
                (float) rand() / (RAND_MAX * RAND_MAX);
			//mat.elements[i * mat.width + j] = 1;
        }
    }
	return true;
}

