#include "createMatrix.h"
#include <iostream>
using namespace std;

// ����������к������ɾ���ṹ��
Matrix createMat(const int height, const int width)
{
	if( width<1 || height<1 )
	{
		cout << "��������ȷ�ľ����к��У�" << endl;
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

// ��srcMat����Ĳ������Ƹ�desMat
// Ϊ���ƾ����׵�ַ
// ����Ϊֻ��Ϊ������һ����ԭ�����С��ͬ�ľ���
Matrix matParameterCopy(Matrix srcMat)
{
	Matrix desMat;
	desMat.height = srcMat.height;
	desMat.width = srcMat.width;
	desMat.size = srcMat.size;
	desMat.stride = srcMat.stride;
	return desMat;
}

// ����������(0-1)������� 
// width ���� ��
// height ���� ��
bool matGen(Matrix mat)
{
	if( mat.width<1 || mat.height<1 )
	{
		cout << "��������ȷ�ľ����к��У�" << endl;
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

