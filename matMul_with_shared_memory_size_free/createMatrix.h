#ifndef CREATEMATRIX_H
#define CREATEMATRIX_H

// c��c++�еľ����������ȴ洢
// M.stride ��ƫ��
// �˴�����ƫ����Ϊ�˷���ȡ������ӿ�
// M.elements �����׵�ַ
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct 
			{
				int width;
				int height;
				int stride;
				size_t size;
				float* elements;
			} Matrix;

// ����������к������ɾ���ṹ��
Matrix createMat(const int height, const int width);

// ��srcMat����Ĳ������Ƹ�desMat
// Ϊ���ƾ����׵�ַ
// ����Ϊֻ��Ϊ������һ����ԭ�����С��ͬ�ľ���
Matrix matParameterCopy(Matrix srcMat);

// ����������(0-1)������� 
// width ���� ��
// height ���� ��
bool matGen(Matrix mat);

#endif