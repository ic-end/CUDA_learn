#ifndef MATCOMPUTE_CPU_H
#define MATCOMPUTE_CPU_H

#include "createMatrix.h"

// ��CPU��ִ�о���ĳ˷�
// C = A * B
Matrix matMul_CPU(Matrix A, Matrix B);

// ��CPU��ִ�о���ļ���
// C = A - B
Matrix matSub_CPU(Matrix A, Matrix B);

// ��CPU��ִ�о�������
// value = sum(mat)
double matSum_CPU(Matrix mat);

#endif