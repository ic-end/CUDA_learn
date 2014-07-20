#ifndef MATCOMPUTE_CPU_H
#define MATCOMPUTE_CPU_H

#include "createMatrix.h"

// 在CPU上执行矩阵的乘法
// C = A * B
Matrix matMul_CPU(Matrix A, Matrix B);

// 在CPU上执行矩阵的减法
// C = A - B
Matrix matSub_CPU(Matrix A, Matrix B);

// 在CPU上执行矩阵的求和
// value = sum(mat)
double matSum_CPU(Matrix mat);

#endif