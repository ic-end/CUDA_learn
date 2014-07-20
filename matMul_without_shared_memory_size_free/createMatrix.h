#ifndef CREATEMATRIX_H
#define CREATEMATRIX_H

// c和c++中的矩阵都是行优先存储
// M.stride 行偏移
// 此处的行偏移是为了方便取矩阵的子块
// M.elements 矩阵首地址
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct 
			{
				int width;
				int height;
				int stride;
				size_t size;
				float* elements;
			} Matrix;

// 根据输入的行和列生成矩阵结构体
Matrix createMat(const int height, const int width);

// 将srcMat矩阵的参数复制给desMat
// 为复制矩阵首地址
// 是因为只是为了生成一个与原矩阵大小相同的矩阵
Matrix matParameterCopy(Matrix srcMat);

// 产生浮点型(0-1)随机矩阵 
// width 列数 宽
// height 行数 高
bool matGen(Matrix mat);

#endif