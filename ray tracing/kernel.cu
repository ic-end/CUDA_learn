#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#include <iostream>
#include <Windows.h>

#include "cpu_bitmap.h"

#include "cudaErrorYoN.h"

using namespace std;

#define DIM 1024

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF     2e10f

struct Sphere 
{
    float   r,b,g; // �����ɫ
    float   radius;// ��İ뾶
    float   x,y,z; // �����������
	// �ṹ���еĺ��� ������c++�е���
    __device__ float hit( float ox, float oy, float *n ) 
	{
		// �ж�(ox, oy)�����Ƿ��������ཻ
		// ����ཻ������������������������洦�ľ���
		// ��Ϊ����������������ཻʱ��ֻ����ӽ������������ܿ���
        float dx = ox - x;
        float dy = oy - y;
        if (dx*dx + dy*dy < radius*radius) 
		{
            float dz = sqrtf( radius*radius - dx*dx - dy*dy );
            *n = dz / sqrtf( radius * radius );
            return dz + z;
        }
        return -INF;
    }
};

#define SPHERES 30

// __constant__ Sphere s[SPHERES];

__global__ void kernel( Sphere *s, unsigned char *ptr ) 
{
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    float   ox = (x - DIM/2);
    float   oy = (y - DIM/2);

    float   r=0, g=0, b=0;
    float   maxz = -INF;
	// �жϹ�����ÿ�������ཻ�����
    for(int i=0; i<SPHERES; i++) 
	{
        float   n;
        float   t = s[i].hit( ox, oy, &n );
        if (t > maxz) 
		{
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
            maxz = t;
        }
    } 

    ptr[offset*4 + 0] = (int)(r * 255);
    ptr[offset*4 + 1] = (int)(g * 255);
    ptr[offset*4 + 2] = (int)(b * 255);
    ptr[offset*4 + 3] = 255;
}

// globals needed by the update routine
struct DataBlock 
{
    unsigned char   *dev_bitmap;
	Sphere          *s;
};



int main( void ) 
{
    DataBlock   data;

    // ��¼��ʼʱ��
    cudaEvent_t     start, stop;
    cudaErrorYoN( cudaEventCreate( &start ), 4);
    cudaErrorYoN( cudaEventCreate( &stop ), 4);
    cudaErrorYoN( cudaEventRecord( start, 0 ), 4);

    CPUBitmap bitmap( DIM, DIM, &data );
    unsigned char   *dev_bitmap;
	Sphere          *s;

    // ��GPU�Ϸ����ڴ��Լ������λͼ
    cudaErrorYoN( cudaMalloc( (void**)&dev_bitmap,
                              bitmap.image_size() ), 1);
    // Ϊ Sphere���ݼ������ڴ�
    cudaErrorYoN( cudaMalloc( (void**)&s,
                              sizeof(Sphere) * SPHERES ), 1);

    // ������ʱ�ڴ棬��CPU�϶����ʼ���������Ƶ�GPU�ڴ��ϣ�Ȼ���ͷ��ڴ�
    Sphere *temp_s = (Sphere*)malloc( sizeof(Sphere) * SPHERES );

	// ΪSPHERES��Բ����λ�ã���ɫ���뾶��Ϣ
    for (int i=0; i<SPHERES; i++) 
	{
        temp_s[i].r = rnd( 1.0f );
        temp_s[i].g = rnd( 1.0f );
        temp_s[i].b = rnd( 1.0f );
        temp_s[i].x = rnd( 1000.0f ) - 500;
        temp_s[i].y = rnd( 1000.0f ) - 500;
        temp_s[i].z = rnd( 1000.0f ) - 500;
        temp_s[i].radius = rnd( 100.0f ) + 20;
    }

    cudaErrorYoN( cudaMemcpy( s, temp_s, 
                                sizeof(Sphere) * SPHERES, 
								cudaMemcpyHostToDevice), 2);
    free( temp_s );

    // generate a bitmap from our sphere data
    dim3    grids(DIM/16,DIM/16);
    dim3    threads(16,16);
    kernel<<<grids,threads>>>( s, dev_bitmap );

    // ��λͼ��GPU�ϸ��Ƶ�������
    cudaErrorYoN( cudaMemcpy( bitmap.get_ptr(), dev_bitmap,
                              bitmap.image_size(),
                              cudaMemcpyDeviceToHost ), 2);

    // ��¼����ʱ��
    cudaErrorYoN( cudaEventRecord( stop, 0 ), 4);
    cudaErrorYoN( cudaEventSynchronize( stop ), 4);

	// ��ʾ����ʱ��
    float   elapsedTime;
    cudaErrorYoN( cudaEventElapsedTime( &elapsedTime, start, stop ), 4); // ���������¼�֮���ʱ��
    printf( "Time to generate:  %3.1f ms\n", elapsedTime );

	// �����¼�
    cudaErrorYoN( cudaEventDestroy( start ), 4);
    cudaErrorYoN( cudaEventDestroy( stop ), 4);

	// �ͷ��ڴ�
    cudaErrorYoN( cudaFree( dev_bitmap ), 3);
	cudaErrorYoN( cudaFree( s ), 3);

    // ��ʾλͼ
    bitmap.display_and_exit();
}

