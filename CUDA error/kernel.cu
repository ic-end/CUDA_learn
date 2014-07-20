#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
//#include "sm_12_atomic_functions.h"

#include <iostream>
#include <Windows.h>
#include <time.h>

#include "cudaErrorYoN.h"

using namespace std;

#define SIZE    (100*1024*1024)
#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

void* big_random_block( int size ) {
    unsigned char *data = (unsigned char*)malloc( size );
    HANDLE_NULL( data );
    for (int i=0; i<size; i++)
        data[i] = rand();

    return data;
}

__global__ void histo_kernel( unsigned char *buffer,
                              long size,
                              unsigned int *histo ) {
    // calculate the starting index and the offset to the next
    // block that each thread will be processing
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (i < size) {
        atomicAdd( &histo[buffer[i]], 1 );
        i += stride;
    }
}

int main( void ) {
    unsigned char *buffer =
                     (unsigned char*)big_random_block( SIZE );

    // capture the start time
    // starting the timer here so that we include the cost of
    // all of the operations on the GPU.
    cudaEvent_t     start, stop;
    cudaErrorYoN( cudaEventCreate( &start ), 4);
    cudaErrorYoN( cudaEventCreate( &stop ), 4);
    cudaErrorYoN( cudaEventRecord( start, 0 ), 4);

    // allocate memory on the GPU for the file's data
    unsigned char *dev_buffer;
    unsigned int *dev_histo;
    cudaErrorYoN( cudaMalloc( (void**)&dev_buffer, SIZE ), 1);
    cudaErrorYoN( cudaMemcpy( dev_buffer, buffer, SIZE,
                              cudaMemcpyHostToDevice ), 1);

    cudaErrorYoN( cudaMalloc( (void**)&dev_histo,
                              256 * sizeof( int ) ), 1);
    cudaErrorYoN( cudaMemset( dev_histo, 0,
                              256 * sizeof( int ) ), 1);

    // kernel launch - 2x the number of mps gave best timing
    cudaDeviceProp  prop;
    cudaErrorYoN( cudaGetDeviceProperties( &prop, 0 ), 4);
    int blocks = prop.multiProcessorCount;
    histo_kernel<<<blocks*2,256>>>( dev_buffer, SIZE, dev_histo );
    
    unsigned int    histo[256];
    cudaErrorYoN( cudaMemcpy( histo, dev_histo,
                              256 * sizeof( int ),
                              cudaMemcpyDeviceToHost ), 2);

    // get stop time, and display the timing results
    cudaErrorYoN( cudaEventRecord( stop, 0 ), 4);
    cudaErrorYoN( cudaEventSynchronize( stop ), 4);
    float   elapsedTime;
    cudaErrorYoN( cudaEventElapsedTime( &elapsedTime,
                                        start, stop ), 4);
    printf( "Time to generate:  %3.1f ms\n", elapsedTime );

    long histoCount = 0;
    for (int i=0; i<256; i++) {
        histoCount += histo[i];
    }
    printf( "Histogram Sum:  %ld\n", histoCount );

    // verify that we have the same counts via CPU
    for (int i=0; i<SIZE; i++)
        histo[buffer[i]]--;
    for (int i=0; i<256; i++) {
        if (histo[i] != 0)
            printf( "Failure at %d!  Off by %d\n", i, histo[i] );
    }

    cudaErrorYoN( cudaEventDestroy( start ), 4);
    cudaErrorYoN( cudaEventDestroy( stop ), 4);
    cudaFree( dev_histo );
    cudaFree( dev_buffer );
    free( buffer );
	Sleep(2000);
    return 0;
}