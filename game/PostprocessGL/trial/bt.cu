#include <cuda_runtime_api.h>
#include "cutil.h"
#include "cutil_inline_runtime.h"
#include "bt.h"
#include <iostream>

#define USE_SHARED_MEM 0

#define FILTER_SIZE (5*5) // 5x5 kernel filter
#define BLOCK_SIZE 16     // block size

__constant__ float kernelFilter_D[FILTER_SIZE];
__constant__ int indexOffsetsU_D[25];
__constant__ int indexOffsetsV_D[25];
__constant__ float invScale_D;
__constant__ float offset_D;

//extern struct balls;
#define GLEW_STATIC  //specify GLEW_STATIC to use the static linked library (.lib) instead of the dynamic linked library (.dll) for GLEW
#include <GL/glew.h>
#include <GL/glut.h>  // GLUT, includes glu.h and gl.h
extern int n;
struct balls  {
   GLfloat* ballRadius = (GLfloat*)malloc(n*sizeof(GLfloat*));  //radius
   GLfloat* ballX = (GLfloat*)malloc(n*sizeof(GLfloat*));
   GLfloat* ballY = (GLfloat*)malloc(n*sizeof(GLfloat*));
   GLfloat* xSpeed = (GLfloat*)malloc(n*sizeof(GLfloat*));
   GLfloat* ySpeed = (GLfloat*)malloc(n*sizeof(GLfloat*));
};


texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texRef;

__global__ void PostprocessKernel(struct balls B)
{
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int bw = blockDim.x;
    unsigned int bh = blockDim.y;
//    B.ballX[0]=0.0;
//    B.ballY[0]=0.0;
//    printf("tx: %d, ty: %d, x: %f, y:%f\n",tx,ty,B.ballX[tx],B.ballY[ty]);
//    printf("%d,%d,%d,%d,%f,%f!\n",tx,ty,bw,bh,B.ballX,B.ballY);
}

void PostprocessCUDA(struct balls B, int n)  {
    dim3 gridDim( 3,10,1);
    dim3 blockDim(2,3,1);
//    std::cout<<"quick: "<<B.ballX[0]<<std::endl;
    cudaHostAlloc(&(B.ballX),sizeof(GLfloat*)*n,0);
    cudaHostAlloc(&(B.ballY),sizeof(GLfloat*)*n,0);
    struct balls bg;
    bg.ballX=B.ballX;
    bg.ballY=B.ballY;
//    cudaMalloc(&Bg,n*sizeof(balls));
//    cudaMalloc(&Bg,n*);  
//    std::cout<<"quicki: "<<B.ballX[1]<<':'<<B.ballY[1]<<std::endl;
//    cudaMemcpy(Bg,B,n*sizeof(balls), cudaMemcpyHostToDevice);
//    std::cout<<"quickii: "<<B.ballX[2]<<':'<<B.ballY[2]<<std::endl;
    PostprocessKernel<<< gridDim, blockDim >>>(bg);
    cudaDeviceSynchronize();
}
