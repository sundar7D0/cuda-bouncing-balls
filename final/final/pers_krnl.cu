//To compile: nvcc pers_krnl.cu -o gbb.out -I /usr/local/cuda-10.1/targets/x86_64-linux/include/ -lcudart -lglut -lGLU -lGL -lGLEW
#include <math.h>  //importing libraries
#include <iostream>
#include "cutil.h"
#include "cutil_inline_runtime.h"
#include <iostream>
#define GLEW_STATIC  //to use the static linked library (.lib) instead of the dynamic linked library (.dll) for GLEW
#include <GL/glew.h>
#include <GL/glut.h>  // GLUT includes glu.h and gl.h
#include <cuda_runtime_api.h>  //CUDA headers
#include <cuda_gl_interop.h>

#define USE_SHARED_MEM 0  //global variables
#define BLOCK_SIZE 16     // block size
#define PI 3.14159265f
#define SRC_BUFFER  0
#define DST_BUFFER  1
#define epsilon 1e-14
#define kx 2
#define ky 2
char title[] = "2D Bouncing Balls";  //windowed mode's title
int windowWidth  = 640;  //windowed mode's width
int windowHeight = 480;  //windowed mode's height
int windowPosX   = 50;  //windowed mode's top-left corner x
int windowPosY   = 50;  //windowed mode's top-left corner y
GLfloat ballRadius = 0.05f;  //ball radius
GLfloat xMax, xMin, yMax, yMin;  //ball's center (x, y) bounds
int refreshMillis = 30;  //refresh period in milliseconds
int max_threadsx=32, max_threadsy=32, max_blocksx=1000, max_blocksy=1000;  //check max_blocks!
GLfloat widthx, widthy;
int *pers_krnl;
int n_x, n_y, n_threadsx, n_threadsy;  //generalize!
int n_blocksx, n_blocksy;
int temp_n_blocksx, temp_n_blocksy;
int halo_x, n_halo_threadsx;
int halo_y, n_halo_threadsy;
int num_cellsx, num_cellsy;
int* worklist;
int* num_works; 
int total=1;
GLfloat energy=0.0;
GLdouble clipAreaXLeft, clipAreaXRight, clipAreaYBottom, clipAreaYTop;  //projection clipping area
struct balls  {
   GLfloat* ballRadius;  // = (GLfloat*)malloc(n*sizeof(GLfloat*));
   GLfloat* x;  // = (GLfloat*)malloc(n*sizeof(GLfloat*));
   GLfloat* y;  // = (GLfloat*)malloc(n*sizeof(GLfloat*));
   GLfloat* xSpeed;  // = (GLfloat*)malloc(n*sizeof(GLfloat*));
   GLfloat* ySpeed;  // = (GLfloat*)malloc(n*sizeof(GLfloat*));
}B;
int n=50;// kx=2, ky=2;  //hyper-parameters
int start=1;
int renew=0, renew_max=100;
int* worklist_new;
int* num_works_new;

#define ERROR_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__);}  //CUDA ERROR_CHECK inline-function
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void page_mem_alloc()  //page-memory allocation to B, pers_krnl 
{
   ERROR_CHECK(cudaHostAlloc((GLfloat**)&B.ballRadius,sizeof(GLfloat)*n,0));
   ERROR_CHECK(cudaHostAlloc((GLfloat**)&B.x,sizeof(GLfloat)*n,0));
   ERROR_CHECK(cudaHostAlloc((GLfloat**)&B.y,sizeof(GLfloat)*n,0));
   ERROR_CHECK(cudaHostAlloc((GLfloat**)&B.xSpeed,sizeof(GLfloat)*n,0));
   ERROR_CHECK(cudaHostAlloc((GLfloat**)&B.ySpeed,sizeof(GLfloat)*n,0));
   ERROR_CHECK(cudaHostAlloc((int**)&pers_krnl, sizeof(int), 0));  //   ++*pers_krnl;//=1;
}

bool valid_ball(int i)  //check for validity of a new ball wrt previous balls
{
   for(int j=0;j<i;j++)
   {
      float dist = sqrt(pow(B.x[i]-B.x[j],2)+pow(B.y[i]-B.y[j],2));  //distance between centers
      if (dist<B.ballRadius[i]+B.ballRadius[j])
         return 0;
   }
   return 1;
}

void initBalls()  //randomly initialize struct members of ball B
{
   for(int i=0;i<n;i++)
   {
      B.ballRadius[i] = ballRadius;
      B.x[i] = 1.0*((static_cast <float> (rand()) - (static_cast <float> (RAND_MAX)/2.0)) / static_cast <float> (RAND_MAX));  //x
      B.y[i] = 1.0*((static_cast <float> (rand()) - (static_cast <float> (RAND_MAX)/2.0)) / static_cast <float> (RAND_MAX));  //y
      while(!valid_ball(i))
      {
         renew+=1;
         B.ballRadius[i] = ballRadius;
         B.x[i] = 1.0*((static_cast <float> (rand()) - (static_cast <float> (RAND_MAX)/2.0)) / static_cast <float> (RAND_MAX));  //x
         B.y[i] = 1.0*((static_cast <float> (rand()) - (static_cast <float> (RAND_MAX)/2.0)) / static_cast <float> (RAND_MAX));  //y
         if(renew>renew_max)
         {
            std::cout<<"Improbable to create more new balls. Reduce 'number of balls' or 'radius'!"<<std::endl;
            exit(0);
         }
      }
      renew=0;
      B.xSpeed[i] = 0.03*((static_cast <float> (rand()) - (static_cast <float> (RAND_MAX)/2.0)) / static_cast <float> (RAND_MAX));  //vx
      B.ySpeed[i] = 0.03*((static_cast <float> (rand()) - (static_cast <float> (RAND_MAX)/2.0)) / static_cast <float> (RAND_MAX));  //vy
   }
}

void draw_circle(GLfloat x, GLfloat y, GLfloat r)  //draw a circle (ball) using triangular fan
{
   int numSegments = 10000;
   GLfloat angle;
   glTranslatef(x,y,0.0f);  //translate to (xPos, yPos)
   glBegin(GL_TRIANGLE_FAN);  //triangle-fan to create circle
   glColor3f(0.0f, 0.0f, 1.0f);  //color
   glVertex2f(0.0f, 0.0f);  //center of circle
   for (int j = 0; j <= numSegments; j++)  //= as last vertex same as first vertex
   {
      angle = j * 2.0f * PI / numSegments;  //360` for all segments
      glVertex2f(cos(angle)*r,sin(angle)*r);
   }
   glEnd();
}

__global__ void processKernel(int* pers_krnl,struct balls B,int* worklist,int* worklist_new,int* num_works,int* num_works_new,int num_cellsx,int num_cellsy,int halo_x,int n_blocksx,int n_halo_threadsx,int halo_y,int n_blocksy,int n_halo_threadsy,float xMax,float xMin,float yMax,float yMin,GLfloat widthx,GLfloat widthy,int* counter)  //CUDA kernel for intensive computations
{
   __shared__ bool x_, x__, y_, y__;  //shared only among a block
   x_=(halo_x==0 || (halo_x==1 && blockIdx.x<n_blocksx-1));x__=(halo_x==1 && blockIdx.x==n_blocksx-1);
   y_=(halo_y==0 || (halo_y==1 && blockIdx.y<n_blocksy-1));y__=(halo_y==1 && blockIdx.y==n_blocksy-1);
   __shared__ int max_counter;
   max_counter=gridDim.x*gridDim.y-1; 
   unsigned int i = threadIdx.x+blockIdx.x*blockDim.x;  //private
   unsigned int j = threadIdx.y+blockIdx.y*blockDim.y;
   unsigned int temp, p, i_new, j_new;
   if((x_ || x__  && threadIdx.x<n_halo_threadsx) && (y_ || y__ && threadIdx.y<n_halo_threadsy))  
      {
         num_works_new[i+num_cellsx*j]=num_works[i+num_cellsx*j];
         for(int k=0;k<kx*ky;k++)
            worklist_new[kx*ky*i+num_cellsx*kx*ky*j+k]=worklist[kx*ky*i+num_cellsx*kx*ky*j+k];
         while(*pers_krnl)
         {
            if(threadIdx.x==0 && blockIdx.x==0 && threadIdx.y==0 && blockIdx.y==0)
            {   
            	printf("b.y1: %f,%f,%f,%f,%f\n",B.y[0],yMax,yMin,xMax,xMin);
            	printf("inside gpu: %d\n",*pers_krnl);
        	}
            while(*pers_krnl!=2)
 
            {
	            if(threadIdx.x==0 && blockIdx.x==0 && threadIdx.y==0 && blockIdx.y==0)
	            {   
	            	printf("inside while: %d,%d,%d,%d,%d\n",threadIdx.x,blockIdx.x,threadIdx.y,blockIdx.y,*pers_krnl);
	        	}            	
           }

            if(threadIdx.x==0 && blockIdx.x==0 && threadIdx.y==0 && blockIdx.y==0)
	            {   
	            	printf("outside while: %d,%d,%d,%d,%d\n",threadIdx.x,blockIdx.x,threadIdx.y,blockIdx.y,*pers_krnl);
	        	}            	
            printf("weeklyy!\n");
            for(int p_=0;p_<min(num_works[i+num_cellsx*j],kx*ky);p_++)
            {
//	            printf("byp:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: %f\n",B.y[p]);            	
               p=worklist[kx*ky*i+num_cellsx*kx*ky*j+p_];
               printf("in %d:%d,%d\n",p,p_,*pers_krnl);
               B.x[p] += B.xSpeed[p];
               B.y[p] += B.ySpeed[p];
               if (B.y[p] > yMax)
               {
                  B.y[p] = yMax;
                  B.ySpeed[p] = -B.ySpeed[p];
               }
               else if (B.y[p] < yMin)
               {
                  B.y[p] = yMin;
                  B.ySpeed[p] = -B.ySpeed[p];
               }
               if (B.x[p] > xMax)
               {
                  B.x[p] = xMax;
                  B.xSpeed[p] = -B.xSpeed[p];
               } 
               else if (B.x[p] < xMin)
               {
                  B.x[p] = xMin;
                  B.xSpeed[p] = -B.xSpeed[p];
               }
               i_new=min(int((B.x[p]+abs(xMin))/widthx),num_cellsx-1);
               j_new=min(int((B.y[p]+abs(yMin))/widthy),num_cellsy-1);
               temp=atomicAdd(&num_works_new[i_new+num_cellsx*j_new],1);  //num_works_new[i_new+num_cellsx*j_new]+=1; done atomically!
               if(temp-1<kx*ky)
                  worklist_new[kx*ky*i_new+num_cellsx*kx*ky*j_new+temp-1]=p;
            }
            if(threadIdx.x==0 && threadIdx.y==0){
            	printf("b.y2: %f\n",B.y[0]);
            	printf("small!\n");
            	printf("max_counter1: %d:%d\n",max_counter,*counter);
                atomicInc((unsigned int*)counter,(unsigned int)max_counter);//atomicInc(&counter,max_counter);
            	printf("max_counter2: %d:%d\n",max_counter,*counter);
            }
            __syncthreads();
            printf("1.i:%d,j:%d,counter:%d\n",i,j,*counter);
//            __threadfence();  //to update 'worklist', 'num_works' among all threads
            while(*counter);  //global_barrier
            num_works[i+num_cellsx*j]=num_works_new[i+num_cellsx*j];
            num_works_new[i+num_cellsx*j]=0;
            for(int k=0;k<kx*ky;k++)
               worklist[kx*ky*i+num_cellsx*kx*ky*j+k]=worklist_new[kx*ky*i+num_cellsx*kx*ky*j+k];
            if(threadIdx.x==0 && threadIdx.y==0)
            {
                        	printf("b.y3: %f\n",B.y[0]);

               atomicInc((unsigned int*)counter,(unsigned int)max_counter);//atomicInc(&counter,max_counter);
            	if(blockIdx.x==0 && blockIdx.y==0)
            	{
	               printf("great gpu: %d\n",*pers_krnl);
    	           	*pers_krnl=1;
        		    printf("3.pers_krnlish: %d\n",*pers_krnl);            		
            	}
            }
            __syncthreads();
            printf("2.i:%d,j:%d,counter:%d\n",i,j,*counter);
//            __threadfence();  //to update 'worklist', 'num_works' among all threads
            while(*counter);  //global_barrier
            printf("3.pers_krnl: %d\n",*pers_krnl);
/*
      GLfloat dist, dist_sin, dist_cos, mid_pt_x, mid_pt_y, m1, m2, speed1, speed1_cos, speed1_sin, speed2, speed2_cos, speed2_sin;
      int o, p;
      for (int m=max(0,i-1);m<=min(i+1,num_cellsx-1);m++)
         for (int n=max(0,j-1);n<=min(j+1,num_cellsy-1);n++)
            for (int p_=0;p_<num_works[i+j*num_cellsx];p_++)
               for (int o_=0; o_<num_works[m+n*num_cellsx];o_++)  //try interchanging this and above 2 loops  
               {
                  o=worklist[kx*ky*m+num_cellsx*kx*ky*n+o_];
                  p=worklist[kx*ky*i+num_cellsx*kx*ky*j+p_];
                  dist=sqrt(pow(B.x[o]-B.x[p],2)+pow(B.y[o]-B.y[p],2));  //identifying overlaps with neighbours using distance between centers
                  printf("\ndist: %d,%d,%d,%d,%f,%f\n",m,n,i,j,dist,B.ballRadius[o]+B.ballRadius[p]);
                  if (dist<B.ballRadius[o]+B.ballRadius[p] && o!=p)
                  {
                     dist_sin = (B.y[o]-B.y[p])/(dist+epsilon);  //p=i,o=j
                     dist_cos = (B.x[o]-B.x[p])/(dist+epsilon);
                     mid_pt_x = (B.x[o]+B.x[p])/2.0;  //exact values depends on u1, u2
                     mid_pt_y = (B.y[o]+B.y[p])/2.0;  //exact values depends on u1, u2
//                     printf("duplicated!$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n");
                     printf("p: %f; o: %f\n",p,o);
                     B.x[p]=mid_pt_x-B.ballRadius[p]*dist_cos;  //position change after overlap
                     B.y[p]=mid_pt_y-B.ballRadius[p]*dist_sin;
//                     B.x[o]=2*mid_pt_x-B.x[p];
//                     B.y[o]=2*mid_pt_y-B.y[p];
                     m1=pow(B.ballRadius[p],3);  //asuming constant density(p) for all balls
                     m2=pow(B.ballRadius[p],3);
                     speed1=sqrt(pow(B.xSpeed[p],2)+pow(B.ySpeed[p],2))+epsilon;
                     speed1_cos=B.xSpeed[p]/speed1;
                     speed1_sin=B.ySpeed[p]/speed1;
                     speed2=sqrt(pow(B.xSpeed[o],2)+pow(B.ySpeed[o],2))+epsilon;
                     speed2_cos=B.xSpeed[o]/speed2;
                     speed2_sin=B.ySpeed[o]/speed2;
                     B.xSpeed[p]=(speed1*(speed1_cos*dist_cos+speed1_sin*dist_sin)*(m1-m2)+2*m2*speed2*(speed2_cos*dist_cos+speed2_sin*dist_sin))*dist_cos/(m1+m2)-speed1*(speed1_sin*dist_cos-speed1_cos*dist_sin)*dist_sin;  //B.xSpeed[p]=-B.xSpeed[p];
                     B.ySpeed[p]=(speed1*(speed1_cos*dist_cos+speed1_sin*dist_sin)*(m1-m2)+2*m2*speed2*(speed2_cos*dist_cos+speed2_sin*dist_sin))*dist_sin/(m1+m2)+speed1*(speed1_sin*dist_cos-speed1_cos*dist_sin)*dist_cos;
//                     B.xSpeed[o]=(speed2*(speed2_cos*dist_cos+speed2_sin*dist_sin)*(m2-m1)+2*m1*speed1*(speed1_cos*dist_cos+speed1_sin*dist_sin))*dist_cos/(m2+m1)-speed2*(speed2_sin*dist_cos-speed2_cos*dist_sin)*dist_sin;
//                     B.ySpeed[o]=(speed2*(speed2_cos*dist_cos+speed2_sin*dist_sin)*(m2-m1)+2*m1*speed1*(speed1_cos*dist_cos+speed1_sin*dist_sin))*dist_sin/(m2+m1)+speed2*(speed2_sin*dist_cos-speed2_cos*dist_sin)*dist_cos;
                  }
             	}
*/
         }
      }
/*   
    if (i==-1 && j==-1)
   {
//      float dist = sqrt(pow(B.x[i]-B.x[j],2)+pow(B.y[i]-B.y[j],2));  //distance between centers
//      printf("dis: %f",dist);
   for(int c=0;c<num_cellsx;c++)
      for(int d=0;d<num_cellsy;d++)
         for(int k=0;k<kx*ky;k++)
            printf("gpu: num_works:%d,c:%d,d:%d,k:%d,wl:%d\n",num_works[c+num_cellsx*d],c,d,k,worklist[kx*ky*c+num_cellsx*kx*ky*d+k]);
   int sum=0;
   for(int k=0;k<num_cellsx*num_cellsy;k++)
      sum+=num_works[k];
   printf("sum: %d\n",sum);
   }
//   printf("t/f:%b",((~halo_x || (halo_x==1 && blockIdx.x<n_blocksx-1) || (halo_x==1 && blockIdx.x==n_blocksx-1 && threadIdx.x<n_halo_threadsx)) && (~halo_y || (halo_y && blockIdx.y<n_blocksy-1) || (halo_y && blockIdx.y==n_blocksy-1 && threadIdx.y<n_halo_threadsy))));
*/
/*
   if((~halo_x || (halo_x==1 && blockIdx.x<n_blocksx-1) || (halo_x==1 && blockIdx.x==n_blocksx-1 && threadIdx.x<n_halo_threadsx)) && (~halo_y || (halo_y && blockIdx.y<n_blocksy-1) || (halo_y && blockIdx.y==n_blocksy-1 && threadIdx.y<n_halo_threadsy)))  
   {
      GLfloat dist;
      for (int m=max(0,i-1);m<=min(i+1,num_cellsx-1);m++)
         for (int n=max(0,j-1);n<=min(j+1,num_cellsy-1);n++)
            for (int p=0;p<num_works[i+j*num_cellsx];p++)
               for (int o=0; o<num_works[m+n*num_cellsx];o++)  //try interchanging this and above 2 loops  
               {
                  o=worklist[kx*ky*m+num_cellsx*kx*ky*n+o];
                  p=worklist[kx*ky*i+num_cellsx*kx*ky*j+p];
                  dist=sqrt(pow(B.x[o]-B.x[p],2)+pow(B.y[o]-B.y[p],2));  //identifying overlaps with neighbours using distance between centers
                  printf("\ndist: %f\n",dist);
                  if (dist<B.ballRadius[o]+B.ballRadius[p] && o!=p)
                  {
                     GLfloat dist_sin = (B.y[o]-B.y[p])/(dist+epsilon);  //p=i,o=j
                     GLfloat dist_cos = (B.x[o]-B.x[p])/(dist+epsilon);
                     GLfloat mid_pt_x = (B.x[o]+B.x[p])/2.0;  //exact values depends on u1, u2
                     GLfloat mid_pt_y = (B.y[o]+B.y[p])/2.0;  //exact values depends on u1, u2
                     printf("duplicated!$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n");
                     printf("p: %f; o: %f\n",p,o);
                     B.x[p]=mid_pt_x-B.ballRadius[p]*dist_cos;  //position change after overlap
                     B.y[p]=mid_pt_y-B.ballRadius[p]*dist_sin;
                     B.x[o]=2*mid_pt_x-B.x[p];
                     B.y[o]=2*mid_pt_y-B.y[p];
                     GLfloat m1=pow(B.ballRadius[p],3);  //asuming constant density(p) for all balls
                     GLfloat m2=pow(B.ballRadius[p],3);
                     GLfloat speed1=sqrt(pow(B.xSpeed[p],2)+pow(B.ySpeed[p],2))+epsilon;
                     GLfloat speed2=sqrt(pow(B.xSpeed[o],2)+pow(B.ySpeed[o],2))+epsilon;
                     GLfloat speed1_cos=B.xSpeed[p]/speed1;
                     GLfloat speed1_sin=B.ySpeed[p]/speed1;
                     GLfloat speed2_cos=B.xSpeed[o]/speed2;
                     GLfloat speed2_sin=B.ySpeed[o]/speed2;
                     B.xSpeed[p]=(speed1*(speed1_cos*dist_cos+speed1_sin*dist_sin)*(m1-m2)+2*m2*speed2*(speed2_cos*dist_cos+speed2_sin*dist_sin))*dist_cos/(m1+m2)-speed1*(speed1_sin*dist_cos-speed1_cos*dist_sin)*dist_sin;
                     B.ySpeed[p]=(speed1*(speed1_cos*dist_cos+speed1_sin*dist_sin)*(m1-m2)+2*m2*speed2*(speed2_cos*dist_cos+speed2_sin*dist_sin))*dist_sin/(m1+m2)+speed1*(speed1_sin*dist_cos-speed1_cos*dist_sin)*dist_cos;
                     B.xSpeed[o]=(speed2*(speed2_cos*dist_cos+speed2_sin*dist_sin)*(m2-m1)+2*m1*speed1*(speed1_cos*dist_cos+speed1_sin*dist_sin))*dist_cos/(m2+m1)-speed2*(speed2_sin*dist_cos-speed2_cos*dist_sin)*dist_sin;
                     B.ySpeed[o]=(speed2*(speed2_cos*dist_cos+speed2_sin*dist_sin)*(m2-m1)+2*m1*speed1*(speed1_cos*dist_cos+speed1_sin*dist_sin))*dist_sin/(m2+m1)+speed2*(speed2_sin*dist_cos-speed2_cos*dist_sin)*dist_cos;
//                     B.xSpeed[p]=-B.xSpeed[p];
//                     B.ySpeed[p]=-B.ySpeed[p];
//                     B.xSpeed[o]=-B.xSpeed[o];
//                     B.ySpeed[o]=-B.ySpeed[o];
                  }
               B.x[p] += B.xSpeed[p];
               B.y[p] += B.ySpeed[p];
               if (B.x[p] > xMax)
               {
                  B.x[p] = xMax;
                  B.xSpeed[p] = -B.xSpeed[p];
               } 
               else if (B.x[p] < xMin)
               {
                  B.x[p] = xMin;
                  B.xSpeed[p] = -B.xSpeed[p];
               }
               if (B.y[p] > yMax)
               {
                  B.y[p] = yMax;
                  B.ySpeed[p] = -B.ySpeed[p];
               }
               else if (B.y[p] < yMin)
               {
                  B.y[p] = yMin;
                  B.ySpeed[p] = -B.ySpeed[p];
               }
      }
*/
}
//    printf("tx: %d, ty: %d, bw: %d, bh: %d, ballRadius: %f, x: %f, y: %f, xspeed: %f, yspeed: %f\n",tx,ty,bw,bh,B.ballRadius[tx],B.x[ty],B.y[ty],B.xSpeed[bh],B.ySpeed[bh]);
//    printf("%d,%d,%d,%d,%f,%f!\n",tx,ty,bw,bh,B.x,B.y);

void processCUDA()  
{
   std::cout<<"kx: "<<kx<<";ky: "<<ky<<"br: "<<ballRadius<<std::endl;
   widthx = kx*2*ballRadius;
   widthy = ky*2*ballRadius;
   n_x = ceil((xMax-xMin)/widthx); 
   n_y = ceil((yMax-yMin)/widthy);
   std::cout<<"widthx: "<<widthx<<" ;widthy: "<<widthy<<";x_max: "<<xMax<<"x_min: "<<xMin<<";n_x: "<<n_x<<";n_y: "<<n_y<<':'<<std::endl;
   n_threadsx = min(n_x,max_threadsx);
   n_threadsy = min(n_y,max_threadsy);
   temp_n_blocksx = int(ceil((float)n_x/(float)n_threadsx));
   n_blocksx = min(temp_n_blocksx,max_blocksx);
   halo_x=0, halo_y=0, n_halo_threadsx=0, n_halo_threadsy=0;
   if (n_blocksx==temp_n_blocksx && n_x%n_threadsx!=0)
   {
         halo_x=1;
         n_halo_threadsx=n_x%n_threadsx;
   }
   temp_n_blocksy = int(ceil((float)n_y/(float)n_threadsy)); 
   n_blocksy = min(temp_n_blocksy,max_blocksy);
   if (n_blocksy==temp_n_blocksy && n_y%n_threadsy!=0)
   {
        halo_y=1;
        n_halo_threadsy=n_y%n_threadsy;
   }
   num_cellsx = (n_blocksx-1)*n_threadsx+halo_x*n_halo_threadsx+(1-halo_x)*n_threadsx;
   num_cellsy = (n_blocksy-1)*n_threadsy+halo_y*n_halo_threadsy+(1-halo_y)*n_threadsy;
//   int worklist[num_cellsx][num_cellsy][kx*ky];
//   int num_works[num_cellsx*num_cellsy]={{0}};
   std::cout<<"halos: "<<halo_x<<" : "<<halo_y<<std::endl;
   std::cout<<"n: "<<n_threadsx<<':'<<n_threadsy<<':'<<n_blocksx<<':'<<n_blocksy<<':'<<n_halo_threadsx<<':'<<n_halo_threadsy<<':'<<num_cellsx<<':'<<num_cellsy<<std::endl;
   int i,j;
   std::cout<<"allocating mem\n";
   cudaFreeHost(worklist);
   cudaFreeHost(num_works);
   ERROR_CHECK(cudaHostAlloc((int**)&worklist,sizeof(int)*(num_cellsx*num_cellsy*kx*ky),0));
   std::cout<<"ll mem\n";
   ERROR_CHECK(cudaHostAlloc((int**)&num_works,sizeof(int)*(num_cellsx*num_cellsy),0));   
   cudaMalloc(&worklist_new,sizeof(int)*(num_cellsx*num_cellsy*kx*ky));  //try keeping this also cudahostalloc (paged mem) to check for any improvements!
   cudaMalloc(&num_works_new,sizeof(int)*(num_cellsx*num_cellsy));
//   start=0;
    std::cout<<"allocated mem\n";
   for(int k=0;k<num_cellsx*num_cellsy;k++)
      num_works[k]=0;      
   std::cout<<"allocated mem\n";
   for(int k=0;k<n;k++)  
   {
      i = min(int((B.x[k]+abs(xMin))/widthx),num_cellsx-1);
      j = min(int((B.y[k]+abs(yMin))/widthy),num_cellsy-1);
//      std::cout<<"i,j: "<<i<<':'<<j<<std::endl;
      if(num_works[i+num_cellsx*j]<kx*ky)
         worklist[kx*ky*i+num_cellsx*kx*ky*j+num_works[i+num_cellsx*j]] = k;
      num_works[i+num_cellsx*j]+=1;
   }
//   for(int c=0;c<num_cellsx;c++)
//      for(int d=0;d<num_cellsy;d++)
//         for(int k=0;k<kx*ky;k++)
//            std::cout<<"num_works: "<<num_works[c+num_cellsx*d]<<" ;c: "<<c<<" ,d: "<<d<<" ,k: "<<k<<"; wl: "<<worklist[kx*ky*c+num_cellsx*kx*ky*d+k]<<std::endl;
   int sum=0;
   for(int k=0;k<num_cellsx*num_cellsy;k++)
      sum+=num_works[k];
   std::cout<<"sum: "<<sum<<std::endl;
//   int a;
//   std::cin>>a;      
   std::cout<<"allocated mem\n";
   dim3 gridDim(n_blocksx,n_blocksy,1);
   std::cout<<"allocated mem\n";
   dim3 blockDim(n_threadsx,n_threadsy,1);
   std::cout<<"allocated mem\n"; 
   std::cout<<"pk: "<<*pers_krnl<<std::endl;  
//	unsigned int hcounter=0, *counter;	cudaMalloc(&counter, sizeof(unsigned));cudaMemcpy(counter, &hcounter, &hcounter, sizeof(unsigned),cudaMemcpyHostToDevice);
	int *counter;
	cudaMalloc(&counter, sizeof(int));
	cudaMemset(&counter, 0, sizeof(int));
	printf("india: %f,%f,%f,%f\n",xMax,xMin,yMax,yMin);
   processKernel<<<gridDim,blockDim>>>(pers_krnl,B,worklist,worklist_new,num_works,num_works_new,num_cellsx,num_cellsy,halo_x,n_blocksx,n_halo_threadsx,halo_y,n_blocksy,n_halo_threadsy,xMax,xMin,yMax,yMin,widthx,widthy,counter);
 //  cudaGetErrorString(cudaGetLastError());  //ERROR_CHECK(cudaPeekAtLastError());
//   cudaDeviceSynchronize();
   total+=1;
//   if(total==100)
//   	exit(0);
   std::cout<<"\nhippy!: "<<total<<std::endl;
}

void displayi()
{
      glClear(GL_COLOR_BUFFER_BIT);  //clear the color buffer
      for(int i=0; i<n; i++)  
      {
         if(i==0)
            energy=0;
         energy+=pow(B.xSpeed[i],2)+pow(B.ySpeed[i],2);
   //      if(i==n-1)  std::cout<<"total system energy: "<<energy<<std::endl;
         glMatrixMode(GL_MODELVIEW);  //to operate on the model-view matrix
         glLoadIdentity();  //reset model-view matrix
         printf("x,y  :: %f:%f\n",B.x[i],B.y[i]);
         draw_circle(B.x[i],B.y[i],B.ballRadius[i]);
      }
      glutSwapBuffers();  //swap front and back buffers (of double buffered mode);  
      processCUDA();
}


void display()  //callback handler for window re-paint event */
{
   std::cout<<"hike1!: "<<*pers_krnl<<std::endl;
   if(*pers_krnl==1)  //no need for 'atomicInc_system' due to single-thread CPU
   {
      glClear(GL_COLOR_BUFFER_BIT);  //clear the color buffer
      for(int i=0; i<n; i++)  
      {
         if(i==0)
            energy=0;
         energy+=pow(B.xSpeed[i],2)+pow(B.ySpeed[i],2);
   //      if(i==n-1)  std::cout<<"total system energy: "<<energy<<std::endl;
         glMatrixMode(GL_MODELVIEW);  //to operate on the model-view matrix
         glLoadIdentity();  //reset model-view matrix
         draw_circle(B.x[i],B.y[i],B.ballRadius[i]);
      }
      glutSwapBuffers();  //swap front and back buffers (of double buffered mode);  
      *pers_krnl=2;
//      refreshMillis+=1;
   }
   else if(*pers_krnl==0)
   {
      *pers_krnl=2;
      processCUDA();
//      *pers_krnl=1;
   }
   std::cout<<"hike2!: "<<*pers_krnl<<std::endl;
}

void reshape(GLsizei width, GLsizei height)  //call back when the windows is re-sized
{
//   std::cout<<"hey!: "<<*pers_krnl<<std::endl;
   if (height == 0) height = 1;  //to prevent divide by 0;  //compute aspect ratio of the new window
   GLfloat aspect = (GLfloat)width / (GLfloat)height;
   glViewport(0, 0, width, height);  //set the viewport to cover the new window
   glMatrixMode(GL_PROJECTION);  //to operate on the projection matrix;  //set the aspect ratio of the clipping area to match the viewport
   glLoadIdentity();  //reset the projection matrix
   if (width >= height)
   {
      clipAreaXLeft   = -1.0 * aspect;
      clipAreaXRight  = 1.0 * aspect;
      clipAreaYBottom = -1.0;
      clipAreaYTop    = 1.0;
   }
   else 
   {
      clipAreaXLeft   = -1.0;
      clipAreaXRight  = 1.0;
      clipAreaYBottom = -1.0 / aspect;
      clipAreaYTop    = 1.0 / aspect;
   }
   gluOrtho2D(clipAreaXLeft, clipAreaXRight, clipAreaYBottom, clipAreaYTop);
   if(*pers_krnl==1 && (xMin!=clipAreaXLeft+ballRadius) || (xMax!=clipAreaXRight-ballRadius) || (yMin!=clipAreaYBottom+ballRadius) || (yMax!=clipAreaYTop-ballRadius))
      *pers_krnl=0;
   xMin = clipAreaXLeft + ballRadius;
   xMax = clipAreaXRight - ballRadius;
   yMin = clipAreaYBottom + ballRadius;
   yMax = clipAreaYTop - ballRadius;
}
 
void timer(int value)  //called back when the timer expired
{
   glutPostRedisplay();  //post a paint request to activate display()
   glutTimerFunc(refreshMillis, timer, 0);  //subsequent timer call at milliseconds
}


int main(int argc, char** argv)
{
   page_mem_alloc();  //cudaGLSetGLDevice(0);  //init cuda--gave 'gpu resource already being used' error
   initBalls();
   glutInit(&argc, argv);  //initialize GLUT
   glutInitDisplayMode(GLUT_DOUBLE);  //enable double buffered mode
   glutInitWindowSize(windowWidth, windowHeight);  //initial window width and height
   glutInitWindowPosition(windowPosX, windowPosY);  //initial window top-left corner (x, y)
   glutCreateWindow(title);  //window title
   glutDisplayFunc(display);  //register callback handler for window re-paint
   glutReshapeFunc(reshape);  //register callback handler for window re-shape
   glutTimerFunc(0, timer, 0);  //first timer call immediately
   glClearColor(0.0, 0.0, 0.0, 1.0);  //set background (clear) color to black
   glutMainLoop();  //enter event-processing loop
   return 0;
}
