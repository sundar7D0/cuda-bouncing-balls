//To compile: nvcc gbb.cu -o gbb.out -I /usr/local/cuda-10.1/targets/x86_64-linux/include/ -lcudart -lglut -lGLU -lGL -lGLEW
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
int n_x, n_y, n_threadsx, n_threadsy;  //generalize!
int n_blocksx, n_blocksy;
int temp_n_blocksx, temp_n_blocksy;
int halo_x, n_halo_threadsx;
int halo_y, n_halo_threadsy;
int num_cellsx, num_cellsy;
int* worklist;
int* num_works;
int *counter;
int *die=0;
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
struct balls B_new;

#define ERROR_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__);}  //CUDA ERROR_CHECK inline-function
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
	  fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	  if (abort) exit(code);
   }
}

void mem_alloc()  //page and device memory allocation
{
   ERROR_CHECK(cudaHostAlloc((GLfloat**)&B.ballRadius,sizeof(GLfloat)*n,0));
   ERROR_CHECK(cudaHostAlloc((GLfloat**)&B.x,sizeof(GLfloat)*n,0));
   ERROR_CHECK(cudaHostAlloc((GLfloat**)&B.y,sizeof(GLfloat)*n,0));
   ERROR_CHECK(cudaHostAlloc((GLfloat**)&B.xSpeed,sizeof(GLfloat)*n,0));
   ERROR_CHECK(cudaHostAlloc((GLfloat**)&B.ySpeed,sizeof(GLfloat)*n,0));
   ERROR_CHECK(cudaMalloc((GLfloat**)&B_new.ballRadius,sizeof(GLfloat)*n));
   ERROR_CHECK(cudaMalloc((GLfloat**)&B_new.x,sizeof(GLfloat)*n));
   ERROR_CHECK(cudaMalloc((GLfloat**)&B_new.y,sizeof(GLfloat)*n));
   ERROR_CHECK(cudaMalloc((GLfloat**)&B_new.xSpeed,sizeof(GLfloat)*n));
   ERROR_CHECK(cudaMalloc((GLfloat**)&B_new.ySpeed,sizeof(GLfloat)*n));
   ERROR_CHECK(cudaHostAlloc((int**)&die, sizeof(int), 0));  //ERROR_CHECK(cudaHostAlloc((int**)&pers_krnl, sizeof(int), 0));++*pers_krnl;
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

__global__ void processKernel(int* die,struct balls B,struct balls B_new,int* worklist,int* worklist_new,int* num_works,int* num_works_new,int num_cellsx,int num_cellsy,int halo_x,int n_blocksx,int n_halo_threadsx,int halo_y,int n_blocksy,int n_halo_threadsy,float xMax,float xMin,float yMax,float yMin,GLfloat widthx,GLfloat widthy,int* counter)  //CUDA kernel for intensive computations
{
   __shared__ bool x_, x__, y_, y__;  //shared only among a block
   x_=(halo_x==0 || (halo_x==1 && blockIdx.x<n_blocksx-1));x__=(halo_x==1 && blockIdx.x==n_blocksx-1);
   y_=(halo_y==0 || (halo_y==1 && blockIdx.y<n_blocksy-1));y__=(halo_y==1 && blockIdx.y==n_blocksy-1);
   __shared__ int max_counter;
   max_counter=gridDim.x*gridDim.y-1;
   unsigned int i = threadIdx.x+blockIdx.x*blockDim.x;  //private
   unsigned int j = threadIdx.y+blockIdx.y*blockDim.y;
   unsigned int temp, i_new, j_new, o, p, current;
   GLfloat dist, dist_sin, dist_cos, mid_pt_x, mid_pt_y, m1, m2, speed1, speed1_cos, speed1_sin, speed2, speed2_cos, speed2_sin;
   if((x_ || x__  && threadIdx.x<n_halo_threadsx) && (y_ || y__ && threadIdx.y<n_halo_threadsy))
	  {
//          while(*pers_krnl)  {
			if(threadIdx.x==0 && blockIdx.x==0 && threadIdx.y==0 && blockIdx.y==0)
			{
				printf("inside gpu: %d\n",*die);
            	printf("b.y1: %f,%f,%f,%f,%f,%d,%d,%d,%d,%d,%d,%d,%d\n",B.y[0],yMax,yMin,xMax,xMin,num_cellsx,num_cellsy,halo_x,n_blocksx,n_halo_threadsx,halo_y,n_blocksy,n_halo_threadsy);
			}
//            while(*pers_krnl!=2);
			printf("weeklyy!\n");
			printf("B_new: %f,%f\n",B_new.x[0],B.x[0]);
	  for (int m=max(0,i-1);m<=min(i+1,num_cellsx-1);m++)
		 for (int n=max(0,j-1);n<=min(j+1,num_cellsy-1);n++)
			for (int p_=0;p_<min(num_works[i+j*num_cellsx],kx*ky);p_++)
			{
				  p=worklist[kx*ky*i+num_cellsx*kx*ky*j+p_];
				  printf("digits: %d,%d,%d,%d,%d,%d\n",i,j,m,n,p,min(num_works[i+j*num_cellsx],kx*ky));
				  B_new.x[p]=B.x[p];
				  B_new.y[p]=B.y[p];
				  B_new.xSpeed[p]=B.xSpeed[p];
				  B_new.ySpeed[p]=B.ySpeed[p];
			   for (int o_=0; o_<min(num_works[m+n*num_cellsx],kx*ky);o_++)  //try interchanging this and above 2 loops  
			   {
				  o=worklist[kx*ky*m+num_cellsx*kx*ky*n+o_];
				  dist=sqrt(pow(B.x[o]-B.x[p],2)+pow(B.y[o]-B.y[p],2));  //identifying overlaps with neighbours using distance between centers
				  printf("\ndist: %d,%d,%d,%d,%d,%d,%f,%f\n",i,j,m,n,p,o,dist,B.ballRadius[o]+B.ballRadius[p]);
				  if (dist<B.ballRadius[o]+B.ballRadius[p] && o!=p)
				  {
					 dist_sin = (B.y[o]-B.y[p])/(dist+epsilon);  //p=i,o=j
					 dist_cos = (B.x[o]-B.x[p])/(dist+epsilon);
					 mid_pt_x = (B.x[o]+B.x[p])/2.0;  //exact values depends on u1, u2
					 mid_pt_y = (B.y[o]+B.y[p])/2.0;  //exact values depends on u1, u2
					 printf("p: %d; o: %d\n",p,o);
					 B_new.x[p]=mid_pt_x-B.ballRadius[p]*dist_cos;  //position change after overlap
					 B_new.y[p]=mid_pt_y-B.ballRadius[p]*dist_sin;
//                     B.new.x[o]=2*mid_pt_x-B.x[p];
//                     B._newy[o]=2*mid_pt_y-B.y[p];
					 m1=pow(B.ballRadius[p],3);  //asuming constant density(p) for all balls
					 m2=pow(B.ballRadius[p],3);
					 speed1=sqrt(pow(B.xSpeed[p],2)+pow(B.ySpeed[p],2))+epsilon;
					 speed1_cos=B.xSpeed[p]/speed1;
					 speed1_sin=B.ySpeed[p]/speed1;
					 speed2=sqrt(pow(B.xSpeed[o],2)+pow(B.ySpeed[o],2))+epsilon;
					 speed2_cos=B.xSpeed[o]/speed2;
					 speed2_sin=B.ySpeed[o]/speed2;
					 B_new.xSpeed[p]=(speed1*(speed1_cos*dist_cos+speed1_sin*dist_sin)*(m1-m2)+2*m2*speed2*(speed2_cos*dist_cos+speed2_sin*dist_sin))*dist_cos/(m1+m2)-speed1*(speed1_sin*dist_cos-speed1_cos*dist_sin)*dist_sin;  //B.xSpeed[p]=-B.xSpeed[p];
					 B_new.ySpeed[p]=(speed1*(speed1_cos*dist_cos+speed1_sin*dist_sin)*(m1-m2)+2*m2*speed2*(speed2_cos*dist_cos+speed2_sin*dist_sin))*dist_sin/(m1+m2)+speed1*(speed1_sin*dist_cos-speed1_cos*dist_sin)*dist_cos;
//                     B_new.xSpeed[o]=(speed2*(speed2_cos*dist_cos+speed2_sin*dist_sin)*(m2-m1)+2*m1*speed1*(speed1_cos*dist_cos+speed1_sin*dist_sin))*dist_cos/(m2+m1)-speed2*(speed2_sin*dist_cos-speed2_cos*dist_sin)*dist_sin;
//                     B_new.ySpeed[o]=(speed2*(speed2_cos*dist_cos+speed2_sin*dist_sin)*(m2-m1)+2*m1*speed1*(speed1_cos*dist_cos+speed1_sin*dist_sin))*dist_sin/(m2+m1)+speed2*(speed2_sin*dist_cos-speed2_cos*dist_sin)*dist_cos;
				  }
				}
			}
			for(int p_=0;p_<min(num_works[i+num_cellsx*j],kx*ky);p_++)
			{
			   p=worklist[kx*ky*i+num_cellsx*kx*ky*j+p_];
			   B_new.x[p] += B_new.xSpeed[p];
			   B_new.y[p] += B_new.ySpeed[p];
			   if (B_new.y[p] > yMax)
			   {
				  B_new.y[p] = yMax;
				  B_new.ySpeed[p] = -B_new.ySpeed[p];
			   }
			   else if (B_new.y[p] < yMin)
			   {
				  B_new.y[p] = yMin;
				  B_new.ySpeed[p] = -B_new.ySpeed[p];
			   }
			   if (B_new.x[p] > xMax)
			   {
				  B_new.x[p] = xMax;
				  B_new.xSpeed[p] = -B_new.xSpeed[p];
			   } 
			   else if (B_new.x[p] < xMin)
			   {
				  B_new.x[p] = xMin;
				  B_new.xSpeed[p] = -B_new.xSpeed[p];
			   }
			   i_new=min(int((B_new.x[p]+abs(xMin))/widthx),num_cellsx-1);
			   j_new=min(int((B_new.y[p]+abs(yMin))/widthy),num_cellsy-1);
			   temp=atomicAdd(&num_works_new[i_new+num_cellsx*j_new],1);  //num_works_new[i_new+num_cellsx*j_new]+=1;  //done atomically!
			   printf("debug: %d,%d,%d,%d,%d\n",i_new,j_new,temp,num_works_new[i_new+num_cellsx*j_new],p);
			   if(temp<kx*ky)
				  worklist_new[kx*ky*i_new+num_cellsx*kx*ky*j_new+temp]=p;
			}
			if(threadIdx.x==0 && threadIdx.y==0)
			{
				printf("b.y2: %f\n",B.y[0]);
				printf("max_counter1: %d:%d\n",max_counter,*counter);
				atomicInc((unsigned int*)counter,(unsigned int)max_counter);//atomicInc(&counter,max_counter);
				printf("Biggggggggggg!\n");
				printf("max_counter2: %d:%d\n",max_counter,*counter);
			}
			__syncthreads();
			printf("1.i:%d,j:%d,counter:%d\n",i,j,*counter);
			printf("i: %d,j:%d,k:%d,wl0:%d,wl1:%d,wl2:%d,wl3:%d\n",i,j,num_works_new[i+num_cellsx*j],worklist[kx*ky*i+num_cellsx*kx*ky*j+0],worklist[kx*ky*i+num_cellsx*kx*ky*j+1],worklist[kx*ky*i+num_cellsx*kx*ky*j+2],worklist[kx*ky*i+num_cellsx*kx*ky*j+3]);
//            __threadfence();  //to update 'worklist', 'num_works' among all threads
			while(*counter);  //global_barrier
			num_works[i+num_cellsx*j]=num_works_new[i+num_cellsx*j];
			num_works_new[i+num_cellsx*j]=0;
			printf("imp: %d,%d,%d\n",i,j,num_works[i+num_cellsx*j]);
			for(int k=0;k<min(num_works[i+num_cellsx*j],kx*ky);k++)  //kx*ky
			{  
			 worklist[kx*ky*i+num_cellsx*kx*ky*j+k]=worklist_new[kx*ky*i+num_cellsx*kx*ky*j+k];
			 current=worklist[kx*ky*i+num_cellsx*kx*ky*j+k];
//			 B.ballRadius[current]=B_new.ballRadius[current];
			 printf("current: %d,%d,%d,%f,%f,%f,%f\n",i,j,current,B_new.x[current],B_new.y[current],B_new.xSpeed[current],B_new.ySpeed[current]);
			 B.x[current]=B_new.x[current];
			 B.y[current]=B_new.y[current];
 			 B.xSpeed[current]=B_new.xSpeed[current];
 			 B.ySpeed[current]=B_new.ySpeed[current];
			}	
			//}
	  }
}

void display()
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
	dim3 gridDim(n_blocksx,n_blocksy,1);
	dim3 blockDim(n_threadsx,n_threadsy,1);
	cudaMalloc(&counter, sizeof(int));
	cudaMemset(&counter, 0, sizeof(int));
	printf("heiko: %d,%d\n",num_cellsx,num_cellsy);
	processKernel<<<gridDim,blockDim>>>(die,B,B_new,worklist,worklist_new,num_works,num_works_new,num_cellsx,num_cellsy,halo_x,n_blocksx,n_halo_threadsx,halo_y,n_blocksy,n_halo_threadsy,xMax,xMin,yMax,yMin,widthx,widthy,counter);
  cudaGetErrorString(cudaGetLastError());  //ERROR_CHECK(cudaPeekAtLastError());  
	cudaDeviceSynchronize();
	total+=1;  
//	if(total==3)  exit(0);
	std::cout<<"\ntotal kernel calls: "<<total<<std::endl;
}

void reshape(GLsizei width, GLsizei height)  //call back when the windows is re-sized
{
   if (height == 0) height = 1;  //to prevent division by 0;  //compute aspect ratio of the new window
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
   xMin = clipAreaXLeft + ballRadius;
   xMax = clipAreaXRight - ballRadius;
   yMin = clipAreaYBottom + ballRadius;
   yMax = clipAreaYTop - ballRadius;
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
   std::cout<<"n: "<<n_threadsx<<':'<<n_threadsy<<':'<<n_blocksx<<':'<<n_blocksy<<':'<<n_halo_threadsx<<':'<<n_halo_threadsy<<':'<<num_cellsx<<':'<<num_cellsy<<std::endl;
   cudaMalloc(&worklist_new,sizeof(int)*(num_cellsx*num_cellsy*kx*ky));
   cudaMalloc(&num_works_new,sizeof(int)*(num_cellsx*num_cellsy));
   cudaFreeHost(worklist);
   cudaFreeHost(num_works);
   ERROR_CHECK(cudaHostAlloc((int**)&worklist,sizeof(int)*(num_cellsx*num_cellsy*kx*ky),0));
   ERROR_CHECK(cudaHostAlloc((int**)&num_works,sizeof(int)*(num_cellsx*num_cellsy),0));   
   for(int k=0;k<num_cellsx*num_cellsy;k++)
	  num_works[k]=0;      
   int i,j;
   for(int k=0;k<n;k++)  
   {
	  i = min(int((B.x[k]+abs(xMin))/widthx),num_cellsx-1);
	  j = min(int((B.y[k]+abs(yMin))/widthy),num_cellsy-1);
	  if(num_works[i+num_cellsx*j]<kx*ky)
		 worklist[kx*ky*i+num_cellsx*kx*ky*j+num_works[i+num_cellsx*j]] = k;
	  num_works[i+num_cellsx*j]+=1;
   }
   for(int c=0;c<num_cellsx;c++)for(int d=0;d<num_cellsy;d++)for(int k=0;k<kx*ky;k++)std::cout<<"num_works: "<<num_works[c+num_cellsx*d]<<" ;c: "<<c<<" ,d: "<<d<<" ,k: "<<k<<"; wl: "<<worklist[kx*ky*c+num_cellsx*kx*ky*d+k]<<std::endl;
}

void timer(int value)  //called back when the timer expired
{
	glutPostRedisplay();  //post a paint request to activate display()
	glutTimerFunc(refreshMillis, timer, 0);  //subsequent timer call at milliseconds
}

int main(int argc, char** argv)
{
	mem_alloc();  //cudaGLSetGLDevice(0);  //to initalize cuda--gave 'gpu resource already being used error
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
