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
#define BLOCK_SIZE 16  // block size
#define PI 3.14159265f
#define SRC_BUFFER  0
#define DST_BUFFER  1
#define eps 1e-30
#define kx 2
#define ky 2
#define repeat 1
#define tiny_dist 0.008
#define epsilon 1
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
int move_w=0,move_s=0,move_a=0,move_d=0;
GLdouble clipAreaXLeft, clipAreaXRight, clipAreaYBottom, clipAreaYTop;  //projection clipping area
struct balls  {
   GLfloat* ballRadius;  // = (GLfloat*)malloc(n*sizeof(GLfloat*));
   GLfloat* x;  // = (GLfloat*)malloc(n*sizeof(GLfloat*));
   GLfloat* y;  // = (GLfloat*)malloc(n*sizeof(GLfloat*));
   GLfloat* xSpeed;  // = (GLfloat*)malloc(n*sizeof(GLfloat*));
   GLfloat* ySpeed;  // = (GLfloat*)malloc(n*sizeof(GLfloat*));
}B;
#define size_n 30  // kx=2, ky=2;  //hyper-parameters
int start=1;
int renew=0, renew_max=100;
int* worklist_new;
int* num_works_new;
struct balls B_new;
bool mouseleftdown = false;   // True if mouse LEFT button is down.

int mousex, mousey;           // Mouse x,y coords, in GLUT format (pixels from upper-left corner).
                              // Only guaranteed to be valid if a mouse button is down.
                              // Saved by mouse, motion.
// Keyboard
const int ESCKEY = 27;        // ASCII value of escape character


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
   ERROR_CHECK(cudaHostAlloc((GLfloat**)&B.ballRadius,sizeof(GLfloat)*size_n,0));
   ERROR_CHECK(cudaHostAlloc((GLfloat**)&B.x,sizeof(GLfloat)*size_n,0));
   ERROR_CHECK(cudaHostAlloc((GLfloat**)&B.y,sizeof(GLfloat)*size_n,0));
   ERROR_CHECK(cudaHostAlloc((GLfloat**)&B.xSpeed,sizeof(GLfloat)*size_n,0));
   ERROR_CHECK(cudaHostAlloc((GLfloat**)&B.ySpeed,sizeof(GLfloat)*size_n,0));
   ERROR_CHECK(cudaMalloc((GLfloat**)&B_new.ballRadius,sizeof(GLfloat)*size_n));
   ERROR_CHECK(cudaMalloc((GLfloat**)&B_new.x,sizeof(GLfloat)*size_n));
   ERROR_CHECK(cudaMalloc((GLfloat**)&B_new.y,sizeof(GLfloat)*size_n));
   ERROR_CHECK(cudaMalloc((GLfloat**)&B_new.xSpeed,sizeof(GLfloat)*size_n));
   ERROR_CHECK(cudaMalloc((GLfloat**)&B_new.ySpeed,sizeof(GLfloat)*size_n));
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
   	srand(2);
   for(int i=0;i<size_n;i++)
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
   cudaMemcpy(B_new.ballRadius,B.ballRadius,sizeof(GLfloat)*size_n,cudaMemcpyHostToDevice);
   cudaMemcpy(B_new.x,B.x,sizeof(GLfloat)*size_n,cudaMemcpyHostToDevice);
   cudaMemcpy(B_new.y,B.y,sizeof(GLfloat)*size_n,cudaMemcpyHostToDevice);
   cudaMemcpy(B_new.xSpeed,B.xSpeed,sizeof(GLfloat)*size_n,cudaMemcpyHostToDevice);
   cudaMemcpy(B_new.ySpeed,B.ySpeed,sizeof(GLfloat)*size_n,cudaMemcpyHostToDevice);
}

void draw_circle(GLfloat x, GLfloat y, GLfloat r, int special=0)  //draw a circle (ball) using triangular fan
{
   int numSegments = 10000;
   GLfloat angle;
   glTranslatef(x,y,0.0f);  //translate to (xPos, yPos)
   glBegin(GL_TRIANGLE_FAN);  //triangle-fan to create circle
   if (special)
   	glColor3f(0.0f, 0.0f, 1.0f);  //blue-color   	
   else
   	glColor3f(1.0f, 0.0f, 0.0f);  //red-color
   glVertex2f(0.0f, 0.0f);  //center of circle
   for (int j = 0; j <= numSegments; j++)  //= as last vertex same as first vertex
   {
	  angle = j * 2.0f * PI / numSegments;  //360` for all segments
	  glVertex2f(cos(angle)*r,sin(angle)*r);
   }
   glEnd();
}

__device__ void naive_collider(struct balls B,struct balls B_new,int p, int o,GLfloat dist)  //'p' collides with 'o'
{
	GLfloat dist_[2], mid_pt[2], mass[2], speed_p, speed_angle_p[2], speed_o, speed_angle_o[2], dx, dy, lineAngle, alpha;
	printf("Balls %d and %d have collided!\n",p,o);
	dist_[1] = (B_new.y[o]-B_new.y[p])/(dist+eps);  //p=i,o=j
	dist_[0] = (B_new.x[o]-B_new.x[p])/(dist+eps);
	mid_pt[0] = (B_new.x[o]+B_new.x[p])/2.0;  //exact values depends on u1, u2
	mid_pt[1] = (B_new.y[o]+B_new.y[p])/2.0;  //exact values depends on u1, u2
	B.x[p] = mid_pt[0]-B_new.ballRadius[p]*dist_[0];  //position change after overlap
	B.y[p] = mid_pt[1]-B_new.ballRadius[p]*dist_[1];
	mass[0] = pow(B_new.ballRadius[p],3);  //asuming constant density(p) for all balls
	mass[1] = pow(B_new.ballRadius[o],3);
	speed_p = sqrt(pow(B_new.xSpeed[p],2)+pow(B_new.ySpeed[p],2));
	speed_angle_p[0] = B_new.xSpeed[p]/(speed_p+eps);
	speed_angle_p[1] = B_new.ySpeed[p]/(speed_p+eps);
	speed_o = sqrt(pow(B_new.xSpeed[o],2)+pow(B_new.ySpeed[o],2));
	speed_angle_o[0] = B_new.xSpeed[o]/(speed_o+eps);
	speed_angle_o[1] = B_new.ySpeed[o]/(speed_o+eps);
	dx = B_new.x[p]-B_new.x[o];
	dy = B_new.y[p]-B_new.y[o];
	alpha=lineAngle = atan2(-dx, dy);
	alpha = fmod(lineAngle,2*PI);
	B.xSpeed[p] = (speed_p*(speed_angle_p[0]*cos(alpha)+speed_angle_p[1]*sin(alpha))*(mass[0]-mass[1])+2*mass[1]*speed_o*(speed_angle_o[0]*cos(alpha)+speed_angle_o[1]*sin(alpha)))*cos(alpha)/(mass[0]+mass[1])-speed_p*(speed_angle_p[1]*cos(alpha)-speed_angle_p[0]*sin(alpha))*sin(alpha);  //B.xSpeed[p]=-B_new.xSpeed[p];	
	B.ySpeed[p] = (speed_p*(speed_angle_p[0]*cos(alpha)+speed_angle_p[1]*sin(alpha))*(mass[0]-mass[1])+2*mass[1]*speed_o*(speed_angle_o[0]*cos(alpha)+speed_angle_o[1]*sin(alpha)))*sin(alpha)/(mass[0]+mass[1])+speed_p*(speed_angle_p[1]*cos(alpha)-speed_angle_p[0]*sin(alpha))*cos(alpha);
//	B.xSpeed[p] = (speed_p*(speed_angle_p[0]*abs(dist_[0])+speed_angle_p[1]*dist_[1])*(mass[0]-mass[1])+2*mass[1]*speed_o*(speed_angle_o[0]*abs(dist_[0])+speed_angle_o[1]*dist_[1]))*abs(dist_[0])/(mass[0]+mass[1])-speed_p*(speed_angle_p[1]*abs(dist_[0])-speed_angle_p[0]*dist_[1])*dist_[1];  //B.xSpeed[p]=-B_new.xSpeed[p];
//	B.ySpeed[p] = (speed_p*(speed_angle_p[0]*abs(dist_[0])+speed_angle_p[1]*dist_[1])*(mass[0]-mass[1])+2*mass[1]*speed_o*(speed_angle_o[0]*abs(dist_[0])+speed_angle_o[1]*dist_[1]))*dist_[1]/(mass[0]+mass[1])+speed_p*(speed_angle_p[1]*abs(dist_[0])-speed_angle_p[0]*dist_[1])*abs(dist_[0]);
//	B.xSpeed[p] = (speed_p*(speed_angle_p[0]*dist_[0]+speed_angle_p[1]*dist_[1])*(mass[0]-mass[1])+2*mass[1]*speed_o*(speed_angle_o[0]*dist_[0]+speed_angle_o[1]*dist_[1]))*dist_[0]/(mass[0]+mass[1])-speed_p*(speed_angle_p[1]*dist_[0]-speed_angle_p[0]*dist_[1])*dist_[1];  //B.xSpeed[p]=-B_new.xSpeed[p];
//	B.ySpeed[p] = (speed_p*(speed_angle_p[0]*dist_[0]+speed_angle_p[1]*dist_[1])*(mass[0]-mass[1])+2*mass[1]*speed_o*(speed_angle_o[0]*dist_[0]+speed_angle_o[1]*dist_[1]))*dist_[1]/(mass[0]+mass[1])+speed_p*(speed_angle_p[1]*dist_[0]-speed_angle_p[0]*dist_[1])*dist_[0];

}

__device__ void gas_collider(struct balls B,struct balls B_new,int p, int o,GLfloat dist,float xMax,float xMin,float yMax,float yMin)  //'p'(1) collides with 'o'(2)
{
	GLfloat dx,dy,ratio,contact[2],lineAngle,temp_dist,normal[2],prev_pos_p[2],prev_dist,position_ratio,point_on_line[2],alpha,gamma,theta,rel_vel[2],vel_rel,mass[2];
	printf("Balls %d and %d have collided!\n",p,o);
	dx = B_new.x[p]-B_new.x[o];
	dy = B_new.y[p]-B_new.y[o];
	ratio = B_new.ballRadius[p]/(dist+eps);
	contact[0] = B_new.x[p]-dx*ratio;
	contact[1] = B_new.y[p]-dy*ratio;
	temp_dist = sqrt(pow(dx,2)+pow(dy,2));
	normal[0] = dx/(temp_dist+eps);
	normal[1] = dy/(temp_dist+eps);
	lineAngle = atan2(-dx, dy);
	prev_pos_p[0] = B_new.x[p]-B.xSpeed[p];
	prev_pos_p[1] = B_new.y[p]-B.ySpeed[p];
	prev_pos_p[0] = max(xMin,min(xMax,prev_pos_p[0]));
	prev_pos_p[1] = max(yMin,min(yMax,prev_pos_p[1]));
	prev_dist = sqrt(pow(prev_pos_p[0]-contact[0],2)+pow(prev_pos_p[1]-contact[1],2));
	position_ratio = B_new.ballRadius[p]/prev_dist;
	point_on_line[0] = contact[0]-(contact[0]-prev_pos_p[0])*position_ratio;
	point_on_line[1] = contact[1]-(contact[1]-prev_pos_p[1])*position_ratio;
	alpha = fmod(lineAngle,2*PI);  //%(2*PI)
	gamma = fmod(atan2(B_new.y[p]-point_on_line[1],B_new.x[p]-point_on_line[0]),2*PI);
	theta = fmod(2*alpha-gamma,2*PI);
	temp_dist = sqrt(pow(B_new.x[p]-point_on_line[0],2)+pow(B_new.y[p]-point_on_line[1],2)); 
	B.x[p] = point_on_line[0]+temp_dist*cos(theta);  //position change after overlap
	B.y[p] = point_on_line[1]+temp_dist*sin(theta);
	rel_vel[0] = B_new.xSpeed[p]-B_new.xSpeed[o];
	rel_vel[1] = B_new.ySpeed[p]-B_new.ySpeed[o];
	vel_rel = rel_vel[0]*normal[0]+rel_vel[1]*normal[1];
	mass[0] = pow(B_new.ballRadius[p],3);  //asuming constant density(p) for all balls
	mass[1] = pow(B_new.ballRadius[o],3);
	temp_dist = (-vel_rel*(1+epsilon))/(1/mass[0]+1/mass[1]);
	B.xSpeed[p] += normal[0]*temp_dist/mass[0];
	B.ySpeed[p] += normal[1]*temp_dist/mass[0];
}

__global__ void processKernel(int* die,struct balls B,struct balls B_new,int* worklist,int* worklist_new,int* num_works,int* num_works_new,int num_cellsx,int num_cellsy,int halo_x,int n_blocksx,int n_halo_threadsx,int halo_y,int n_blocksy,int n_halo_threadsy,float xMax,float xMin,float yMax,float yMin,GLfloat widthx,GLfloat widthy,int* counter,int move_w,int move_s,int move_a,int move_d)  //CUDA kernel for intensive computations
{
   __shared__ bool x_, x__, y_, y__;  //shared only among a block
   x_=(halo_x==0 || (halo_x==1 && blockIdx.x<n_blocksx-1));x__=(halo_x==1 && blockIdx.x==n_blocksx-1);
   y_=(halo_y==0 || (halo_y==1 && blockIdx.y<n_blocksy-1));y__=(halo_y==1 && blockIdx.y==n_blocksy-1);
   __shared__ int max_counter;
   max_counter=gridDim.x*gridDim.y-1;
   unsigned int i = threadIdx.x+blockIdx.x*blockDim.x;  //private
   unsigned int j = threadIdx.y+blockIdx.y*blockDim.y;
   unsigned int temp, i_new, j_new, o, p, current;
   GLfloat dist;
   if((x_ || x__  && threadIdx.x<n_halo_threadsx) && (y_ || y__ && threadIdx.y<n_halo_threadsy))
	  {
//          while(*pers_krnl)  {
			if(threadIdx.x==0 && blockIdx.x==0 && threadIdx.y==0 && blockIdx.y==0)
			{
				printf("inside gpu: %d\n",*die);
//				*die+=1;
            	printf("b.y1: %f,%f,%f,%f,%f,%d,%d,%d,%d,%d,%d,%d,%d\n",B.y[0],yMax,yMin,xMax,xMin,num_cellsx,num_cellsy,halo_x,n_blocksx,n_halo_threadsx,halo_y,n_blocksy,n_halo_threadsy);
			}
//            while(*pers_krnl!=2);
			printf("weeklyy!\n");
			printf("B_new: %f,%f\n",B_new.x[0],B.x[0]);
			num_works[i+num_cellsx*j]=num_works_new[i+num_cellsx*j];
			num_works_new[i+num_cellsx*j]=0;
			for(int k=0;k<num_works[i+num_cellsx*j];k++)  //min(num_works[i+num_cellsx*j],kx*ky)
			{  
			 worklist[size_n*i+num_cellsx*size_n*j+k]=worklist_new[size_n*i+num_cellsx*size_n*j+k];
			 current=worklist[size_n*i+num_cellsx*size_n*j+k];
//			 B.ballRadius[current]=B_new.ballRadius[current];
			 printf("current: %d,%d,%d,%f,%f,%f,%f\n",i,j,current,B_new.x[current],B_new.y[current],B_new.xSpeed[current],B_new.ySpeed[current]);
			 B.x[current]=B_new.x[current];
			 B.y[current]=B_new.y[current];
 			 B.xSpeed[current]=B_new.xSpeed[current];
 			 B.ySpeed[current]=B_new.ySpeed[current];
			}	
	  for(int rp=0;rp<repeat;rp++)
	   for (int m=max(0,i-1);m<=min(i+1,num_cellsx-1);m++)
		 for (int n=max(0,j-1);n<=min(j+1,num_cellsy-1);n++)
			for (int p_=0;p_<num_works[i+num_cellsx*j];p_++)  //min(num_works[i+num_cellsx*j],kx*ky)
			{
				  p=worklist_new[size_n*i+num_cellsx*size_n*j+p_];
				  printf("digits: %d,%d,%d,%d,%d,%d\n",i,j,m,n,p,min(num_works[i+j*num_cellsx],kx*ky));
			   for (int o_=0; o_<num_works[i+num_cellsx*j];o_++)  //try interchanging this and above 2 loops  //min(num_works[m+n*num_cellsx],kx*ky)
			   {
				  o=worklist_new[size_n*m+num_cellsx*size_n*n+o_];
				  dist=sqrt(pow(B_new.x[o]-B_new.x[p],2)+pow(B_new.y[o]-B_new.y[p],2));  //identifying overlaps with neighbours using distance between centers
				  printf("\ndist: %d,%d,%d,%d,%d,%d,%f,%f\n",i,j,m,n,p,o,dist,B_new.ballRadius[o]+B_new.ballRadius[p]);
				  if (dist<B_new.ballRadius[o]+B_new.ballRadius[p]+tiny_dist && o!=p)
				  {
				  	if(p==0 or o==0)
				  	{
				  		*die=1;
				  	}
//				  	gas_collider(B,B_new,p,o,dist,xMax,xMin,yMax,yMin);
					naive_collider(B,B_new,p,o,dist);
				  }
				}
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
//            __threadfence();  //to update 'worklist', 'num_works' among all threads
			while(*counter);  //global_barrier
			printf("1.i:%d,j:%d,counter:%d\n",i,j,*counter);
			for(int p_=0;p_<num_works[i+num_cellsx*j];p_++)  //min(num_works[i+num_cellsx*j],kx*ky)
			{
			   p=worklist[size_n*i+num_cellsx*size_n*j+p_];
			   B_new.x[p] = (p==0) ? B_new.x[p]+(move_d-move_a)*0.08 : B_new.x[p]+B.xSpeed[p];
			   B_new.y[p] = (p==0) ? B_new.y[p]+(move_w-move_s)*0.08 : B_new.y[p]+B.ySpeed[p];
			   B_new.xSpeed[p] = B.xSpeed[p];
			   B_new.ySpeed[p] = B.ySpeed[p];
			   if (B_new.y[p] > yMax)
			   {
				  B_new.y[p] = yMax;
				  B_new.ySpeed[p] = -B.ySpeed[p];
			   }
			   else if (B_new.y[p] < yMin)
			   {
				  B_new.y[p] = yMin;
				  B_new.ySpeed[p] = -B.ySpeed[p];
			   }
			   if (B_new.x[p] > xMax)
			   {
				  B_new.x[p] = xMax;
				  B_new.xSpeed[p] = -B.xSpeed[p];
			   } 
			   else if (B_new.x[p] < xMin)
			   {
				  B_new.x[p] = xMin;
				  B_new.xSpeed[p] = -B.xSpeed[p];
			   }
			   i_new=min(int((B_new.x[p]+abs(xMin))/widthx),num_cellsx-1);
			   j_new=min(int((B_new.y[p]+abs(yMin))/widthy),num_cellsy-1);
			   temp=atomicAdd(&num_works_new[i_new+num_cellsx*j_new],1);  //num_works_new[i_new+num_cellsx*j_new]+=1;  //done atomically!
			   printf("debug: %d,%d,%d,%d,%d\n",i_new,j_new,temp,num_works_new[i_new+num_cellsx*j_new],p);
//			   if(temp<n)  //kx*ky
				  worklist_new[size_n*i_new+num_cellsx*size_n*j_new+temp]=p;
			}
			printf("i: %d,j:%d,k:%d,wl0:%d,wl1:%d,wl2:%d,wl3:%d\n",i,j,num_works_new[i+num_cellsx*j],worklist[size_n*i+num_cellsx*size_n*j+0],worklist[size_n*i+num_cellsx*size_n*j+1],worklist[size_n*i+num_cellsx*size_n*j+2],worklist[size_n*i+num_cellsx*size_n*j+3]);
			printf("imp: %d,%d,%d\n",i,j,num_works[i+num_cellsx*j]);
			//}
	  }
}

void display()
{
	  glClear(GL_COLOR_BUFFER_BIT);  //clear the color buffer
	  for(int i=0; i<size_n; i++)  
	  {
		 if(i==0)
			energy=0;
		 energy+=pow(B.xSpeed[i],2)+pow(B.ySpeed[i],2);
   //      if(i==n-1)  std::cout<<"total system energy: "<<energy<<std::endl;
		 glMatrixMode(GL_MODELVIEW);  //to operate on the model-view matrix
		 glLoadIdentity();  //reset model-view matrix
		 draw_circle(B.x[i],B.y[i],B.ballRadius[i],i);
	  }
	  glutSwapBuffers();  //swap front and back buffers (of double buffered mode);  
	dim3 gridDim(n_blocksx,n_blocksy,1);
	dim3 blockDim(n_threadsx,n_threadsy,1);
	cudaMalloc(&counter, sizeof(int));
	cudaMemset(&counter, 0, sizeof(int));
	printf("heiko: %d,%d\n",num_cellsx,num_cellsy);
	GLfloat x_coord=B.x[0],y_coord=B.y[0];
	int more_x_left=0,more_x_right=0,more_y_left=0,more_y_right=0;
/*
	for(int i=1;i<size_n;i++)
	{
		if(B.x[i]>x_coord>B.x[i]-0.99)
			more_x_right+=1;
		if(B.x[i]+0.09>x_coord>B.x[i])
			more_x_left+=1;
		if(B.y[i]>y_coord>B.y[i]-0.09)
			more_y_right+=1;
		if(B.y[i]+0.09>y_coord>B.y[i])
			more_y_left+=1;
	}
	if(more_x_left>more_x_right)
		move_a=1;
	else
		move_d=1;
	if(more_x_left==0 && more_x_right==0)
		move_a=move_d=0;
	if(more_y_left>more_y_right)
		move_w=1;
	else
		move_s=1;
	if(more_y_left==0 && more_y_right==0)
		move_w=move_s=0;
	if(total%100==0)
		move_w=move_a=1;
*/
	processKernel<<<gridDim,blockDim>>>(die,B,B_new,worklist,worklist_new,num_works,num_works_new,num_cellsx,num_cellsy,halo_x,n_blocksx,n_halo_threadsx,halo_y,n_blocksy,n_halo_threadsy,xMax,xMin,yMax,yMin,widthx,widthy,counter,move_w,move_s,move_a,move_d);
	move_w=move_s=move_a=move_d=0;
	if(*die==1)
	{
		std::cout<<"done and dusted!"<<std::endl;
//		exit(0);
	}
//  cudaGetErrorString(cudaGetLastError());  //ERROR_CHECK(cudaPeekAtLastError());  
//	cudaDeviceSynchronize();
	total+=1;  
//	if(total==10)  exit(0);
	std::cout<<"\ntotal kernel calls: "<<total<<" ; die: "<<*die<<"mouse x,y: "<<mousex<<":"<<mousey<<std::endl;
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
   cudaMalloc(&worklist_new,sizeof(int)*(num_cellsx*num_cellsy*size_n));
   cudaMalloc(&num_works_new,sizeof(int)*(num_cellsx*num_cellsy));
   cudaFreeHost(worklist);
   cudaFreeHost(num_works);
   ERROR_CHECK(cudaHostAlloc((int**)&worklist,sizeof(int)*(num_cellsx*num_cellsy*size_n),0));
   ERROR_CHECK(cudaHostAlloc((int**)&num_works,sizeof(int)*(num_cellsx*num_cellsy),0));   
   for(int k=0;k<num_cellsx*num_cellsy;k++)
	  num_works[k]=0;      
   int i,j;
   for(int k=0;k<size_n;k++)  
   {
	  i = min(int((B.x[k]+abs(xMin))/widthx),num_cellsx-1);
	  j = min(int((B.y[k]+abs(yMin))/widthy),num_cellsy-1);
	  if(num_works[i+num_cellsx*j]<kx*ky)
		 worklist[size_n*i+num_cellsx*size_n*j+num_works[i+num_cellsx*j]] = k;
	  num_works[i+num_cellsx*j]+=1;
   }
   for(int c=0;c<num_cellsx;c++)for(int d=0;d<num_cellsy;d++)for(int k=0;k<size_n;k++)std::cout<<"num_works: "<<num_works[c+num_cellsx*d]<<" ;c: "<<c<<" ,d: "<<d<<" ,k: "<<k<<"; wl: "<<worklist[size_n*c+num_cellsx*size_n*d+k]<<std::endl;
   cudaMemcpy(worklist_new,worklist,sizeof(int)*(num_cellsx*num_cellsy*size_n),cudaMemcpyHostToDevice);
   cudaMemcpy(num_works_new,num_works,sizeof(int)*(num_cellsx*num_cellsy),cudaMemcpyHostToDevice);
}

void timer(int value)  //called back when the timer expired
{
	glutPostRedisplay();  //post a paint request to activate display()
	glutTimerFunc(refreshMillis, timer, 0);  //subsequent timer call at milliseconds
}

void mouse(int button, int state, int x, int y)
{
   // Save the left button state
   if (button == GLUT_LEFT_BUTTON)
   {
      mouseleftdown = (state == GLUT_DOWN);
      glutPostRedisplay();  // Left button has changed; redisplay!
   }

   // Save the mouse position
   mousex = x;
   mousey = y;
}

void keyboard(unsigned char key, int x, int y)
{
	switch (key) {
        case 033:  //octal equivalent of the Escape key
            exit(0);
            break;
        case 'w':
        	move_w=1;
            break;
        case 's':
        	move_s=1;
            break;
        case 'a':
        	move_a=1;
            break;
        case 'd':
        	move_d=1;
            break;
    }
/*
   switch (key)
   {
   case ESCKEY:  // ESC: Quit
      exit(0);
      break;
   }
*/
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
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutReshapeFunc(reshape);  //register callback handler for window re-shape
	glutTimerFunc(0, timer, 0);  //first timer call immediately
	glClearColor(0.0, 0.0, 0.0, 1.0);  //set background (clear) color to black
	glutMainLoop();  //enter event-processing loop
	return 0;
}
