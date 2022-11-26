//nvcc bb.cpp bt.cu -o bbb.out -I /usr/local/cuda-10.1/targets/x86_64-linux/include/ -lcudart -lglut -lGLU -lGL -lGLEW
#include <math.h>
#include <iostream>
#include "cutil.h"
#include "cutil_inline_runtime.h"
#include <iostream>
#define GLEW_STATIC  //specify GLEW_STATIC to use the static linked library (.lib) instead of the dynamic linked library (.dll) for GLEW
#include <GL/glew.h>
#include <GL/glut.h>  // GLUT, includes glu.h and gl.h
#include <cuda_runtime_api.h>  //CUDA headers
#include <cuda_gl_interop.h>

#define USE_SHARED_MEM 0
#define BLOCK_SIZE 16     // block size
#define PI 3.14159265f
#define SRC_BUFFER  0
#define DST_BUFFER  1

char title[] = "2D Bouncing Balls";  //windowed mode's title
int windowWidth  = 640;  //windowed mode's width
int windowHeight = 480;  //windowed mode's height
int windowPosX   = 50;  //windowed mode's top-left corner x
int windowPosY   = 50;  //windowed mode's top-left corner y
GLfloat ballRadius = 0.0005f;  //ball radius
GLfloat xMax, xMin, yMax, yMin;  //ball's center (x, y) bounds
int refreshMillis = 30;  //refresh period in milliseconds
GLfloat epsilon=0.0;
GLfloat energy=0.0;
GLdouble clipAreaXLeft, clipAreaXRight, clipAreaYBottom, clipAreaYTop;  //projection clipping area
struct balls  {
   GLfloat* ballRadius;  // = (GLfloat*)malloc(n*sizeof(GLfloat*));
   GLfloat* x;  // = (GLfloat*)malloc(n*sizeof(GLfloat*));
   GLfloat* y;  // = (GLfloat*)malloc(n*sizeof(GLfloat*));
   GLfloat* xSpeed;  // = (GLfloat*)malloc(n*sizeof(GLfloat*));
   GLfloat* ySpeed;  // = (GLfloat*)malloc(n*sizeof(GLfloat*));
}B;
int n=100;
int start=1;

#define ERROR_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void page_mem_alloc()
{
   ERROR_CHECK(cudaHostAlloc((GLfloat**)&B.ballRadius,sizeof(GLfloat)*n,0));
   ERROR_CHECK(cudaHostAlloc((GLfloat**)&B.x,sizeof(GLfloat)*n,0));
   ERROR_CHECK(cudaHostAlloc((GLfloat**)&B.y,sizeof(GLfloat)*n,0));
   ERROR_CHECK(cudaHostAlloc((GLfloat**)&B.xSpeed,sizeof(GLfloat)*n,0));
   ERROR_CHECK(cudaHostAlloc((GLfloat**)&B.ySpeed,sizeof(GLfloat)*n,0));
}

void initBalls()  {
   for(int i=0;i<n;i++)  {
      B.ballRadius[i] = ballRadius;
      B.x[i] = 1.0*((static_cast <float> (rand()) - (static_cast <float> (RAND_MAX)/2.0)) / static_cast <float> (RAND_MAX));  //x
      B.y[i] = 1.0*((static_cast <float> (rand()) - (static_cast <float> (RAND_MAX)/2.0)) / static_cast <float> (RAND_MAX));  //y
      B.xSpeed[i] = 0.03*((static_cast <float> (rand()) - (static_cast <float> (RAND_MAX)/2.0)) / static_cast <float> (RAND_MAX));  //vx
      B.ySpeed[i] = 0.03*((static_cast <float> (rand()) - (static_cast <float> (RAND_MAX)/2.0)) / static_cast <float> (RAND_MAX));  //vy
   }
}

void draw_circle(GLfloat x, GLfloat y, GLfloat r)  {
   int numSegments = 10000;
   GLfloat angle;
   glTranslatef(x,y,0.0f);  //translate to (xPos, yPos)
   glBegin(GL_TRIANGLE_FAN);  //triangle-fan to create circle
   glColor3f(0.0f, 0.0f, 1.0f);  //color
   glVertex2f(0.0f, 0.0f);  //center of circle
   for (int j = 0; j <= numSegments; j++)  {  //last vertex same as first vertex
      angle = j * 2.0f * PI / numSegments;  //360` for all segments
      glVertex2f(cos(angle)*r,sin(angle)*r);
   }
   glEnd();
}

__global__ void PostprocessKernel(struct balls B)
{
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int bw = blockDim.x;
    unsigned int bh = blockDim.y;
//    B.x[0]=0.0;
//    B.y[0]=0.0;
    printf("tx: %d, ty: %d, bw: %d, bh: %d, ballRadius: %f, x: %f, y: %f, xspeed: %f, yspeed: %f\n",tx,ty,bw,bh,B.ballRadius[tx],B.x[ty],B.y[ty],B.xSpeed[bh],B.ySpeed[bh]);
//    printf("%d,%d,%d,%d,%f,%f!\n",tx,ty,bw,bh,B.x,B.y);
}

void PostprocessCUDA()  {
    std::cout<<"hel: "<<(xMax-xMin)/(4*ballRadius)<<':'<<(yMax-yMin)/(4*ballRadius)<<std::endl;
    dim3 gridDim( 3,10,1);
    dim3 blockDim(2,3,1);
    PostprocessKernel<<< gridDim, blockDim >>>(B);
//    cudaDeviceSynchronize();
}


void display() {  //callback handler for window re-paint event */
   glClear(GL_COLOR_BUFFER_BIT);  //clear the color buffer
   for(int i=0; i<n; i++)  {
      if(i==0)
         energy=0;
      energy+=pow(B.xSpeed[i],2)+pow(B.ySpeed[i],2);
      if(i==n-1)
         std::cout<<"total system energy: "<<energy<<std::endl;
      for(int j=0;j<n && j!=i;j++)  {  //identifying overlaps
         GLfloat dist = sqrt(pow(B.x[i]-B.x[j],2)+pow(B.y[i]-B.y[j],2));  //distance between centers
         if (dist<B.ballRadius[i]+B.ballRadius[j])  {
            GLfloat dist_sin = (B.y[j]-B.y[i])/(dist+epsilon);
            GLfloat dist_cos = (B.x[j]-B.x[i])/(dist+epsilon);
            GLfloat mid_pt_x = (B.x[j]+B.x[i])/2.0;  //exactly depends on u1, u2
            GLfloat mid_pt_y = (B.y[j]+B.y[i])/2.0;  //exactly depends on u1, u2
            B.x[i]=mid_pt_x-B.ballRadius[i]*dist_cos;  //position change after overlap
            B.y[i]=mid_pt_y-B.ballRadius[i]*dist_sin;
            B.x[j]=2*mid_pt_x-B.x[i];
            B.y[j]=2*mid_pt_y-B.y[i];
/*   
            B[i].xSpeed=cer*pow(B[j].ballRadius,3)*(B[j].xSpeed-B[i].xSpeed)+pow(B[i].ballRadius,3)*B[i].xSpeed+pow(B[j].ballRadius,3)*B[j].xSpeed/(pow(B[i].ballRadius,3)+pow(B[j].ballRadius,3));
            B[i].xSpeed=((pow(B[i].ballRadius,3)-pow(B[j].ballRadius,3))*B[i].xSpeed+2*pow(B[j].ballRadius,3)*B[j].xSpeed)/(pow(B[i].ballRadius,3)+pow(B[j].ballRadius,3))
*/
            GLfloat m1=pow(B.ballRadius[i],3);  //asuming constant density(p) for all balls
            GLfloat m2=pow(B.ballRadius[j],3);
            GLfloat speed1=sqrt(pow(B.xSpeed[i],2)+pow(B.ySpeed[i],2))+epsilon;
            GLfloat speed2=sqrt(pow(B.xSpeed[j],2)+pow(B.ySpeed[j],2))+epsilon;
            GLfloat speed1_cos=B.xSpeed[i]/speed1;
            GLfloat speed1_sin=B.ySpeed[i]/speed1;
            GLfloat speed2_cos=B.xSpeed[j]/speed2;
            GLfloat speed2_sin=B.ySpeed[j]/speed2;
            B.xSpeed[i]=(speed1*(speed1_cos*dist_cos+speed1_sin*dist_sin)*(m1-m2)+2*m2*speed2*(speed2_cos*dist_cos+speed2_sin*dist_sin))*dist_cos/(m1+m2)-speed1*(speed1_sin*dist_cos-speed1_cos*dist_sin)*dist_sin;
            B.ySpeed[i]=(speed1*(speed1_cos*dist_cos+speed1_sin*dist_sin)*(m1-m2)+2*m2*speed2*(speed2_cos*dist_cos+speed2_sin*dist_sin))*dist_sin/(m1+m2)+speed1*(speed1_sin*dist_cos-speed1_cos*dist_sin)*dist_cos;
            B.xSpeed[j]=(speed2*(speed2_cos*dist_cos+speed2_sin*dist_sin)*(m2-m1)+2*m1*speed1*(speed1_cos*dist_cos+speed1_sin*dist_sin))*dist_cos/(m2+m1)-speed2*(speed2_sin*dist_cos-speed2_cos*dist_sin)*dist_sin;
            B.ySpeed[j]=(speed2*(speed2_cos*dist_cos+speed2_sin*dist_sin)*(m2-m1)+2*m1*speed1*(speed1_cos*dist_cos+speed1_sin*dist_sin))*dist_sin/(m2+m1)+speed2*(speed2_sin*dist_cos-speed2_cos*dist_sin)*dist_cos;
/*
            B.xSpeed[i]=-B.xSpeed[i];
            B.ySpeed[i]=-B.ySpeed[i];
            B.xSpeed[j]=-B.xSpeed[j];
            B.ySpeed[j]=-B.ySpeed[j];
*/
         }
      }
      glMatrixMode(GL_MODELVIEW);  //to operate on the model-view matrix
      glLoadIdentity();  //reset model-view matrix
      draw_circle(B.x[i],B.y[i],B.ballRadius[i]);
      B.x[i] += B.xSpeed[i];
      B.y[i] += B.ySpeed[i];
      if (B.x[i] > xMax) {
         B.x[i] = xMax;
         B.xSpeed[i] = -B.xSpeed[i];
      } else if (B.x[i] < xMin) {
         B.x[i] = xMin;
         B.xSpeed[i] = -B.xSpeed[i];
      }
      if (B.y[i] > yMax) {
         B.y[i] = yMax;
         B.ySpeed[i] = -B.ySpeed[i];
      } else if (B.y[i] < yMin) {
         B.y[i] = yMin;
         B.ySpeed[i] = -B.ySpeed[i];
      }
   }
   if(start==1){
      PostprocessCUDA();
//      start=0;
   }
//   PostprocessCUDA();  //(b,n)
   glutSwapBuffers();  //swap front and back buffers (of double buffered mode)
//   refreshMillis+=1;
}

void reshape(GLsizei width, GLsizei height) {  //call back when the windows is re-sized
   if (height == 0) height = 1;  //to prevent divide by 0;  //compute aspect ratio of the new window
   GLfloat aspect = (GLfloat)width / (GLfloat)height;
    glViewport(0, 0, width, height);  //set the viewport to cover the new window
   glMatrixMode(GL_PROJECTION);  //to operate on the projection matrix;  //set the aspect ratio of the clipping area to match the viewport
   glLoadIdentity();  //reset the projection matrix
   if (width >= height) {
      clipAreaXLeft   = -1.0 * aspect;
      clipAreaXRight  = 1.0 * aspect;
      clipAreaYBottom = -1.0;
      clipAreaYTop    = 1.0;
   } else {
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
}
 
void timer(int value) {  //called back when the timer expired
   glutPostRedisplay();  //post a paint request to activate display()
   glutTimerFunc(refreshMillis, timer, 0);  //subsequent timer call at milliseconds
}


int main(int argc, char** argv) {
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
