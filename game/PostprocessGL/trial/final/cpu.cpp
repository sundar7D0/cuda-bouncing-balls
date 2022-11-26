//To compile: g++ cpu.cpp -lglut -lGLU -lGL -lGLEW -o cpu.out
//nvcc gbb.cu -o gbb.out -I /usr/local/cuda-10.1/targets/x86_64-linux/include/ -lcudart -lglut -lGLU -lGL -lGLEW  --for the gpu code.
#include <math.h>  //importing libraries
#include <iostream>
//#include "cutil.h"
//#include "cutil_inline_runtime.h"
#include <iostream>
#define GLEW_STATIC  //to use the static linked library (.lib) instead of the dynamic linked library (.dll) for GLEW
#include <GL/glew.h>
#include <GL/glut.h>  // GLUT includes glu.h and gl.h
//#include <cuda_runtime_api.h>  //CUDA headers
//#include <cuda_gl_interop.h>

clock_t start, end;
#define USE_SHARED_MEM 0  //global variables
#define BLOCK_SIZE 16     // block size
#define PI 3.14159265f
#define SRC_BUFFER  0
#define DST_BUFFER  1
#define eps 1e-30
#define tiny_dist 0.008
#define epsilon 1
char title[] = "Dodge Bubble";  //windowed mode's title
int windowWidth  = 640;  //windowed mode's width
int windowHeight = 480;  //windowed mode's height
int windowPosX   = 50;  //windowed mode's top-left corner x
int windowPosY   = 50;  //windowed mode's top-left corner y
GLfloat ballRadius = 0.025f;  //ball radius
GLfloat xMax, xMin, yMax, yMin;  //ball's center (x, y) bounds
int refreshMillis = 30;  //refresh period in milliseconds
int move_w=0,move_s=0,move_a=0,move_d=0;
GLfloat energy=0.0, dist;
GLdouble clipAreaXLeft, clipAreaXRight, clipAreaYBottom, clipAreaYTop;  //projection clipping area
#define size_n 50  //700  //hyper-parameter
struct balls  {
   GLfloat* ballRadius;  // = (GLfloat*)malloc(n*sizeof(GLfloat*));
   GLfloat* x;  // = (GLfloat*)malloc(n*sizeof(GLfloat*));
   GLfloat* y;  // = (GLfloat*)malloc(n*sizeof(GLfloat*));
   GLfloat* xSpeed;  // = (GLfloat*)malloc(n*sizeof(GLfloat*));
   GLfloat* ySpeed;  // = (GLfloat*)malloc(n*sizeof(GLfloat*));
}B;
int renew=0, renew_max=1000;
int print_=1;
int total=0;

void mem_alloc()  //memory allocation to B
{
   B.ballRadius = (GLfloat*)malloc(size_n*sizeof(GLfloat*));
   B.x = (GLfloat*)malloc(size_n*sizeof(GLfloat*));
   B.y = (GLfloat*)malloc(size_n*sizeof(GLfloat*));
   B.xSpeed = (GLfloat*)malloc(size_n*sizeof(GLfloat*));
   B.ySpeed = (GLfloat*)malloc(size_n*sizeof(GLfloat*));
}

bool valid_ball(int i)  //check for validity of a new ball wrt previous balls
{
   for(int j=0;j<i;j++)
   {
      float dist = sqrt(pow(B.x[i]-B.x[j],2)+pow(B.y[i]-B.y[j],2));  //distance between centers
      if (dist<B.ballRadius[i]+B.ballRadius[j])
         return 1;  //0
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
     B.xSpeed[i] = 0.03*((static_cast <float> (rand()) - (static_cast <float> (RAND_MAX)/2.0)) / static_cast <float> (RAND_MAX));  //v_x
     B.ySpeed[i] = 0.03*((static_cast <float> (rand()) - (static_cast <float> (RAND_MAX)/2.0)) / static_cast <float> (RAND_MAX));  //v_y
   }
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

void naive_collider(struct balls B,int i, int j, GLfloat dist)  //'i' collides with 'j'
{
   GLfloat dist_[2], mid_pt[2], mass[2], speed_i, speed_angle_i[2], speed_j, speed_angle_j[2], dx, dy, lineAngle, alpha;
   if(print_)
        printf("Balls %d and %d have collided!\n",i,j);
   dist_[1] = (B.y[j]-B.y[i])/(dist+eps);  //p=i,o=j
   dist_[0] = (B.x[j]-B.x[i])/(dist+eps);
   mid_pt[0] = (B.x[j]+B.x[i])/2.0;  //exact values depends on u1, u2
   mid_pt[1] = (B.y[j]+B.y[i])/2.0;  //exact values depends on u1, u2
   B.x[i] = mid_pt[0]-B.ballRadius[i]*dist_[0];  //position change after overlap
   B.y[i] = mid_pt[1]-B.ballRadius[i]*dist_[1];
   mass[0] = pow(B.ballRadius[i],3);  //asuming constant density(p) for all balls
   mass[1] = pow(B.ballRadius[j],3);
   speed_i = sqrt(pow(B.xSpeed[i],2)+pow(B.ySpeed[j],2));
   speed_angle_i[0] = B.xSpeed[i]/(speed_i+eps);
   speed_angle_i[1] = B.ySpeed[i]/(speed_i+eps);
   speed_j = sqrt(pow(B.xSpeed[i],2)+pow(B.ySpeed[j],2));
   speed_angle_j[0] = B.xSpeed[j]/(speed_j+eps);
   speed_angle_j[1] = B.ySpeed[j]/(speed_j+eps);
   dx = B.x[i]-B.x[j];
   dy = B.y[i]-B.y[j];
   alpha=lineAngle = atan2(-dx, dy);
   alpha = fmod(lineAngle,2*PI);
   B.xSpeed[i] = (speed_i*(speed_angle_i[0]*cos(alpha)+speed_angle_i[1]*sin(alpha))*(mass[0]-mass[1])+2*mass[1]*speed_j*(speed_angle_j[0]*cos(alpha)+speed_angle_j[1]*sin(alpha)))*cos(alpha)/(mass[0]+mass[1])-speed_i*(speed_angle_i[1]*cos(alpha)-speed_angle_i[0]*sin(alpha))*sin(alpha);  //B.xSpeed[p]=-B.xSpeed[p];   
   B.ySpeed[i] = (speed_i*(speed_angle_i[0]*cos(alpha)+speed_angle_i[1]*sin(alpha))*(mass[0]-mass[1])+2*mass[1]*speed_j*(speed_angle_j[0]*cos(alpha)+speed_angle_j[1]*sin(alpha)))*sin(alpha)/(mass[0]+mass[1])+speed_i*(speed_angle_i[1]*cos(alpha)-speed_angle_i[0]*sin(alpha))*cos(alpha);
   B.xSpeed[j] = (speed_j*(speed_angle_j[0]*cos(alpha)+speed_angle_j[1]*sin(alpha))*(mass[0]-mass[1])+2*mass[1]*speed_i*(speed_angle_i[0]*cos(alpha)+speed_angle_i[1]*sin(alpha)))*cos(alpha)/(mass[0]+mass[1])-speed_j*(speed_angle_j[1]*cos(alpha)-speed_angle_j[0]*sin(alpha))*sin(alpha);  //B.xSpeed[p]=-B.xSpeed[p];   
   B.ySpeed[j] = (speed_j*(speed_angle_j[0]*cos(alpha)+speed_angle_j[1]*sin(alpha))*(mass[0]-mass[1])+2*mass[1]*speed_i*(speed_angle_i[0]*cos(alpha)+speed_angle_i[1]*sin(alpha)))*sin(alpha)/(mass[0]+mass[1])+speed_j*(speed_angle_j[1]*cos(alpha)-speed_angle_j[0]*sin(alpha))*cos(alpha);
}

void display()  //callback handler for window re-paint event
{
   end=clock();
   std::cout<<"Time taken222: "<<double(end-start)/double(CLOCKS_PER_SEC)<<std::endl;
   glClear(GL_COLOR_BUFFER_BIT);  //clear the color buffer
   for(int i=0; i<size_n; i++)  
   {
      if(i==0)
         energy=0;
      energy+=pow(B.xSpeed[i],2)+pow(B.ySpeed[i],2);
//    if(i==n-1)  std::cout<<"total system energy: "<<energy<<std::endl;
      glMatrixMode(GL_MODELVIEW);  //to operate on the model-view matrix
      glLoadIdentity();  //reset model-view matrix
      draw_circle(B.x[i],B.y[i],B.ballRadius[i],i);
   }
   glutSwapBuffers();  //swap front and back buffers (of double buffered mode);  
// refreshMillis+=1;
   for(int i=0;i<size_n;i++)
   {
      B.x[i] = (i==0) ? B.x[i]+(move_d-move_a)*0.08 : B.x[i]+B.xSpeed[i];
      B.y[i] = (i==0) ? B.y[i]+(move_w-move_s)*0.08 : B.y[i]+B.ySpeed[i];
      if (B.y[i] > yMax)
      {
         B.y[i] = yMax;
         B.ySpeed[i] = -B.ySpeed[i];
      }
      else if (B.y[i] < yMin)
      {
         B.y[i] = yMin;
         B.ySpeed[i] = -B.ySpeed[i];
      }
      if (B.x[i] > xMax)
      {
         B.x[i] = xMax;
         B.xSpeed[i] = -B.xSpeed[i];
      } 
      else if (B.x[i] < xMin)
      {
         B.x[i] = xMin;
         B.xSpeed[i] = -B.xSpeed[i];
      }
   }
   start=clock();
   for(int i=0;i<size_n;i++)
   {
      for(int j=0;j<size_n;j++)
      {
         dist=sqrt(pow(B.x[i]-B.x[j],2)+pow(B.y[i]-B.y[j],2));  //identifying overlaps with neighbours using distance between centers
         if(dist<B.ballRadius[i]+B.ballRadius[j]+tiny_dist && i!=j)
         {
//          if(i==0 or j==0)
//            *die=1;
            naive_collider(B,i,j,dist);
         }
      }
   }
   move_w=move_s=move_a=move_d=0;
   end = clock();
  std::cout<<"Time taken: "<<double(end-start)/double(CLOCKS_PER_SEC)<<std::endl;
  GLfloat dist, error=0;
    for(int i=0;i<size_n;i++)
    {
        for(int j=0;j<size_n;j++)
        {
            dist=sqrt(pow(B.x[i]-B.x[j],2)+pow(B.y[i]-B.y[j],2));  //identifying overlaps with neighbours using distance between centers
            if(dist<B.ballRadius[i]+B.ballRadius[j] && i!=j)
            {
                error+=B.ballRadius[i]+B.ballRadius[j]-dist;
            }
        }
    }
    total+=1;
    std::cout<<"Total: "<<total<<std::endl;
   std::cout<<"Average error: "<<error/(size_n*total)<<std::endl;
   start=clock();
}

void reshape(GLsizei width, GLsizei height)  //call back when window is re-sized
{
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

void keyboard(unsigned char key, int x, int y)
{
   switch (key) {
 //       case 033:  //octal equivalent of the Escape key
  //          exit(0);
   //         break;
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
   glutPostRedisplay();
}

int main(int argc, char** argv)
{
   mem_alloc();  //cudaGLSetGLDevice(0);  //init cuda--gave 'gpu resource already being used' error
   initBalls();
   glutInit(&argc, argv);  //initialize GLUT
   glutInitDisplayMode(GLUT_DOUBLE);  //enable double buffered mode
   glutInitWindowSize(windowWidth, windowHeight);  //initial window width and height
   glutInitWindowPosition(windowPosX, windowPosY);  //initial window top-left corner (x, y)
   glutCreateWindow(title);  //window title
   glutDisplayFunc(display);  //register callback handler for window re-paint
   glutKeyboardFunc(keyboard);
   glutReshapeFunc(reshape);  //register callback handler for window re-shape
   glutTimerFunc(0, timer, 0);  //first timer call immediately
   glClearColor(0.0, 0.0, 0.0, 1.0);  //set background (clear) color to black
   glutMainLoop();  //enter event-processing loop
   return 0;
}