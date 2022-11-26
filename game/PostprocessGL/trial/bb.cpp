#include <math.h>     // Needed for sin, cos
#include <iostream>
#include "bt.h"  //for bt.cu


#define GLEW_STATIC  //specify GLEW_STATIC to use the static linked library (.lib) instead of the dynamic linked library (.dll) for GLEW
#include <GL/glew.h>
#include <GL/glut.h>  // GLUT, includes glu.h and gl.h

int n=100;
int start=1;
struct balls  {
   GLfloat* ballRadius = (GLfloat*)malloc(n*sizeof(GLfloat*));  //radius
   GLfloat* ballX = (GLfloat*)malloc(n*sizeof(GLfloat*));
   GLfloat* ballY = (GLfloat*)malloc(n*sizeof(GLfloat*));
   GLfloat* xSpeed = (GLfloat*)malloc(n*sizeof(GLfloat*));
   GLfloat* ySpeed = (GLfloat*)malloc(n*sizeof(GLfloat*));
}B;


#include <cuda_runtime_api.h>  //CUDA headers
#include <cuda_gl_interop.h>


#define PI 3.14159265f
#define SRC_BUFFER  0
#define DST_BUFFER  1

char title[] = "Bouncing Ball (2D)";  //windowed mode's title
int windowWidth  = 640;  //windowed mode's width
int windowHeight = 480;  //windowed mode's height
int windowPosX   = 50;  //windowed mode's top-left corner x
int windowPosY   = 50;  //windowed mode's top-left corner y
GLfloat ballRadius = 0.05f;  //ball radius
GLfloat ballXMax, ballXMin, ballYMax, ballYMin;  //ball's center (x, y) bounds
int refreshMillis = 30;  //refresh period in milliseconds
GLfloat epsilon=0.0;
GLfloat energy=0.0;
GLdouble clipAreaXLeft, clipAreaXRight, clipAreaYBottom, clipAreaYTop;  //projection clipping area

void initialize_balls()  {
   for(int i=0;i<n;i++)  {
      B.ballRadius[i] = ballRadius;
      B.ballX[i] = 1.0*((static_cast <float> (rand()) - (static_cast <float> (RAND_MAX)/2.0)) / static_cast <float> (RAND_MAX));  //x
      B.ballY[i] = 1.0*((static_cast <float> (rand()) - (static_cast <float> (RAND_MAX)/2.0)) / static_cast <float> (RAND_MAX));  //y
      B.xSpeed[i] = 0.03*((static_cast <float> (rand()) - (static_cast <float> (RAND_MAX)/2.0)) / static_cast <float> (RAND_MAX));  //vx
      B.ySpeed[i] = 0.03*((static_cast <float> (rand()) - (static_cast <float> (RAND_MAX)/2.0)) / static_cast <float> (RAND_MAX));  //vy
   }
}

void initGL() {  //initialize OpenGL graphics
   glClearColor(0.0, 0.0, 0.0, 1.0);  //set background (clear) color to black
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

void display() {  //callback handler for window re-paint event */
   glClear(GL_COLOR_BUFFER_BIT);  //clear the color buffer
   for(int i=0; i<n; i++)  {
      if(i==0)
         energy=0;
      energy+=pow(B.xSpeed[i],2)+pow(B.ySpeed[i],2);
//      if(i==n-1)
//         std::cout<<"total energy of system: "<<energy<<std::endl;
      for(int j=0;j<n,j!=i;j++)  {  //identifying overlaps
         GLfloat dist = sqrt(pow(B.ballX[i]-B.ballX[j],2)+pow(B.ballY[i]-B.ballY[j],2));  //distance between centers
         if (dist<B.ballRadius[i]+B.ballRadius[j])  {
            GLfloat dist_sin = (B.ballY[j]-B.ballY[i])/(dist+epsilon);
            GLfloat dist_cos = (B.ballX[j]-B.ballX[i])/(dist+epsilon);
            GLfloat mid_pt_x = (B.ballX[j]+B.ballX[i])/2.0;
            GLfloat mid_pt_y = (B.ballY[j]+B.ballY[i])/2.0;
            B.ballX[i]=mid_pt_x-B.ballRadius[i]*dist_cos;  //position change after overlap
            B.ballY[i]=mid_pt_y-B.ballRadius[i]*dist_sin;
            B.ballX[j]=2*mid_pt_x-B.ballX[i];
            B.ballY[j]=2*mid_pt_y-B.ballY[i];
/*    
            B[i].xSpeed=cer*pow(B[j].ballRadius,3)*(B[j].xSpeed-B[i].xSpeed)+pow(B[i].ballRadius,3)*B[i].xSpeed+pow(B[j].ballRadius,3)*B[j].xSpeed/(pow(B[i].ballRadius,3)+pow(B[j].ballRadius,3));
            B[i].xSpeed=((pow(B[i].ballRadius,3)-pow(B[j].ballRadius,3))*B[i].xSpeed+2*pow(B[j].ballRadius,3)*B[j].xSpeed)/(pow(B[i].ballRadius,3)+pow(B[j].ballRadius,3))
*/
            GLfloat m1=pow(B.ballRadius[i],3);
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
      draw_circle(B.ballX[i],B.ballY[i],B.ballRadius[i]);
      B.ballX[i] += B.xSpeed[i];
      B.ballY[i] += B.ySpeed[i];
      if (B.ballX[i] > ballXMax) {
         B.ballX[i] = ballXMax;
         B.xSpeed[i] = -B.xSpeed[i];
      } else if (B.ballX[i] < ballXMin) {
         B.ballX[i] = ballXMin;
         B.xSpeed[i] = -B.xSpeed[i];
      }
      if (B.ballY[i] > ballYMax) {
         B.ballY[i] = ballYMax;
         B.ySpeed[i] = -B.ySpeed[i];
      } else if (B.ballY[i] < ballYMin) {
         B.ballY[i] = ballYMin;
         B.ySpeed[i] = -B.ySpeed[i];
      }
   }
   if(start==1){
      PostprocessCUDA(B,n);
//      start=0;
   }
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
   ballXMin = clipAreaXLeft + ballRadius;
   ballXMax = clipAreaXRight - ballRadius;
   ballYMin = clipAreaYBottom + ballRadius;
   ballYMax = clipAreaYTop - ballRadius;
   std::cout<<"x_dist: "<<ballXMax-ballXMin<<"; y_dist: "<<ballYMax-ballYMin<<std::endl;
}
 
void Timer(int value) {  //called back when the timer expired
   glutPostRedisplay();  //post a paint request to activate display()
   glutTimerFunc(refreshMillis, Timer, 0);  //subsequent timer call at milliseconds
}

int main(int argc, char** argv) {  //main() where glut starts
   initialize_balls();
   cudaGLSetGLDevice(0);  //init cuda
   glutInit(&argc, argv);  //initialize GLUT
   glutInitDisplayMode(GLUT_DOUBLE);  //enable double buffered mode
   glutInitWindowSize(windowWidth, windowHeight);  //initial window width and height
   glutInitWindowPosition(windowPosX, windowPosY);  //initial window top-left corner (x, y)
   glutCreateWindow(title);  //window title
   glutDisplayFunc(display);  //register callback handler for window re-paint
   glutReshapeFunc(reshape);  //register callback handler for window re-shape
   glutTimerFunc(0, Timer, 0);  //first timer call immediately
   initGL();  //our own OpenGL initialization
   glutMainLoop();  //enter event-processing loop
   return 0;
}