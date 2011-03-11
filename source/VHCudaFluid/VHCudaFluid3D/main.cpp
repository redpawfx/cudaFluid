#include <windows.h>
#include <stdio.h>
#include <GL/glew.h>
#include "gl_helper.h"
#include <cmath>

namespace cu{
	#include <cuda_runtime_api.h>
	#include <driver_functions.h>
	#include <cuda_gl_interop.h>

	#include <cutil.h>
	#include <cutil_inline_runtime.h>

}

#include "../Common3D/vhFluidSolver3D.h"

typedef unsigned int uint;
typedef unsigned char uchar;

unsigned int timer = 0;

cu::float3 viewRotation= cu::make_float3(1, 0, 0);
cu::float3 viewTranslation = cu::make_float3(0, 0, -12.0f);
float invViewMatrix[12];

//float density = 0.05f;
float density = 1.0f; //0.1

struct CPUAnimBitmap {
	float *pixels;
    int     width, height;
    void    *dataBlock;

	float* get_ptr( void ) const   { return pixels; }
	long image_size( void ) const { return width * height * sizeof(cu::float4); }

};

CPUAnimBitmap  bitmap;
CPUAnimBitmap  sliceBitmap;
VHFluidSolver3D  data;
GLuint gl_Tex;
GLuint gl_sliceTex;
GLuint pbo = 0;     // OpenGL pixel buffer object
struct cu::cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

int pause = 0;
int displaySlice = 0;
int interop = 1;

void initPixelBuffer()
{
    if (pbo) {
		// unregister this buffer object from CUDA C
		cu::cutilSafeCall(cu::cudaGraphicsUnregisterResource(cuda_pbo_resource));

		// delete old buffer
        glDeleteBuffersARB(1, &pbo);
        glDeleteTextures(1, &gl_Tex);
    }

    // create pixel buffer object for display
    glGenBuffersARB(1, &pbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, bitmap.image_size(), 0, GL_STREAM_DRAW_ARB);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // register this buffer object with CUDA
	cu::cutilSafeCall(cu::cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cu::cudaGraphicsMapFlagsWriteDiscard));	

	// create texture for display
	//glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &gl_Tex);
	glBindTexture(GL_TEXTURE_2D, gl_Tex);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, bitmap.width, bitmap.height, 0, GL_RGBA, GL_FLOAT,  NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
}


static void updateFluidParams(VHFluidSolver3D* data) {

	data->f = 0;
	data->fps = 30;
	data->substeps = 1;
	data->jacIter = 20;

	//data->fluidSize = cu::make_float3(10.0,10.0,10.0);
	data->fluidSize = cu::make_float3(4.0,8.0,2.0);

	data->borderPosX = 1;
	data->borderNegX = 1;
	data->borderPosY = 1;
	data->borderNegY = 1;
	data->borderPosZ = 1;
	data->borderNegZ = 1;


	data->densDis = 0.0;
	data->densBuoyStrength = 1;
	data->densBuoyDir = cu::make_float3(0.0,1.0,0.0);

	data->velDamp = 0.01;
	data->vortConf = 5.0; //5

	data->noiseStr = 0.0; //0.4
	data->noiseFreq = 2.0;
	data->noiseOct = 3;
	data->noiseLacun = 4.0;
	data->noiseSpeed = 0.01;
	data->noiseAmp = 0.5;

	data->lightPos = cu::make_float3(10,10,0);

	data->nEmit = 1;
	data->emitters[0].amount = 1;
	data->emitters[0].posX = 0;
	data->emitters[0].posY = -1.6; //-4
	data->emitters[0].posZ = 0;
	data->emitters[0].radius = 0.7;

	data->nColliders = 0;
	data->colliders[0].posX = 0;
	data->colliders[0].posY = 0;
	data->colliders[0].posZ = 0;
	data->colliders[0].radius = 1;

}

// static method used for glut callbacks
static void idle_func( void ) {
    static int ticks = 1;
   // CPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
 //   bitmap->fAnim( bitmap->dataBlock, ticks++ );
    glutPostRedisplay();
}

// static method used for glut callbacks
static void Key(unsigned char key, int x, int y) {
    switch (key) {
		case 27:
     //       CPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
     //       bitmap->animExit( bitmap->dataBlock );
            //delete bitmap;
            exit(0);
		case 'r':
			 reset3DFluid(&data);
			 break;

		case 'p':
			 if(pause==0)
				 pause = 1;
			 else
				pause = 0;
			 break;

		case 's':
			 if(displaySlice==0)
				 displaySlice = 1;
			 else
				displaySlice = 0;
			 break;
    }
}

int vwidth = 512;
int vheight = 512;
float fov = 35;

static void reshapeFunc(int w, int h)
{
    glViewport(0, 0, w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
	/*gluLookAt(0.0,0.0,4.0, 
		      0.0,0.0,0.0,
			  0.0f,1.0f,0.0f);*/
	//glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);
	//glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);
	/*glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);*/

	vwidth = w;
	vheight = h;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	gluPerspective(fov,(float)w/(float)h,1,1000);
    //glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
	//glOrtho(-1.0,1.0,-1.0,1.0,-1.0,1.0);
	//glFrustum(-1.0,1.0,-1.0,1.0,1.5,20.0);

}

int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
        buttonState  |= 1<<button;
    else if (state == GLUT_UP)
        buttonState = 0;

    ox = x; oy = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = x - ox;
    dy = y - oy;

    if (buttonState == 3) {
        // left+middle = zoom
        viewTranslation.z += dy / 100.0;
    } 
    else if (buttonState & 2) {
        // middle = translate
        viewTranslation.x += dx / 100.0;
        viewTranslation.y -= dy / 100.0;
    }
    else if (buttonState & 1) {
        // left = rotate
        viewRotation.x += dy / 5.0;
        viewRotation.y += dx / 5.0;
    }

    ox = x; oy = y;
    glutPostRedisplay();
}

void computeFPS()
{

    char fps[256];
	float ifps = 1.f / (cu::cutGetAverageTimerValue(timer) / 1000.f);
    sprintf(fps, "3D Fluid: %3.1f fps", ifps);  

    glutSetWindowTitle(fps);

	cu::cutResetTimer(timer);  

}

void drawWireCube(float x, float y, float z) {

	glBegin(GL_LINE_STRIP);									// Draw A Quad
		glVertex3f( x, y,-z);					// Top Right Of The Quad (Top)
		glVertex3f(-x, y,-z);					// Top Left Of The Quad (Top)
		glVertex3f(-x, y, z);					// Bottom Left Of The Quad (Top)
		glVertex3f( x, y, z);					// Bottom Right Of The Quad (Top)
		glVertex3f( x, y,-z);
	glEnd();
	
	glBegin(GL_LINE_STRIP);
		glVertex3f( x,-y, z);					// Top Right Of The Quad (Bottom)
		glVertex3f(-x,-y, z);					// Top Left Of The Quad (Bottom)
		glVertex3f(-x,-y,-z);					// Bottom Left Of The Quad (Bottom)
		glVertex3f( x,-y,-z);					// Bottom Right Of The Quad (Bottom)
		glVertex3f( x,-y, z);
	glEnd();
		
	glBegin(GL_LINES);
		glVertex3f(-x, y, z);					// Top Left Of The Quad (Front)
		glVertex3f(-x,-y, z);					// Bottom Left Of The Quad (Front)
		glVertex3f( x, y, z);					// Top Right Of The Quad (Front)
		glVertex3f( x,-y, z);					// Bottom Right Of The Quad (Front)
	
		glVertex3f(-x,-y,-z);					// Top Left Of The Quad (Back)
		glVertex3f(-x, y,-z);					// Bottom Left Of The Quad (Back)
		glVertex3f( x, y,-z);					// Bottom Right Of The Quad (Back)
		glVertex3f( x,-y,-z);					// Top Right Of The Quad (Back)
	glEnd();
}

void display()
{
	cu::cutStartTimer(timer);  


	if (pause == 0)
		solve3DFluid(&data);

	if (displaySlice == 0) {

		GLfloat modelView[16];
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
			glLoadIdentity();
			glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);
			glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);
			glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);
		glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
		glPopMatrix();

		invViewMatrix[0] = modelView[0]; invViewMatrix[1] = modelView[4]; invViewMatrix[2] = modelView[8]; invViewMatrix[3] = modelView[12];
		invViewMatrix[4] = modelView[1]; invViewMatrix[5] = modelView[5]; invViewMatrix[6] = modelView[9]; invViewMatrix[7] = modelView[13];
		invViewMatrix[8] = modelView[2]; invViewMatrix[9] = modelView[6]; invViewMatrix[10] = modelView[10]; invViewMatrix[11] = modelView[14];


		copyInvViewMatrix(invViewMatrix, sizeof(cu::float4)*3);

		int shadows = 1;
		float stepMul = 1;
		float shadowStepMul = 2;
		float shadowThres = 0.9;
		float shadowDens = 0.9;

		if (interop == 1) {

			cu::cutilSafeCall(cu::cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
			cu::float4 *d_output;
			size_t num_bytes; 
			cu::cutilSafeCall(cu::cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,  cuda_pbo_resource));
			cu::cudaMemset(d_output, 0, bitmap.image_size());

			float focalLength = -1/tan(3.14159*fov/(2.0*180));

			render_kernel(&data, d_output, bitmap.width, bitmap.height, density, focalLength,
				shadows, stepMul, shadowStepMul, shadowThres, shadowDens);

			cu::cutilSafeCall(cu::cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

			//glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

			glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
			glBindTexture(GL_TEXTURE_2D, gl_Tex);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, bitmap.width, bitmap.height, GL_RGBA, GL_FLOAT, 0);
			glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

		} else {

			float focalLength = -1/tan(3.14159*fov/(2.0*180));
			render_kernel(&data, data.output_display, bitmap.width, bitmap.height, density, focalLength,
				shadows, stepMul, shadowStepMul, shadowThres, shadowDens);
			cu::cudaMemcpy( bitmap.get_ptr(), data.output_display, bitmap.image_size(), cu::cudaMemcpyDeviceToHost );

			glBindTexture(GL_TEXTURE_2D, gl_Tex);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, bitmap.width, bitmap.height, GL_RGBA, GL_FLOAT, bitmap.get_ptr());

			/*for(int i = 0; i<bitmap.width*bitmap.height; i++) {
				if (bitmap.pixels[i]!=0)
					printf("valf : %f\n", bitmap.pixels[i]);
			}*/

		}

	} else {

		if (interop == 1) {

			cu::cutilSafeCall(cu::cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
			cu::float4 *d_output;
			size_t num_bytes; 
			cu::cutilSafeCall(cu::cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,  cuda_pbo_resource));
			cu::cudaMemset(d_output, 0, sliceBitmap.image_size());

			renderFluidSlice(&data,d_output,0.5,0,2,1.0);

			cu::cutilSafeCall(cu::cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

			//glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

			glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
			glBindTexture(GL_TEXTURE_2D, gl_sliceTex);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, sliceBitmap.width, sliceBitmap.height, GL_RGBA, GL_FLOAT, 0);
			glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

		} else {

			renderFluidSlice(&data,data.output_display_slice,0.5,0,2,1.0);

			cu::cudaMemcpy( sliceBitmap.get_ptr(), data.output_display_slice, sliceBitmap.image_size(), cu::cudaMemcpyDeviceToHost );
			glBindTexture(GL_TEXTURE_2D, gl_sliceTex);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, sliceBitmap.width, sliceBitmap.height, GL_RGBA, GL_FLOAT, sliceBitmap.get_ptr());
		
			/*for(int i = 0; i<sliceBitmap.width*sliceBitmap.height; i++) {
				if(sliceBitmap.pixels[i]!=0)
					printf("valf : %f\n", sliceBitmap.pixels[i]);
			}*/
	}

		}

	//glDrawPixels( bitmap.width, bitmap.height, GL_RGBA, GL_FLOAT, bitmap.pixels );

	// display results
	glClear(GL_COLOR_BUFFER_BIT);

	glDisable(GL_DEPTH_TEST);



	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);


	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	//glBindTexture(GL_TEXTURE_2D, gl_Tex);

	glEnable(GL_TEXTURE_2D);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);

	float ratio = (float)vheight/(float)vwidth;

		glBegin(GL_QUADS);
		glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f*ratio, -1.0f);
		glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f*ratio, -1.0f);
		glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f*ratio, 1.0f);
		glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f*ratio, 1.0f);
		glEnd();


	glDisable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();


	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glLoadIdentity();

	glPushMatrix();
        glLoadIdentity();
		glTranslatef(viewTranslation.x, viewTranslation.y, viewTranslation.z);
		glRotatef(viewRotation.y, 0.0, 1.0, 0.0);
        glRotatef(viewRotation.x, 1.0, 0.0, 0.0);

		//glutWireCube(2);
		drawWireCube(data.fluidSize.x*0.5,data.fluidSize.y*0.5,data.fluidSize.z*0.5);
    glPopMatrix();

	glDisable(GL_BLEND);
  
    glutSwapBuffers();
	glutReportErrors();


	cu::cutStopTimer(timer);  

    computeFPS();
}




int main( void ) {

	int width = 512;
	int height = 512;

	int c=1;
    char* dummy = "";

    glutInit( &c, &dummy );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
    glutInitWindowSize( width, height );
	//glutInitWindowPosition(100, 100);
    glutCreateWindow( "bitmap" );

	if (interop == 1) {

		glewInit();
		cu::cutilSafeCall(cu::cudaGLSetGLDevice( cu::cutGetMaxGflopsDeviceId() ));
	}

	width = 256;
	height = 256;


	bitmap.width = width;
	bitmap.height = height;
	bitmap.pixels = new float[bitmap.width * bitmap.height * sizeof(cu::float4)];

	data.colOutput = 1;
	data.emitters = new FluidEmitter[1];
	data.colliders = new Collider[1];

	int dim = 50;
	init3DFluid(&data,60,120,30);
	//init3DFluid(&data,240,480,120);
	//solve3DFluid(&data);

	sliceBitmap.width = dim;
	sliceBitmap.height = dim;
	sliceBitmap.pixels = new float[sliceBitmap.width * sliceBitmap.height * sizeof(cu::float4)];

	if (data.colOutput==1) {
		//printf("tada");
		cu::cudaMalloc( (void**)&data.output_display,sizeof(cu::float4)*width*height );
		cu::cudaMalloc( (void**)&data.output_display_slice,sizeof(cu::float4)*dim*dim );
	}



	updateFluidParams(&data);
	reset3DFluid(&data);

	solve3DFluid(&data);

    glutKeyboardFunc(Key);
    //glutDisplayFunc(Draw);
	glutDisplayFunc(display);
	glutMouseFunc(mouse);
    glutIdleFunc( idle_func );
	glutReshapeFunc(reshapeFunc);
	glutMotionFunc(motion);


	glGenTextures(1, &gl_sliceTex);
    glBindTexture(GL_TEXTURE_2D, gl_sliceTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, sliceBitmap.width, sliceBitmap.height, 0, GL_RGBA, GL_FLOAT,  sliceBitmap.get_ptr());
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, sliceBitmap.width, sliceBitmap.height, 0, GL_RGBA, GL_FLOAT,  NULL);

//glutReportErrors();

	cu::cutCreateTimer( &timer);

	if (interop == 1) {
		initPixelBuffer();

	} else {

		// create texture for display
		glGenTextures(1, &gl_Tex);
		glBindTexture(GL_TEXTURE_2D, gl_Tex);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, bitmap.width, bitmap.height, 0, GL_RGBA, GL_FLOAT,  bitmap.get_ptr());
	}

    glutMainLoop();

	delete bitmap.pixels;
	delete sliceBitmap.pixels;

	if (interop == 1) {
		cudaGraphicsUnregisterResource(cuda_pbo_resource);
		glDeleteBuffersARB(1, &pbo);
	}

	glDeleteTextures(1, &gl_Tex);
	glDeleteTextures(1, &gl_sliceTex);


	cu::cudaThreadExit();
	//clearFluid(&data);
}