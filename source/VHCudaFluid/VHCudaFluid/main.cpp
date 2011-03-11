#include <windows.h>
#include <stdio.h>
#include <GL/glew.h>
#include "gl_helper.h"

namespace cu{
	#include <cuda_runtime_api.h>

	#include <cuda_gl_interop.h>

	#include <cutil.h>
	#include <cutil_inline_runtime.h>

}

#include "../Common/vhFluidSolver.h"

struct CPUAnimBitmap {
	float *pixels;
    int     width, height;
    void    *dataBlock;

	float* get_ptr( void ) const   { return pixels; }
	long image_size( void ) const { return width * height * sizeof(cu::float4); }

};

CPUAnimBitmap  bitmap;
VHFluidSolver  data;
GLuint gl_Tex;

GLuint pbo = 0;     // OpenGL pixel buffer object
struct cu::cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

int pause = 0;
unsigned int timer = 0;

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
	glGenTextures(1, &gl_Tex);
	glBindTexture(GL_TEXTURE_2D, gl_Tex);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, bitmap.width, bitmap.height, 0, GL_RGBA, GL_FLOAT,  NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
}


static void updateFluidParams(VHFluidSolver* data) {

	data->f = 0;
	data->fps = 30;
	data->substeps = 1;
	data->jacIter = 50;

	data->fluidSize = cu::make_float2(10.0,10.0);

	data->borderPosX = 1;
	data->borderNegX = 1;
	data->borderPosY = 1;
	data->borderNegY = 1;

	data->densDis = 0.0;
	data->densBuoyStrength = 1;
	data->densBuoyDir = cu::make_float2(0.0,1.0);

	data->velDamp = 0.01;
	data->vortConf = 5.0; //5

	data->noiseStr = 1.0; //1.0
	data->noiseFreq = 1.0;
	data->noiseOct = 3;
	data->noiseLacun = 4.0;
	data->noiseSpeed = 0.01;
	data->noiseAmp = 0.5;

	data->nEmit = 1;
	data->emitters[0].amount = 1;
	data->emitters[0].posX = 0;
	data->emitters[0].posY = -4;
	data->emitters[0].radius = 0.5;

	data->nColliders = 0;
	data->colliders[0].posX = 0;
	data->colliders[0].posY = 0;
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
			 resetFluid(&data);
			 break;
		  case 'p':
			 if(pause==0)
				 pause = 1;
			 else
				pause = 0;
			 break;
    }
}

static void reshapeFunc(int w, int h)
{
    glViewport(0, 0, w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

}

void computeFPS()
{

    char fps[256];
	float ifps = 1.f / (cu::cutGetAverageTimerValue(timer) / 1000.f);
    sprintf(fps, "2D Fluid: %3.1f fps", ifps);  

    glutSetWindowTitle(fps);

	cu::cutResetTimer(timer);  

}

// static method used for glut callbacks
static void Draw( void ) {

	cu::cutStartTimer(timer);  

    glClearColor( 0.0, 0.0, 0.0, 1.0 );
    glClear( GL_COLOR_BUFFER_BIT );

	if (pause == 0)
		solveFluid(&data);

	if (interop == 1) {

		cu::cutilSafeCall(cu::cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
		cu::float4 *d_output;
		size_t num_bytes; 
		cu::cutilSafeCall(cu::cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,  cuda_pbo_resource));
		cu::cudaMemset(d_output, 0, bitmap.image_size());

		//render_kernel(&data, d_output, bitmap.width, bitmap.height, density, brightness);
		renderFluid(&data,d_output,0,1);

		cu::cutilSafeCall(cu::cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

		//glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
		glBindTexture(GL_TEXTURE_2D, gl_Tex);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, bitmap.width, bitmap.height, GL_RGBA, GL_FLOAT, 0);
		glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	} else {

		renderFluid(&data,data.output_display,0,1);

		cu::cudaMemcpy( bitmap.get_ptr(), data.output_display, bitmap.image_size(), cu::cudaMemcpyDeviceToHost );
		glBindTexture(GL_TEXTURE_2D, gl_Tex);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, bitmap.width, bitmap.height, GL_RGBA, GL_FLOAT, bitmap.get_ptr());

	}

	//glDrawPixels( bitmap.width, bitmap.height, GL_RGBA, GL_FLOAT, bitmap.pixels );


	glDisable(GL_DEPTH_TEST);

	glEnable(GL_TEXTURE_2D);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 0.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 0.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, 1.0f);
	glEnd();

	glDisable(GL_TEXTURE_2D);

	glBindTexture(GL_TEXTURE_2D, 0);

    glutSwapBuffers();
	//glutReportErrors();

	cu::cutStopTimer(timer);  

    computeFPS();
}

int main( void ) {

	int c=1;
    char* dummy = "";
	
	int width = 512;
	int height = 512;

    glutInit( &c, &dummy );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
    glutInitWindowSize( width, height );
	//glutInitWindowPosition(100, 100);
    glutCreateWindow( "bitmap" );

	if (interop == 1) {

		glewInit();
		cu::cutilSafeCall(cu::cudaGLSetGLDevice( cu::cutGetMaxGflopsDeviceId() ));
	}

	int dim = 200;

	bitmap.width = dim;
	bitmap.height = dim;
	bitmap.pixels = new float[bitmap.width * bitmap.height * 4];

	data.colOutput = 1;
	data.emitters = new FluidEmitter[1];
	data.colliders = new Collider[1];

	initFluid(&data,dim,dim);
	updateFluidParams(&data);
	resetFluid(&data);

    glutKeyboardFunc(Key);
    glutDisplayFunc(Draw);
   // if (clickDrag != NULL)
   //     glutMouseFunc( mouse_func );
    glutIdleFunc( idle_func );
	glutReshapeFunc(reshapeFunc);

	if (interop == 1) {
		initPixelBuffer();

	} else {

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

	cu::cutCreateTimer( &timer);

    glutMainLoop();

	cu::cudaThreadExit();
	clearFluid(&data);
}