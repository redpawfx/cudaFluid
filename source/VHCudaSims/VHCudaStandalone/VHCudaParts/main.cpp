#include <stdio.h>
#include <GL/glew.h>

#include "../../CudaCommon/CudaParticlesSystem/vhParticlesSystem.h"

namespace cu {
	#include <cuda_runtime_api.h>
	#include <cuda_gl_interop.h>

	#include <vector_functions.h>
	#include <cutil.h>
	#include <cutil_inline_runtime.h>
}

#include "GL/glext.h"
#include "GL/glut.h"

void computeFPS();

int vwidth = 512;
int vheight = 512;
float fov = 35;

unsigned int timer = 0;

cu::float3 viewRotation= cu::make_float3(1, 0, 0);
cu::float3 viewTranslation = cu::make_float3(0, 0, -12.0f);

void idle_func( void ) {
    static int ticks = 1;
    glutPostRedisplay();
}

void Key(unsigned char key, int x, int y) {

    switch (key) {
		case 27:
            //exit(0);

		case 'r':
			
			 break;

		case 'p':
			
			 break;

		case 's':
			
			 break;
    }
}

void reshapeFunc(int w, int h) {

    glViewport(0, 0, w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

	vwidth = w;
	vheight = h;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	gluPerspective(fov,(float)w/(float)h,1,1000);

}

int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y){

    if (state == GLUT_DOWN)
        buttonState  |= 1<<button;
    else if (state == GLUT_UP)
        buttonState = 0;

    ox = x; oy = y;
    glutPostRedisplay();
}

void motion(int x, int y){

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

void display(){

	cu::cutStartTimer(timer);

	VHParticlesSystem* particlesSystem = VHParticlesSystem::systemsList[0];

	// maxparts/life
	particlesSystem->emitParticles();

	particlesSystem->updateParticles();

	particlesSystem->noiseOffset.y +=0.01;


	glClear(GL_COLOR_BUFFER_BIT);

	glEnable( GL_BLEND );
	glBlendFunc( GL_SRC_ALPHA, GL_ONE );

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(viewTranslation.x, viewTranslation.y, viewTranslation.z);
	glRotatef(viewRotation.y, 0.0, 1.0, 0.0);
    glRotatef(viewRotation.x, 1.0, 0.0, 0.0);

	//glutWireCube(2);

	particlesSystem->draw();

	glDisable( GL_BLEND );
  
    glutSwapBuffers();
	glutReportErrors();


	cu::cutStopTimer(timer);  

    computeFPS();
}

void computeFPS() {

    char fps[256];
	float ifps = 1.f / (cu::cutGetAverageTimerValue(timer) / 1000.f);
    sprintf(fps, "Parts: %3.1f fps", ifps);  

    glutSetWindowTitle(fps);

	cu::cutResetTimer(timer);  

}


int main( void ) {

	int c=1;
    char* dummy = "";

    glutInit( &c, &dummy );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
    glutInitWindowSize( vwidth, vheight );
    glutCreateWindow( "parts" );

	glewInit();
	cu::cutilSafeCall(cu::cudaGLSetGLDevice(cu::cutGetMaxGflopsDeviceId() ));

	glutKeyboardFunc(Key);
	glutDisplayFunc(display);
	glutMouseFunc(mouse);
    glutIdleFunc( idle_func );
	glutReshapeFunc(reshapeFunc);
	glutMotionFunc(motion);

	cu::cutCreateTimer( &timer);

	VHParticlesSystem* particlesSystem = new VHParticlesSystem();
	//particlesSystem->initParticlesSystem(5000000);
	particlesSystem->initParticlesSystem(4000000);

	particlesSystem->emitters[0].amount = 200000;
	//particlesSystem->gravity = cu::make_float3(0,0,0);
	particlesSystem->noiseAmp = cu::make_float3(2,2,2);
	
	glutMainLoop();


	delete particlesSystem;

	cu::cudaThreadExit();

}