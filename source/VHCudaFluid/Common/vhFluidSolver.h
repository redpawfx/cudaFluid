#ifndef __FLUID2D_H__
#define __FLUID2D_H__

#include "vhObjects.h"

namespace cu{
	#include <vector_functions.h>
}

typedef unsigned int  uint;
typedef unsigned char uchar;

struct VHFluidSolver {

	int f;
	int nEmit;
	FluidEmitter* emitters;

	int nColliders;
	Collider* colliders;

	int fps;
	int substeps;
	int jacIter;

	cu::int2 res;

	cu::float2 fluidSize;

	float densDis;
	float densBuoyStrength;
	cu::float2 densBuoyDir;

	float velDamp;
	float vortConf;

	float noiseStr;
	float noiseFreq;
	int noiseOct;
	float noiseLacun;
	float noiseSpeed;
	float noiseAmp;

	int colOutput;

	int borderPosX;
	int borderNegX;
	int borderPosY;
	int borderNegY;

	cu::float4		*output_display;

	float			*dev_noise;
	cu::float2          *dev_vel;
	float           *dev_dens;
	float           *dev_pressure;
	float           *dev_div;
	float           *dev_vort;
	cu::float4			*dev_obstacles;


    //cudaEvent_t     start, stop;
    float           totalTime;
    float           frames;

	long domainSize( void ) const { return res.x * res.y * sizeof(float); }

};

extern "C" void initFluid(VHFluidSolver* fluidSolver, int dimX, int dimY);
extern "C" void clearFluid(VHFluidSolver* fluidSolver);
extern "C" void solveFluid(VHFluidSolver* fluidSolver);
extern "C" void resetFluid(VHFluidSolver* fluidSolver);
extern "C" void renderFluid(VHFluidSolver* fluidSolver, cu::float4* d_output, int previewType, float maxBounds);

#endif  // __DATABLOCK_H__