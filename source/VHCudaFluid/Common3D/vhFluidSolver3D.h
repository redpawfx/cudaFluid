#ifndef __FLUID3D_H__
#define __FLUID3D_H__

#include "vhObjects3D.h"

namespace cu{
	#include <vector_functions.h>
}

typedef unsigned int  uint;
typedef unsigned char uchar;


struct VHFluidSolver3D {

	int f;
	int nEmit;
	FluidEmitter* emitters;

	int nColliders;
	Collider* colliders;

	int fps;
	int substeps;
	int jacIter;
	
	cu::cudaExtent res;

	cu::float3 fluidSize;

	int borderNegX;
	int borderPosX;
	int borderNegY;
	int borderPosY;
	int borderNegZ;
	int borderPosZ;

	float densDis;
	float densBuoyStrength;
	cu::float3 densBuoyDir;

	float velDamp;
	float vortConf;

	float noiseStr;
	float noiseFreq;
	int noiseOct;
	float noiseLacun;
	float noiseSpeed;
	float noiseAmp;

	cu::float3 lightPos;

	int colOutput;

	cu::float4		*output_display;
	cu::float4		*output_display_slice;

	float			*dev_noise;
	cu::float4      *dev_vel;
	float			*dev_dens;
	float           *dev_pressure;
	float           *dev_div;
	cu::float4           *dev_vort;
	cu::float4		*dev_obstacles;

    //cudaEvent_t     start, stop;
    float           totalTime;
    float           frames;

	long domainSize( void ) const { return res.width * res.height * res.depth * sizeof(float); }

};

extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);
extern "C" void render_kernel(VHFluidSolver3D* fluidSolver, cu::float4 *d_output, uint imageW, uint imageH,
							  float density, float focalLength,
							  int doShadows, float stepMul, float shadowStepMul, float shadowThres, float shadowDens);

extern "C" void init3DFluid(VHFluidSolver3D* fluidSolver, int dimX, int dimY, int dimZ);
extern "C" void clear3DFluid(VHFluidSolver3D* fluidSolver);
extern "C" void solve3DFluid(VHFluidSolver3D* fluidSolver);
extern "C" void reset3DFluid(VHFluidSolver3D* fluidSolver);
extern "C" void renderFluidSlice(VHFluidSolver3D* d, cu::float4* d_output,float slice,
								 int sliceType, int sliceAxis, float sliceBounds);

#endif  // __DATABLOCK_H__