#include <stdio.h>
#include <cuda.h>
#include <cutil_math.h>

#include "perlinKernel.cu";

#include "vhObjects.h"

#define PI 3.1415926535897932f

texture<float,2>  texNoise;
static cudaArray *noiseArray = NULL;

texture<float2,2>  texVel;
static cudaArray *velArray = NULL;

texture<float,2>  texDens;
static cudaArray *densArray = NULL;

texture<float,2>  texPressure;
static cudaArray *pressureArray = NULL;

texture<float,2>  texDiv;
static cudaArray *divArray = NULL;

texture<float,2>  texVort;
static cudaArray *vortArray = NULL;

texture<float4,2> texObstacles;
static cudaArray *obstArray = NULL;

cudaChannelFormatDesc descFloat;
cudaChannelFormatDesc descFloat2;
cudaChannelFormatDesc descFloat4;


struct VHFluidSolver {

	int f;
	int nEmit;
	FluidEmitter* emitters;

	int nColliders;
	Collider* colliders;

	int fps;
	int substeps;
	int jacIter;

	int2 res;

	float2 fluidSize;

	float densDis;
	float densBuoyStrength;
	float2 densBuoyDir;

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

    float4		*output_display;

	float			*dev_noise;
    float2          *dev_vel;
	float           *dev_dens;
	float           *dev_pressure;
	float           *dev_div;
	float           *dev_vort;
	float4			*dev_obstacles;


    //cudaEvent_t     start, stop;
    float           totalTime;
    float           frames;

	long domainSize( void ) const { return res.x * res.y * sizeof(float); }
};

__device__ float linstep2d(float val, float minval, float maxval) {

	return clamp((val-minval)/(maxval-minval), -1.0f, 1.0f);

}

__global__ void obst_to_color( float4 *optr, const float4 *outSrc, int2 gres ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    //int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float4 l = outSrc[offset];
		optr[offset].x=optr[offset].y=optr[offset].z=optr[offset].w = l.w;

	}
}

__global__ void float_to_color( float4 *optr, const float *outSrc, int2 gres, float minBound, float maxBound ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    //int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float l = outSrc[offset];
		optr[offset].x=optr[offset].y=optr[offset].z=optr[offset].w = linstep2d(l,minBound,maxBound);

	}
}


__global__ void float2_to_color( float4 *optr, const float2 *outSrc, int2 gres,float minBound, float maxBound ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    //int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float r = outSrc[offset].x;
		float g = outSrc[offset].y;

		optr[offset].x=linstep2d(r,minBound,maxBound);
		optr[offset].y=linstep2d(g,minBound,maxBound);
		optr[offset].z=0.5;
		optr[offset].w = 1;

	}
}

__global__ void createBorder(float4 *obst, int2 gres, int posX, int negX, int posY, int negY) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
   // int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		obst[offset] = make_float4(0,0,0,0);

		if (negX == 1 && x == 0) {
			obst[offset].w = 1;
		}

		if(posX == 1 && x==(gres.x-1)) {
			obst[offset].w = 1;
		}

		if(negY == 1 && y==0) {
			obst[offset].w = 1;
		}

		if(posY == 1 && y==(gres.y-1))  {
			obst[offset].w = 1;
		}

	}
}

__global__ void addCollider(float4 *obst, float radius, float2 position, int2 gres, float2 vel) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    //int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float2 coords = make_float2(x,y);
		float2 pos = (position - coords);
		float scaledRadius = radius;


		if (dot(pos,pos)<(scaledRadius*scaledRadius)) {
			obst[offset].x = vel.x;
			obst[offset].y = vel.y;
			obst[offset].w = 1;
		}

	}
}

__global__ void addDens(float *dens, float timestep, float radius, float amount, float2 position, int2 gres) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
   // int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float2 coords = make_float2(x,y);
		float2 pos = (position - coords);

		if (dot(pos,pos)<(radius*radius))
			dens[offset] += timestep*amount;


	}

}

__global__ void addVel(float2 *vel, float timestep, float radius, float2 strength, float2 position, int2 gres) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    //int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float2 coords = make_float2(x,y);
		float2 pos = (position - coords);

		if (dot(pos,pos)<(radius*radius))
			vel[offset] += timestep*strength;

	}
}

__global__ void addDensBuoy(float2 *vel, float timestep, float strength, float2 dir, int2 gres) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
   // int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float xc = x+0.5;
		float yc = y+0.5;

		vel[offset] += timestep*strength*dir*tex2D(texDens,xc,yc);

	}
}

//Simple kernel fills an array with perlin noise
__global__ void k_perlin(float* noise, unsigned int width, unsigned int height, 
			 float2 delta, unsigned char* d_perm,
			 float time, int octaves, float lacun, float gain, float freq, float amp)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float xCur = ((float) (idx%width)) * delta.x;
  float yCur = ((float) (idx/width)) * delta.y;

 if(threadIdx.x < 256)
    // Optimization: this causes bank conflicts
    s_perm[threadIdx.x] = d_perm[threadIdx.x];
  // this synchronization can be important if there are more that 256 threads
  __syncthreads();

  
  // Each thread creates one pixel location in the texture (textel)
  if(idx < width*height) {
    noise[idx] = noise1D(xCur, yCur, time, octaves, lacun, gain, freq, amp);
	//noise[idx] = noise1D(xCur, yCur, z, octaves, 2.f, 0.75f, 0.3, 0.5);
  }
}

__global__ void addNoise(float2 *vel, float timestep, float strength, int2 gres) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    //int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float xc = x+0.5;
		float yc = y+0.5;

		float noise = strength*timestep*tex2D(texNoise,xc,yc)*tex2D(texDens,xc,yc);
		
		vel[offset] += make_float2(noise,noise);

	}
}


__global__ void advectVel(float2 *vel, float timestep, float dissipation, float2 invGridSize, int2 gres) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
   // int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float xc = x+0.5;
		float yc = y+0.5;

		float solid = tex2D(texObstacles,xc,yc).w;

		if (solid > 0) {
			vel[offset] = make_float2(0,0);
			return;
		}

		float2 coords = make_float2(xc,yc);

		float2 pos = coords - timestep * invGridSize * tex2D(texVel,xc,yc)*make_float2((float)gres.x,(float)gres.y);

		vel[offset] = (1-dissipation*timestep) * tex2D(texVel, pos.x,pos.y);

	}

}

__global__ void advectDens(float *dens, float timestep, float dissipation, float2 invGridSize, int2 gres) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    //int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float xc = x+0.5;
		float yc = y+0.5;

		float solid = tex2D(texObstacles,xc,yc).w;

		if (solid > 0) {
			dens[offset] = 0;
			return;
		}

		float2 coords = make_float2(xc,yc);

		float2 pos = coords - timestep * invGridSize * tex2D(texVel,xc,yc)*make_float2((float)gres.x,(float)gres.y);

		dens[offset] = (1-dissipation*timestep) * tex2D(texDens, pos.x,pos.y);

	}
}


__global__ void divergence(float *div, int2 gres, float2 invCellSize) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
   // int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float xc = x+0.5;
		float yc = y+0.5;

		float2 vL = tex2D(texVel,xc-1,yc);
		float2 vR = tex2D(texVel,xc+1,yc);
		float2 vT = tex2D(texVel,xc,yc+1);
		float2 vB = tex2D(texVel,xc,yc-1);

		//obstacles
		float4 oL = tex2D(texObstacles,xc-1,yc);
		float4 oR = tex2D(texObstacles,xc+1,yc);
		float4 oT = tex2D(texObstacles,xc,yc+1);
		float4 oB = tex2D(texObstacles,xc,yc-1);

		// Use obstacle velocities for solid cells:
		if (oL.w>0) vL = make_float2(oL.x,oL.y);
		if (oR.w>0) vR = make_float2(oR.x,oR.y);
		if (oT.w>0) vT = make_float2(oT.x,oT.y);
		if (oB.w>0) vB = make_float2(oB.x,oB.y);

		div[offset] = 0.5 * (invCellSize.x*(vR.x - vL.x) + invCellSize.y*(vT.y - vB.y));

	}
}

__global__ void vorticity(float *vort, int2 gres, float2 invCellSize) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    //int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float xc = x+0.5;
		float yc = y+0.5;

		float2 vL = tex2D(texVel,xc-1,yc);
		float2 vR = tex2D(texVel,xc+1,yc);
		float2 vT = tex2D(texVel,xc,yc+1);
		float2 vB = tex2D(texVel,xc,yc-1);

		//obstacles
		float4 oL = tex2D(texObstacles,xc-1,yc);
		float4 oR = tex2D(texObstacles,xc+1,yc);
		float4 oT = tex2D(texObstacles,xc,yc+1);
		float4 oB = tex2D(texObstacles,xc,yc-1);

		// Use obstacle velocities for solid cells:
		if (oL.w>0) vL = make_float2(oL.x,oL.y);
		if (oR.w>0) vR = make_float2(oR.x,oR.y);
		if (oT.w>0) vT = make_float2(oT.x,oT.y);
		if (oB.w>0) vB = make_float2(oB.x,oB.y);

		vort[offset] = 0.5 * (invCellSize.x*(vR.y - vL.y) - invCellSize.y*(vT.x - vB.x));

	}
}

__global__ void vortConf(float2 *vel, float timestep, float strength, int2 gres) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    //int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float xc = x+0.5;
		float yc = y+0.5;

		float vortL = tex2D(texVort,xc-1,yc);
		float vortR = tex2D(texVort,xc+1,yc);
		float vortT = tex2D(texVort,xc,yc+1);
		float vortB = tex2D(texVort,xc,yc-1);

		float vortC = tex2D(texVort,xc,yc);

		float2 force = 0.5*make_float2(gres.x*(abs(vortT)-abs(vortB)), gres.y*(abs(vortR) - abs(vortL)));
		const float EPSILON = 2.4414e-4; // 2^-12

		float magSqr = max(EPSILON, dot(force, force)); 
  		force *= pow((float)magSqr,(float)-0.5); 
 		force =  strength * vortC * make_float2(1, -1) * force;

		vel[offset] += force*timestep;

	}
}

__global__ void jacobi(float *pressure, float alpha, float rBeta, int2 gres) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    //int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float xc = x+0.5;
		float yc = y+0.5;

		float pL = tex2D(texPressure,xc-1,yc);
		float pR = tex2D(texPressure,xc+1,yc);
		float pT = tex2D(texPressure,xc,yc+1);
		float pB = tex2D(texPressure,xc,yc-1);

		float pC = tex2D(texPressure,xc,yc);

		//obstacles
		float4 oL = tex2D(texObstacles,xc-1,yc);
		float4 oR = tex2D(texObstacles,xc+1,yc);
		float4 oT = tex2D(texObstacles,xc,yc+1);
		float4 oB = tex2D(texObstacles,xc,yc-1);

		// Use center pressure for solid cells:
		if (oL.w>0) pL = pC;
		if (oR.w>0) pR = pC;
		if (oT.w>0) pT = pC;
		if (oB.w>0) pB = pC;


		float dC = tex2D(texDiv,xc,yc);

		pressure[offset] = (pL + pR + pB + pT + alpha * dC) * rBeta;

	}
}

__global__ void projection(float2 *vel, int2 gres, float2 invCellSize) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    //int offset = x + y * blockDim.x * gridDim.x;
	int offset = x + y * gres.x;

	if (x<gres.x && y<gres.y) {

		float xc = x+0.5;
		float yc = y+0.5;

		float pL = tex2D(texPressure,xc-1,yc);
		float pR = tex2D(texPressure,xc+1,yc);
		float pT = tex2D(texPressure,xc,yc+1);
		float pB = tex2D(texPressure,xc,yc-1);

		float pC = tex2D(texPressure,xc,yc);

			//obstacles
		float4 oL = tex2D(texObstacles,xc-1,yc);
		float4 oR = tex2D(texObstacles,xc+1,yc);
		float4 oT = tex2D(texObstacles,xc,yc+1);
		float4 oB = tex2D(texObstacles,xc,yc-1);

		float2 obstV = make_float2(0,0);
		float2 vMask = make_float2(1,1);

		if (oT.w > 0) { pT = pC; obstV.y = oT.y; vMask.y = 0; }
		if (oB.w > 0) { pB = pC; obstV.y = oB.y; vMask.y = 0; }
		if (oR.w > 0) { pR = pC; obstV.x = oR.x; vMask.x = 0; }
		if (oL.w > 0) { pL = pC; obstV.x = oL.x; vMask.x = 0; }

		float2 grad = 0.5*make_float2(invCellSize.x*(pR-pL), invCellSize.y*(pT-pB));

		float2 vNew = tex2D(texVel,xc,yc) - grad;

		vel[offset] = vMask*vNew + obstV;

	}
}




static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void swapDataPointers(void **a, void **b) {
	void* temp = *a;
	*a = *b;
	*b = temp;
}

static void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

void calcNoise(VHFluidSolver* d) {

  int nThreads=256; // must be equal or larger than 256! (see s_perm)
  int totalThreads = d->res.x * d->res.y;
  int nBlocks = totalThreads/nThreads; 
  nBlocks += ((totalThreads%nThreads)>0)?1:0;
  
  float xExtent = d->fluidSize.x;
  float yExtent = d->fluidSize.y;
  float xDelta = xExtent/(float)d->res.x;
  float yDelta = yExtent/(float)d->res.y;
  
  if(!d_perm) { // for convenience allocate and copy d_perm here
    cudaMalloc((void**) &d_perm,sizeof(h_perm));
    cudaMemcpy(d_perm,h_perm,sizeof(h_perm),cudaMemcpyHostToDevice);
    checkCUDAError("d_perm malloc or copy failed!");
  }

  k_perlin<<< nBlocks, nThreads>>>(d->dev_noise, d->res.x, d->res.y, make_float2(xDelta, yDelta), d_perm, d->f*d->noiseSpeed,
									d->noiseOct, d->noiseLacun, 0.75f, d->noiseFreq, d->noiseAmp);
									//3,4,0.75,1,0.5
  
  // make certain the kernel has completed 
  cudaThreadSynchronize();
  checkCUDAError("kernel failed!");


}


extern "C" void solveFluid(VHFluidSolver* d) {

	//printf("frame : %d\n", d->f);

	dim3    blocks(d->res.x/16 + (!(d->res.x%16)?0:1),d->res.y/16 + (!(d->res.y%16)?0:1));
    dim3    threads(16,16);


	float timestep = 1.0/(d->fps*d->substeps);
	float radius = 0;
	float2 position = make_float2(0,0);
	float2 invGridSize = make_float2(1/d->fluidSize.x,1/d->fluidSize.y);
	//float2 invCellSize = make_float2(d->res.x/d->fluidSize.x, d->res.y/d->fluidSize.y);
	float2 invCellSize = make_float2(1.0,1.0);

	float alpha = -(1.0/invCellSize.x*1.0/invCellSize.y);
	float rBeta = 0.25;

	for (int i=0; i<d->substeps; i++) {

		createBorder<<<blocks,threads>>>(d->dev_obstacles,d->res,d->borderPosX,d->borderNegX,d->borderPosY,d->borderNegY);

		for (int j=0; j<d->nColliders; j++) {
			position = make_float2(d->res.x/2+d->res.x/d->fluidSize.x*d->colliders[j].posX,d->res.y/2+d->res.y/d->fluidSize.y*d->colliders[j].posY);
			radius = d->colliders[j].radius*d->res.x/(d->fluidSize.x);
			float2 vel = 1.0f/timestep * make_float2((d->colliders[j].posX - d->colliders[j].oldPosX),
									d->colliders[j].posY - d->colliders[j].oldPosY);

			addCollider<<<blocks,threads>>>(d->dev_obstacles,radius, position, d->res, vel);
		}
	
		
		cudaMemcpyToArray( velArray, 0, 0, d->dev_vel, d->res.x*d->res.y*sizeof(float2), cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray( obstArray, 0, 0, d->dev_obstacles, d->res.x*d->res.y*sizeof(float4), cudaMemcpyDeviceToDevice);
		advectVel<<<blocks,threads>>>(d->dev_vel,timestep,d->velDamp,invGridSize,d->res);


		cudaMemcpyToArray( velArray, 0, 0, d->dev_vel, d->res.x*d->res.y*sizeof(float2), cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray( densArray, 0, 0, d->dev_dens, d->res.x*d->res.y*sizeof(float), cudaMemcpyDeviceToDevice);
		advectDens<<<blocks,threads>>>(d->dev_dens,timestep,d->densDis,invGridSize,d->res);
		
		for (int j=0; j<d->nEmit; j++) {
			//addDens<<<blocks,threads>>>(d->dev_inDens,timestep,radius,invGridSize,0.2,position);
			position = make_float2(d->res.x/2+d->res.x/d->fluidSize.x*d->emitters[j].posX,d->res.y/2+d->res.y/d->fluidSize.y*d->emitters[j].posY);
			radius = d->emitters[j].radius*d->res.x/(d->fluidSize.x);
			addDens<<<blocks,threads>>>(d->dev_dens,timestep,radius,d->emitters[j].amount,position,d->res);
		}

		//addVel<<<blocks,threads>>>(d->dev_inVel,timestep,radius,invGridSize,make_float2(0.0,0.02),position);

		addDensBuoy<<<blocks,threads>>>(d->dev_vel,timestep,d->densBuoyStrength,make_float2(d->densBuoyDir.x,d->densBuoyDir.y),d->res);

		if(d->noiseStr != 0) {
			calcNoise(d);
			cudaMemcpyToArray( noiseArray, 0, 0, d->dev_noise, d->res.x*d->res.y*sizeof(float), cudaMemcpyDeviceToDevice);
			addNoise<<<blocks,threads>>>(d->dev_vel, timestep, d->noiseStr, d->res);
		} else {
			cudaMemset(d->dev_noise,0, sizeof(float) * d->res.x * d->res.y);
		}

		cudaMemcpyToArray( velArray, 0, 0, d->dev_vel, d->res.x*d->res.y*sizeof(float2), cudaMemcpyDeviceToDevice);
		vorticity<<<blocks,threads>>>(d->dev_vort,d->res, invCellSize);

		cudaMemcpyToArray( vortArray, 0, 0, d->dev_vort, d->res.x*d->res.y*sizeof(float), cudaMemcpyDeviceToDevice);
		vortConf<<<blocks,threads>>>(d->dev_vel,timestep,d->vortConf,d->res);

		cudaMemcpyToArray( velArray, 0, 0, d->dev_vel, d->res.x*d->res.y*sizeof(float2), cudaMemcpyDeviceToDevice);
		divergence<<<blocks,threads>>>(d->dev_div,d->res,invCellSize);

		cudaMemset(d->dev_pressure,0, sizeof(float) * d->res.x * d->res.y);

		cudaMemcpyToArray( divArray, 0, 0, d->dev_div, d->res.x*d->res.y*sizeof(float), cudaMemcpyDeviceToDevice);
		for (int i=0; i<d->jacIter; i++) {
			cudaMemcpyToArray( pressureArray, 0, 0, d->dev_pressure, d->res.x*d->res.y*sizeof(float), cudaMemcpyDeviceToDevice);
			jacobi<<<blocks,threads>>>(d->dev_pressure, alpha, rBeta,d->res);
		}


		cudaMemcpyToArray( velArray, 0, 0, d->dev_vel, d->res.x*d->res.y*sizeof(float2), cudaMemcpyDeviceToDevice);
		cudaMemcpyToArray( pressureArray, 0, 0, d->dev_pressure, d->res.x*d->res.y*sizeof(float), cudaMemcpyDeviceToDevice);
		projection<<<blocks,threads>>>(d->dev_vel,d->res,invCellSize);
		

	}

	d->f++;
}

extern "C" void initFluid(VHFluidSolver* data, int dimX, int dimY) {

	data->res.x = dimX;
	data->res.y = dimY;

	int width = data->res.x;
	int height = data->res.y;


	if (data->colOutput==1) {
		//printf("tada");
		HANDLE_ERROR( cudaMalloc( (void**)&data->output_display,sizeof(float4)*width*height ) );
	}

	descFloat = cudaCreateChannelDesc<float>();
	descFloat2 = cudaCreateChannelDesc<float2>();
	descFloat4 = cudaCreateChannelDesc<float4>();


	HANDLE_ERROR( cudaMalloc( (void**)&data->dev_noise,sizeof(float)*width*height ) );
	texNoise.filterMode = cudaFilterModeLinear;
	HANDLE_ERROR(cudaMallocArray(&noiseArray, &descFloat, width, height));
	HANDLE_ERROR(cudaBindTextureToArray(texNoise, noiseArray, descFloat));

    HANDLE_ERROR( cudaMalloc( (void**)&data->dev_vel,sizeof(float2)*width*height ) );
	texVel.filterMode = cudaFilterModeLinear;
	HANDLE_ERROR(cudaMallocArray(&velArray, &descFloat2, width, height));
	HANDLE_ERROR(cudaBindTextureToArray(texVel, velArray, descFloat2));

	HANDLE_ERROR( cudaMalloc( (void**)&data->dev_dens,sizeof(float)*width*height ) );
	texDens.filterMode = cudaFilterModeLinear;
	HANDLE_ERROR(cudaMallocArray(&densArray, &descFloat, width, height));
	HANDLE_ERROR(cudaBindTextureToArray(texDens, densArray, descFloat));

	HANDLE_ERROR( cudaMalloc( (void**)&data->dev_pressure,sizeof(float)*width*height ) );
	texPressure.filterMode = cudaFilterModeLinear;
	HANDLE_ERROR(cudaMallocArray(&pressureArray, &descFloat, width, height));
	HANDLE_ERROR(cudaBindTextureToArray(texPressure, pressureArray, descFloat));

	HANDLE_ERROR( cudaMalloc( (void**)&data->dev_div,sizeof(float)*width*height ) );
	texDiv.filterMode = cudaFilterModeLinear;
	HANDLE_ERROR(cudaMallocArray(&divArray, &descFloat, width, height));
	HANDLE_ERROR(cudaBindTextureToArray(texDiv, divArray, descFloat));

	HANDLE_ERROR( cudaMalloc( (void**)&data->dev_vort,sizeof(float)*width*height ) );
	texVort.filterMode = cudaFilterModeLinear;
	HANDLE_ERROR(cudaMallocArray(&vortArray, &descFloat, width, height));
	HANDLE_ERROR(cudaBindTextureToArray(texVort, vortArray, descFloat));

	HANDLE_ERROR( cudaMalloc( (void**)&data->dev_obstacles,sizeof(float4)*width*height ) );
	texObstacles.filterMode = cudaFilterModeLinear;
	HANDLE_ERROR(cudaMallocArray(&obstArray, &descFloat4, width, height));
	HANDLE_ERROR(cudaBindTextureToArray(texObstacles, obstArray, descFloat4));


}

// clean up memory allocated on the GPU
extern "C" void clearFluid(VHFluidSolver *d ) {

	if (d->res.x != -1) {

		cudaUnbindTexture( texNoise );
		HANDLE_ERROR( cudaFree( d->dev_noise ) );
		HANDLE_ERROR(cudaFreeArray(noiseArray));

		cudaUnbindTexture( texVel );
		HANDLE_ERROR( cudaFree( d->dev_vel ) );
		HANDLE_ERROR(cudaFreeArray(velArray));

		cudaUnbindTexture( texDens );
		HANDLE_ERROR( cudaFree( d->dev_dens ) );
		HANDLE_ERROR(cudaFreeArray(densArray));

		cudaUnbindTexture( texPressure );
		HANDLE_ERROR( cudaFree( d->dev_pressure ) );
		HANDLE_ERROR(cudaFreeArray(pressureArray));

		cudaUnbindTexture( texDiv );
		HANDLE_ERROR( cudaFree( d->dev_div ) );
		HANDLE_ERROR(cudaFreeArray(divArray));


		cudaUnbindTexture( texVort );
		HANDLE_ERROR( cudaFree( d->dev_vort ) );
		HANDLE_ERROR(cudaFreeArray(vortArray));

		cudaUnbindTexture( texObstacles );
		HANDLE_ERROR( cudaFree( d->dev_obstacles ) );
		HANDLE_ERROR(cudaFreeArray(obstArray));

		if (d_perm) {
			HANDLE_ERROR( cudaFree( d_perm ) );
			d_perm = NULL;
		}

		if (d->colOutput==1)
			HANDLE_ERROR( cudaFree( d->output_display ) );
	}

}

extern "C" void resetFluid(VHFluidSolver *d) {

	dim3    blocks(d->res.x/16 + (!(d->res.x%16)?0:1),d->res.y/16 + (!(d->res.y%16)?0:1));
    dim3    threads(16,16);

	cudaMemset(d->dev_vel,0, sizeof(float2) * d->res.x * d->res.y);
	cudaMemset(d->dev_dens,0, sizeof(float) * d->res.x * d->res.y);
	cudaMemset(d->dev_pressure,0, sizeof(float) * d->res.x * d->res.y);
	cudaMemset(d->dev_div,0, sizeof(float) * d->res.x * d->res.y);
	cudaMemset(d->dev_vort,0, sizeof(float) * d->res.x * d->res.y);
	cudaMemset(d->dev_noise,0, sizeof(float) * d->res.x * d->res.y);
	cudaMemset(d->dev_obstacles,0, sizeof(float4) * d->res.x * d->res.y);

	d->f = 0;

	createBorder<<<blocks,threads>>>(d->dev_obstacles,d->res,d->borderPosX,d->borderNegX,d->borderPosY,d->borderNegY);

}

extern "C" void renderFluid(VHFluidSolver *d, float4* d_output, int previewType, float maxBounds) {

	dim3    blocks(d->res.x/16 + (!(d->res.x%16)?0:1),d->res.y/16 + (!(d->res.y%16)?0:1));
    dim3    threads(16,16);

	//cudaMemcpyToArray( densArray, 0, 0, d->dev_dens, d->res.x*d->res.y*sizeof(float), cudaMemcpyDeviceToDevice);

	if(previewType == 0) {
		float_to_color<<<blocks,threads>>>(d_output, d->dev_dens,d->res,0,maxBounds );
	} else if(previewType == 1) {
		float2_to_color<<<blocks,threads>>>(d_output, d->dev_vel,d->res,-maxBounds,maxBounds );
	} else if(previewType == 2) {
		float_to_color<<<blocks,threads>>>(d_output, d->dev_noise,d->res,-maxBounds,maxBounds );
	} else if(previewType == 3) {
		float_to_color<<<blocks,threads>>>(d_output, d->dev_pressure,d->res,-maxBounds,maxBounds );
	} else if(previewType == 4) {
		float_to_color<<<blocks,threads>>>(d_output, d->dev_vort,d->res,-maxBounds,maxBounds );
	} else if(previewType == 5) {
		obst_to_color<<<blocks,threads>>>(d_output, d->dev_obstacles,d->res );
	}
	

}