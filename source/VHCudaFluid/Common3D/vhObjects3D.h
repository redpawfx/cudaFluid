#ifndef __VHOBJECTS_H__
#define __VHOBJECTS_H__


struct FluidEmitter {

	float posX;
	float posY;
	float posZ;

	float radius;
	float amount;
	
};

struct Collider {

	float posX;
	float posY;
	float posZ;

	float oldPosX;
	float oldPosY;
	float oldPosZ;

	float radius;
	
};

#endif 