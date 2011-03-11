#include "fluidNode3D.h"

MTypeId     fluidNode3D::id( 0x800008 );

MObject     fluidNode3D::aEmitters;
MObject     fluidNode3D::aColliders;

MObject     fluidNode3D::aInTime;
MObject     fluidNode3D::aMayaFluid;
MObject     fluidNode3D::aDensCopy;
MObject     fluidNode3D::aOutTime;

MObject		fluidNode3D::aStartFrame;
MObject		fluidNode3D::aSubsteps;
MObject		fluidNode3D::aJacIter;

MObject		fluidNode3D::aFluidSize;
MObject		fluidNode3D::aSizeX;
MObject		fluidNode3D::aSizeY;
MObject		fluidNode3D::aSizeZ;

MObject		fluidNode3D::aRes;
MObject		fluidNode3D::aResX;
MObject		fluidNode3D::aResY;
MObject		fluidNode3D::aResZ;

MObject		fluidNode3D::aBorderNegX;
MObject		fluidNode3D::aBorderPosX;
MObject		fluidNode3D::aBorderNegY;
MObject		fluidNode3D::aBorderPosY;
MObject		fluidNode3D::aBorderNegZ;
MObject		fluidNode3D::aBorderPosZ;


MObject		fluidNode3D::aDensDis;
MObject		fluidNode3D::aDensBuoyStrength;

MObject		fluidNode3D::aDensBuoyDir;
MObject		fluidNode3D::aDensBuoyDirX;
MObject		fluidNode3D::aDensBuoyDirY;
MObject		fluidNode3D::aDensBuoyDirZ;

MObject		fluidNode3D::aVelDamp;
MObject		fluidNode3D::aVortConf;

MObject		fluidNode3D::aNoiseStr;
MObject		fluidNode3D::aNoiseFreq;
MObject		fluidNode3D::aNoiseOct;
MObject		fluidNode3D::aNoiseLacun;
MObject		fluidNode3D::aNoiseSpeed;
MObject		fluidNode3D::aNoiseAmp;

MObject		fluidNode3D::aPreview;
MObject		fluidNode3D::aDrawCube;
MObject		fluidNode3D::aOpaScale;
MObject		fluidNode3D::aStepMul;
MObject		fluidNode3D::aDisplayRes;

MObject		fluidNode3D::aDoShadows;
MObject		fluidNode3D::aLightPosX;
MObject		fluidNode3D::aLightPosY;
MObject		fluidNode3D::aLightPosZ;
MObject		fluidNode3D::aLightPos;
MObject		fluidNode3D::aShadowDens;
MObject		fluidNode3D::aShadowStepMul;
MObject		fluidNode3D::aShadowThres;

MObject		fluidNode3D::aDisplaySlice;
MObject		fluidNode3D::aSliceType;
MObject		fluidNode3D::aSliceAxis;
MObject		fluidNode3D::aSlicePos;
MObject		fluidNode3D::aSliceBounds;



//
void* fluidNode3D::creator()
{
	return new fluidNode3D();
}

void fluidNode3D::drawWireCube(float x, float y, float z) {

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

void fluidNode3D::draw( M3dView & view, const MDagPath & path, 
						M3dView::DisplayStyle style,
						M3dView::DisplayStatus status ){



	view.beginGL(); 

	MObject thisNode = thisMObject();

	MPlug prevPlug( thisNode, aPreview );
	prevPlug.getValue( preview );

	MPlug displaySlicePlug( thisNode, aDisplaySlice);
	displaySlicePlug.getValue( displaySlice );


	if(preview) {

		if ( ( style == M3dView::kFlatShaded ) ||  ( style == M3dView::kGouraudShaded ) ) {  

				MPlug displayResPlug( thisNode, aDisplayRes );
				int displayResTemp;
				displayResPlug.getValue(displayResTemp);
				
				if (displayEnum != displayResTemp) {
					displayEnum = displayResTemp;

					switch(displayEnum) {

						case 0 :
							displayX = displayY = 128;
							break;
						case 1 :
							displayX = displayY = 256;
							break;
						case 2 :
							displayX = displayY = 512;
							break;
						case 3 :
							displayX = displayY = 768;
							break;
						case 4 :
							displayX = displayY = 1024;
							break;
					}

					initPixelBuffer(true);
				}

				MMatrix mayaModelView;
				view.modelViewMatrix(mayaModelView);

				MTransformationMatrix modelViewTrans(mayaModelView);

				MEulerRotation eul = modelViewTrans.eulerRotation();

				double rot[3];
				MTransformationMatrix::RotationOrder rotOrder;
				modelViewTrans.getRotation(rot, rotOrder);

				MAngle xRot(rot[0]);
				MAngle yRot(rot[1]);
				MAngle zRot(rot[2]);

				/*std::cout << "Deg...................." << std::endl;
				std::cout << "xRotDeg : " << xRot.asDegrees() << std::endl;
				std::cout << "xRotDeg : " << yRot.asDegrees() << std::endl;
				std::cout << "zRotDeg : " << zRot.asDegrees() << std::endl;
				std::cout << "Rad----------" << std::endl;
				std::cout << "xRotRad : " << xRot.asRadians() << std::endl;
				std::cout << "yRotRad : " << yRot.asRadians() << std::endl;
				std::cout << "zRotRad : " << zRot.asRadians() << std::endl;*/
				

				MVector mayaTrans = modelViewTrans.getTranslation(MSpace::kWorld);

				/*std::cout << "xTrans : " << mayaTrans.x << std::endl;
				std::cout << "yTrans : " << mayaTrans.y << std::endl;
				std::cout << "zTrans : " << mayaTrans.z << std::endl;*/

				glPushAttrib(GL_CURRENT_BIT|GL_VIEWPORT_BIT|GL_COLOR_BUFFER_BIT);

				GLfloat modelView[16];
				glMatrixMode(GL_MODELVIEW);
				glPushMatrix();
					glLoadIdentity();
					glRotatef(-xRot.asDegrees(), 1.0, 0.0, 0.0);
					glRotatef(-yRot.asDegrees(), 0.0, 1.0, 0.0);
					glRotatef(-zRot.asDegrees(), 0.0, 0.0, 1.0);
					glTranslatef(-mayaTrans.x, -mayaTrans.y, -mayaTrans.z);
				glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
				glPopMatrix();

				float invViewMatrix[12];

				invViewMatrix[0] = modelView[0]; invViewMatrix[1] = modelView[4]; invViewMatrix[2] = modelView[8]; invViewMatrix[3] = modelView[12];
				invViewMatrix[4] = modelView[1]; invViewMatrix[5] = modelView[5]; invViewMatrix[6] = modelView[9]; invViewMatrix[7] = modelView[13];
				invViewMatrix[8] = modelView[2]; invViewMatrix[9] = modelView[6]; invViewMatrix[10] = modelView[10]; invViewMatrix[11] = modelView[14];

				/*invViewMatrix[0] = modelView(0,0); invViewMatrix[1] = modelView(1,0); invViewMatrix[2] = modelView(2,0); invViewMatrix[3] = -modelView(3,0);
				invViewMatrix[4] = modelView(0,1); invViewMatrix[5] = modelView(1,1); invViewMatrix[6] = modelView(2,1); invViewMatrix[7] = -modelView(3,1);
				invViewMatrix[8] = modelView(0,2); invViewMatrix[9] = modelView(1,2); invViewMatrix[10] = modelView(2,2); invViewMatrix[11] = -modelView(3,2);*/
				

				copyInvViewMatrix(invViewMatrix, sizeof(cu::float4)*3);

				cu::cutilSafeCall(cu::cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
				cu::float4 *d_output;
				size_t num_bytes; 
				cu::cutilSafeCall(cu::cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,  cuda_pbo_resource));
				cu::cudaMemset(d_output, 0, displayX*displayY*sizeof(cu::float4));

				unsigned int vx, vy, vwidth, vheight;
				view.viewport(vx,vy,vwidth,vheight);

				float ratio = (float)vheight/(float)vwidth;

				MMatrix mayaProj;
				view.projectionMatrix(mayaProj);

				double focalLength = mayaProj(1,1);

				//std::cout << "focalLength : " << focalLength << std::endl;

				MPlug opaScalePlug( thisNode, aOpaScale );
				float opaScale;
				opaScalePlug.getValue(opaScale);

				MPlug stepMulPlug( thisNode, aStepMul );
				float stepMul;
				stepMulPlug.getValue(stepMul);

				MPlug doShadowsPlug( thisNode, aDoShadows );
				int doShadows;
				doShadowsPlug.getValue(doShadows);

				MPlug shadowDensPlug( thisNode, aShadowDens );
				float shadowDens;
				shadowDensPlug.getValue(shadowDens);

				MPlug shadowStepMulPlug( thisNode, aShadowStepMul );
				float shadowStepMul;
				shadowStepMulPlug.getValue(shadowStepMul);

				MPlug shadowThresPlug( thisNode, aShadowThres );
				float shadowThres;
				shadowThresPlug.getValue(shadowThres);

				MPlug lightPosXPlug( thisNode, aLightPosX );
				float lightPosX;
				lightPosXPlug.getValue(lightPosX);

				MPlug lightPosYPlug( thisNode, aLightPosY );
				float lightPosY;
				lightPosYPlug.getValue(lightPosY);

				MPlug lightPosZPlug( thisNode, aLightPosZ );
				float lightPosZ;
				lightPosZPlug.getValue(lightPosZ);

				MVector lightPos(lightPosX, lightPosY, lightPosZ);

				MFnTransform fnTransform;
				MFnDagNode fnDag;
				MDagPath path;

				fnDag.setObject(this->thisMObject());
				MObject fluidTransform = fnDag.parent(0);
				fnDag.setObject(fluidTransform);
				fnDag.getPath(path);
				fnTransform.setObject(path);
				MTransformationMatrix fluidMatrix = fnTransform.transformation();

				MTransformationMatrix lightMatrix = MTransformationMatrix();
				lightMatrix.setTranslation(lightPos, MSpace::kWorld);

				lightMatrix = lightMatrix.asMatrix()*fluidMatrix.asMatrixInverse();
				MVector pos = MTransformationMatrix(lightMatrix).getTranslation(MSpace::kWorld);


				fluidSolver.lightPos = cu::make_float3(pos.x,pos.y,pos.z);


				if (fluidInitialized)
					render_kernel(&fluidSolver, d_output, displayX, displayY, opaScale, -focalLength,
					doShadows, stepMul, shadowStepMul, shadowThres, shadowDens);

				cu::cutilSafeCall(cu::cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

				//glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

				glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
				glBindTexture(GL_TEXTURE_2D, gl_Tex);
				glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, displayX, displayY, GL_RGBA, GL_FLOAT, 0);
				glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

				glMatrixMode(GL_PROJECTION);
				glPushMatrix();
				glLoadIdentity();
				glOrtho(-1.0, 1.0, -1.0, 1.0, 0, 1.0);

				glMatrixMode(GL_MODELVIEW);
				glPushMatrix();
				glLoadIdentity();

				//glEnable(GL_BLEND);
				//glBlendFunc(GL_SRC_ALPHA, GL_ONE);
				glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

				glEnable(GL_TEXTURE_2D);

				glColor4f(1.0, 1.0, 1.0,0.5f);

				//glDepthMask( GL_FALSE );

				glDisable(GL_DEPTH_TEST);

				glBegin(GL_QUADS);
				glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f*ratio, -1.0f);
				glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f*ratio, -1.0f);
				glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f*ratio, 1.0f);
				glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f*ratio, 1.0f);
				glEnd();

				glEnable(GL_DEPTH_TEST);

				//glDepthMask( GL_TRUE );


				glDisable(GL_TEXTURE_2D);
				//glDisable(GL_BLEND);

				glBindTexture(GL_TEXTURE_2D, 0);

				glMatrixMode(GL_PROJECTION);
				glPopMatrix();

				glMatrixMode(GL_MODELVIEW);
				glPopMatrix();

				glPopAttrib();

				}			


			}

			
		if(displaySlice) {

			MPlug slicePosPlug( thisNode, aSlicePos );
			float slicePos;
			slicePosPlug.getValue(slicePos);

			MPlug sliceAxisPlug( thisNode, aSliceAxis );
			int sliceAxis;
			sliceAxisPlug.getValue(sliceAxis);

			float slicePos3D;

			MPlug resXPlug( thisNode, aResX );
			int resX;
			resXPlug.getValue(resX);

			MPlug resYPlug( thisNode, aResY );
			int resY;
			resYPlug.getValue(resY);

			MPlug resZPlug( thisNode, aResZ );
			int resZ;
			resZPlug.getValue(resZ);


			if (sliceAxis == 2) {
				slicePos3D = (slicePos-0.5)*fluidSolver.fluidSize.z;

				if (displaySliceX != resX || displaySliceY != resY) {
					displaySliceX = resX;
					displaySliceY = resY;
					initPixelBuffer(false);
				}

			} else if (sliceAxis == 0) {
				slicePos3D = (slicePos-0.5)*fluidSolver.fluidSize.x;

				if (displaySliceX != resZ || displaySliceY != resY) {
					displaySliceX = resZ;
					displaySliceY = resY;
					initPixelBuffer(false);
				}

			} else {
				slicePos3D = (slicePos-0.5)*fluidSolver.fluidSize.y;

				if (displaySliceX != resX || displaySliceY != resZ) {
					displaySliceX = resX;
					displaySliceY = resZ;
					initPixelBuffer(false);
				}
			}
			
			float sizeX = fluidSolver.fluidSize.x * 0.5;
			float sizeY = fluidSolver.fluidSize.y * 0.5;
			float sizeZ = fluidSolver.fluidSize.z * 0.5;


			if ( ( style == M3dView::kFlatShaded ) ||  ( style == M3dView::kGouraudShaded ) ) {  
				// Push the color settings
				// 
				glPushAttrib( GL_CURRENT_BIT );

				/*if ( status == M3dView::kActive ) {
					view.setDrawColor( 0, M3dView::kActiveColors );
				} else {
					view.setDrawColor( 0, M3dView::kDormantColors );
				} */ 

				cu::cutilSafeCall(cu::cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
				cu::float4 *d_output;
				size_t num_bytes; 
				cu::cutilSafeCall(cu::cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,  cuda_pbo_resource));
				cu::cudaMemset(d_output, 0, displaySliceX*displaySliceY*sizeof(cu::float4));

				MPlug sliceTypePlug( thisNode, aSliceType );
				int sliceType;
				sliceTypePlug.getValue(sliceType);

				MPlug sliceBoundsPlug( thisNode, aSliceBounds );
				float sliceBounds;
				sliceBoundsPlug.getValue(sliceBounds);

				if (fluidInitialized)
					renderFluidSlice(&fluidSolver,d_output,slicePos,sliceType,sliceAxis,sliceBounds);

				cu::cutilSafeCall(cu::cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

				//glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

				glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
				glBindTexture(GL_TEXTURE_2D, gl_SliceTex);
				glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, displaySliceX, displaySliceY, GL_RGBA, GL_FLOAT, 0);
				glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

				glEnable(GL_TEXTURE_2D);

				glColor3f(1.0, 1.0, 1.0);

				if (sliceAxis == 2) {

					glBegin( GL_QUADS );
					glTexCoord2f(0.0f, 0.0f); glVertex3f(-sizeX,-sizeY,slicePos3D);
					glTexCoord2f(0.0f, 1.0f); glVertex3f(-sizeX,sizeY,slicePos3D);
					glTexCoord2f(1.0f, 1.0f); glVertex3f(sizeX,sizeY,slicePos3D);
					glTexCoord2f(1.0f, 0.0f); glVertex3f(sizeX,-sizeY,slicePos3D);
					glEnd();

				} else if (sliceAxis == 0) {

					glBegin( GL_QUADS );
					glTexCoord2f(0.0f, 0.0f); glVertex3f(slicePos3D,-sizeY,-sizeZ);
					glTexCoord2f(0.0f, 1.0f); glVertex3f(slicePos3D,sizeY,-sizeZ);
					glTexCoord2f(1.0f, 1.0f); glVertex3f(slicePos3D,sizeY,sizeZ);
					glTexCoord2f(1.0f, 0.0f); glVertex3f(slicePos3D,-sizeY,sizeZ);
					glEnd();
				}

				else {

					glBegin( GL_QUADS );
					glTexCoord2f(0.0f, 0.0f); glVertex3f(-sizeX,slicePos3D,-sizeZ);
					glTexCoord2f(0.0f, 1.0f); glVertex3f(-sizeX,slicePos3D,sizeZ);
					glTexCoord2f(1.0f, 1.0f); glVertex3f(sizeX,slicePos3D,sizeZ);
					glTexCoord2f(1.0f, 0.0f); glVertex3f(sizeX,slicePos3D,-sizeZ);
					glEnd();
				}


				glDisable(GL_TEXTURE_2D);

				glBindTexture(GL_TEXTURE_2D, 0);

				glPopAttrib();

			}

		if (sliceAxis == 2) {

			glBegin( GL_LINE_STRIP );
				glVertex3f(-sizeX,-sizeY,slicePos3D);
				glVertex3f(-sizeX,sizeY,slicePos3D);
				glVertex3f(sizeX,sizeY,slicePos3D);
				glVertex3f(sizeX,-sizeY,slicePos3D);
				glVertex3f(-sizeX,-sizeY,slicePos3D);
			glEnd();

		} else if (sliceAxis == 0) {

			glBegin( GL_LINE_STRIP );
				glVertex3f(slicePos3D,-sizeY,-sizeZ);
				glVertex3f(slicePos3D,sizeY,-sizeZ);
				glVertex3f(slicePos3D,sizeY,sizeZ);
				glVertex3f(slicePos3D,-sizeY,sizeZ);
				glVertex3f(slicePos3D,-sizeY,-sizeZ);
			glEnd();

		} else {

			glBegin( GL_LINE_STRIP );
				glVertex3f(-sizeX,slicePos3D,-sizeZ);
				glVertex3f(-sizeX,slicePos3D,sizeZ);
				glVertex3f(sizeX,slicePos3D,sizeZ);
				glVertex3f(sizeX,slicePos3D,-sizeZ);
				glVertex3f(-sizeX,slicePos3D,-sizeZ);
			glEnd();

		}




		}

	MPlug drawCubePlug( thisNode, aDrawCube );
	int drawCube;
	drawCubePlug.getValue(drawCube);

	if (drawCube)  
		drawWireCube(fluidSolver.fluidSize.x*0.5,fluidSolver.fluidSize.y*0.5,fluidSolver.fluidSize.z*0.5);

	view.endGL();


}

//
MStatus fluidNode3D::initialize()
{

	MFnUnitAttribute	uAttr;
	MFnNumericAttribute nAttr;
	MFnTypedAttribute	tAttr;
	MFnEnumAttribute	eAttr;
	MFnMessageAttribute	mAttr;
	MStatus				stat;

	aEmitters = mAttr.create("emitters","ems",&stat);
	CHECK_MSTATUS(stat);
	mAttr.setArray(true);
	stat = addAttribute(aEmitters);

	aColliders = mAttr.create("colliders","cols",&stat);
	CHECK_MSTATUS(stat);
	mAttr.setArray(true);
	stat = addAttribute(aColliders);

	aInTime =  uAttr.create( "inTime", "t", MFnUnitAttribute::kTime, 0.0, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aInTime);

	aMayaFluid = mAttr.create("mayaFluid", "mf", &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aMayaFluid);

	aOutTime =  uAttr.create( "outTime", "ot", MFnUnitAttribute::kTime, 0.0, &stat);
	CHECK_MSTATUS(stat);
	uAttr.setWritable(false);
	stat = addAttribute(aOutTime);

	aStartFrame =  nAttr.create( "startFrame", "sf", MFnNumericData::kInt, 1, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aStartFrame);

	aSubsteps = nAttr.create("substeps", "step", MFnNumericData::kInt, 1, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aSubsteps);

	aJacIter = nAttr.create("jacIter", "ji", MFnNumericData::kInt, 30, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aJacIter);

	//maya doesn't like sx sy

	aSizeX = nAttr.create("sizeX", "fsx", MFnNumericData::kFloat, 10.0, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aSizeX);

	aSizeY = nAttr.create("sizeY", "fsy", MFnNumericData::kFloat, 10.0, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aSizeY);

	aSizeZ = nAttr.create("sizeZ", "fsz", MFnNumericData::kFloat, 10.0, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aSizeZ);

	aFluidSize = nAttr.create("fluidSize", "fs", aSizeX, aSizeY, aSizeZ, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aFluidSize);

	//maya doesn't like rx ry

	aResX = nAttr.create("resX", "rsX", MFnNumericData::kInt, 50, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aResX);

	aResY = nAttr.create("resY", "rsy", MFnNumericData::kInt, 50, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aResY);

	aResZ = nAttr.create("resZ", "rsz", MFnNumericData::kInt, 50, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aResZ);

	aRes = nAttr.create("res", "res", aResX, aResY, aResZ, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	nAttr.setDefault(50,50,50);
	stat = addAttribute(aRes);

	aBorderNegX = nAttr.create("borderNegX", "bnX", MFnNumericData::kBoolean, true, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aBorderNegX);

	aBorderPosX = nAttr.create("borderPosX", "bpX", MFnNumericData::kBoolean, true, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aBorderPosX);

	aBorderNegY = nAttr.create("borderNegY", "bnY", MFnNumericData::kBoolean, true, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aBorderNegY);

	aBorderPosY = nAttr.create("borderPosY", "bpY", MFnNumericData::kBoolean, true, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aBorderPosY);

	aBorderNegZ = nAttr.create("borderNegZ", "bnZ", MFnNumericData::kBoolean, true, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aBorderNegZ);

	aBorderPosZ = nAttr.create("borderPosZ", "bpZ", MFnNumericData::kBoolean, true, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aBorderPosZ);

	aDensDis = nAttr.create("densDis", "dd", MFnNumericData::kFloat, 0.0, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aDensDis);

	aDensBuoyStrength = nAttr.create("densBuoyStr", "dbs", MFnNumericData::kFloat, 1.0, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aDensBuoyStrength);

	aDensBuoyDirX = nAttr.create("densBuoyDirX", "dbdx", MFnNumericData::kFloat, 0.0, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aDensBuoyDirX);
	
	aDensBuoyDirY = nAttr.create("densBuoyDirY", "dbdy", MFnNumericData::kFloat, 1.0, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aDensBuoyDirY);

	aDensBuoyDirZ = nAttr.create("densBuoyDirZ", "dbdz", MFnNumericData::kFloat, 0.0, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aDensBuoyDirZ);


	aDensBuoyDir = nAttr.create("densBuoyDir", "dbd", aDensBuoyDirX, aDensBuoyDirY, aDensBuoyDirZ, &stat);
	//aDensBuoyDir = nAttr.create("densBuoyDir", "dbd", aDensBuoyDirX, aDensBuoyDirY, MObject::kNullObj, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	nAttr.setDefault(0.0,1.0,0.0);
	stat = addAttribute(aDensBuoyDir);

	aVelDamp = nAttr.create("velDamp", "vd", MFnNumericData::kFloat, 0.0, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aVelDamp);

	aVortConf = nAttr.create("vortConf", "vc", MFnNumericData::kFloat, 2.0, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aVortConf);

	aNoiseStr = nAttr.create("noiseStr", "nst", MFnNumericData::kFloat, 0.0, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aNoiseStr);

	aNoiseFreq = nAttr.create("noiseFreq", "nfr", MFnNumericData::kFloat, 1.0, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aNoiseFreq);

	aNoiseOct = nAttr.create("noiseOct", "noc", MFnNumericData::kInt, 3.0, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aNoiseOct);

	aNoiseLacun = nAttr.create("noiseLacun", "nlc", MFnNumericData::kFloat, 4.0, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aNoiseLacun);

	aNoiseSpeed = nAttr.create("noiseSpeed", "nsp", MFnNumericData::kFloat, 0.01, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aNoiseSpeed);

	aNoiseAmp = nAttr.create("noiseAmp", "nam", MFnNumericData::kFloat, 0.5, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	stat = addAttribute(aNoiseAmp);

	aPreview = nAttr.create("preview", "prv", MFnNumericData::kBoolean, 1, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aPreview);

	aDrawCube = nAttr.create("drawCube", "drc", MFnNumericData::kBoolean, 1, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aDrawCube);

	aOpaScale = nAttr.create("opaScale", "opa", MFnNumericData::kFloat, 0.1, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setMin(0.0);
	nAttr.setSoftMax(1.0);
	stat = addAttribute(aOpaScale);

	aStepMul = nAttr.create("stepMul", "smul", MFnNumericData::kFloat, 1.0, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setMin(1.0);
	stat = addAttribute(aStepMul);

	aDisplayRes = eAttr.create("displayRes", "dres", 1, &stat);
	CHECK_MSTATUS(stat);
	eAttr.addField("128",0);
	eAttr.addField("256",1);
	eAttr.addField("512",2);
	eAttr.addField("768",3);
	eAttr.addField("1024",4);
	stat = addAttribute(aDisplayRes);

	aDoShadows = nAttr.create("doShadows", "dsh", MFnNumericData::kBoolean, 0, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aDoShadows);

	aLightPosX = nAttr.create("lightPosX", "lipx", MFnNumericData::kFloat, 10.0, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aLightPosX);

	aLightPosY = nAttr.create("lightPosY", "lipy", MFnNumericData::kFloat, 10.0, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aLightPosY);

	aLightPosZ = nAttr.create("lightPosZ", "lipz", MFnNumericData::kFloat, 0.0, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aLightPosZ);

	aLightPos = nAttr.create("lightPos", "lipos", aLightPosX, aLightPosY, aLightPosZ, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setKeyable(true);
	nAttr.setDefault(10.0,10.0,0.0);
	stat = addAttribute(aLightPos);

	aShadowDens = nAttr.create("shadowDens", "shd", MFnNumericData::kFloat, 0.9, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setMin(0.0);
	nAttr.setSoftMax(1.0);
	stat = addAttribute(aShadowDens);

	aShadowStepMul = nAttr.create("shadowStepMul", "ssm", MFnNumericData::kFloat, 2.0, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setMin(1.0);
	stat = addAttribute(aShadowStepMul);

	aShadowThres = nAttr.create("shadowThres", "sht", MFnNumericData::kFloat, 0.9, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setMin(0.0);
	nAttr.setMax(1.0);
	stat = addAttribute(aShadowThres);

	aDisplaySlice = nAttr.create("displaySlice", "disl", MFnNumericData::kBoolean, 0, &stat);
	CHECK_MSTATUS(stat);
	stat = addAttribute(aDisplaySlice);


	aSliceType = eAttr.create("sliceType", "sty", 0, &stat);
	CHECK_MSTATUS(stat);
	eAttr.addField("Density",0);
	eAttr.addField("Velocity",1);
	eAttr.addField("Noise",2);
	eAttr.addField("Pressure",3);
	eAttr.addField("Vorticity",4);
	eAttr.addField("Obstacles",5);
	stat = addAttribute(aSliceType);

	aSliceAxis = eAttr.create("sliceAxis", "sax", 2, &stat);
	CHECK_MSTATUS(stat);
	eAttr.addField("X",0);
	eAttr.addField("Y",1);
	eAttr.addField("Z",2);
	stat = addAttribute(aSliceAxis);

	aSlicePos = nAttr.create("slicePos", "spo", MFnNumericData::kFloat, 0.5, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setMin(0);
	nAttr.setMax(1);
	nAttr.setKeyable(true);
	stat = addAttribute(aSlicePos);

	aSliceBounds = nAttr.create("sliceBounds", "sbo", MFnNumericData::kFloat, 1.0, &stat);
	CHECK_MSTATUS(stat);
	nAttr.setMin(0);
	//nAttr.setMax(1);
	nAttr.setKeyable(true);
	stat = addAttribute(aSliceBounds);


	stat = attributeAffects(aInTime, aOutTime);

	return MS::kSuccess;
} 

void fluidNode3D::initPixelBuffer(bool initpbo){

	if (initpbo) {

	 if (pbo) {
		// unregister this buffer object from CUDA C
		cu::cutilSafeCall(cu::cudaGraphicsUnregisterResource(cuda_pbo_resource));

		// delete old buffer
        glDeleteBuffersARB(1, &pbo);
        glDeleteTextures(1, &gl_Tex);
		glDeleteTextures(1, &gl_SliceTex);
    }

    // create pixel buffer object for display
    glGenBuffersARB(1, &pbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, displayX*displayY*sizeof(cu::float4), 0, GL_STREAM_DRAW_ARB);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // register this buffer object with CUDA
	cu::cutilSafeCall(cu::cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cu::cudaGraphicsMapFlagsWriteDiscard));	

	// create texture for display
	glGenTextures(1, &gl_Tex);
	glBindTexture(GL_TEXTURE_2D, gl_Tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, displayX, displayY, 0, GL_RGBA, GL_FLOAT,  NULL);
	glBindTexture(GL_TEXTURE_2D, 0);

	}

	if (pbo) {
		glDeleteTextures(1, &gl_SliceTex);

	}

	// create texture for display
	glGenTextures(1, &gl_SliceTex);
	glBindTexture(GL_TEXTURE_2D, gl_SliceTex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, displaySliceX, displaySliceY, 0, GL_RGBA, GL_FLOAT,  NULL);
	glBindTexture(GL_TEXTURE_2D, 0);

}

fluidNode3D::fluidNode3D() {

	currentMayaFluidName = "";
	solverTime = 0.0;

	fluidSolver.colOutput = 0;
	
	fluidSolver.res = cu::make_cudaExtent(-1,-1,-1);

	fluidSolver.emitters = new FluidEmitter[1];
	fluidSolver.colliders = new Collider[1];

	fluidSolver.nEmit = 0;
	fluidSolver.nColliders = 0;

	preview = false;
	displaySlice = false;

	glewInit();
	cu::cutilSafeCall(cu::cudaGLSetGLDevice( cu::cutGetMaxGflopsDeviceId() ));
	displaySliceX = displaySliceY = 50;
	displayX = displayY = 256;
	pbo = 0;
	initPixelBuffer(true);

	displayEnum = 1;
	fluidInitialized = false;

}


fluidNode3D::~fluidNode3D() {

	clear3DFluid(&fluidSolver);

	delete fluidSolver.emitters;
	delete fluidSolver.colliders;

	cudaGraphicsUnregisterResource(cuda_pbo_resource);

	cu::cutilSafeCall(cu::cudaThreadExit());

	glDeleteBuffersARB(1, &pbo);
	glDeleteTextures(1, &gl_Tex);
	glDeleteTextures(1, &gl_SliceTex);
}

void fluidNode3D::changeFluidRes(int x, int y, int z) {


	clear3DFluid(&fluidSolver);
	init3DFluid(&fluidSolver,x,y,z);

	size_t free, total;

	cu::cudaMemGetInfo(&free, &total);
        
     //printf("mem = %lu %lu\n", free, total);
	std::cout << "Free Mem " << free << std::endl;
	
	fluidInitialized = true;

}


MStatus fluidNode3D::compute (const MPlug& plug, MDataBlock& data) {

	MStatus returnStatus;

	 if(plug == aOutTime) {

		MDataHandle inTimeHandle = data.inputValue (aInTime, &returnStatus);
		CHECK_MSTATUS( returnStatus );

		MTime currentTime(inTimeHandle.asTime());
		currentTime.setUnit(MTime::uiUnit());

		int currentFrame = (int)currentTime.value();

		MDataHandle startFrameHandle = data.inputValue (aStartFrame, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		int startFrame = startFrameHandle.asInt();

		MFnTransform fnTransform;
		MFnDagNode fnDag;
		MDagPath path;
		MPlugArray emittersArray;

		fnDag.setObject(this->thisMObject());
		MObject fluidTransform = fnDag.parent(0);
		fnDag.setObject(fluidTransform);
		fnDag.getPath(path);
		fnTransform.setObject(path);
		//MVector fluidPos = fnTransform.getTranslation(MSpace::kWorld);
		MTransformationMatrix fluidMatrix = fnTransform.transformation();
		
		MPlug emittersPlug(this->thisMObject(),aEmitters);
		int nPlugs = emittersPlug.numElements();
		int conPlugs = emittersPlug.numConnectedElements();

		if (fluidSolver.nEmit != conPlugs){
			fluidSolver.nEmit = conPlugs;
			delete fluidSolver.emitters;
			fluidSolver.emitters = new FluidEmitter[conPlugs];
		}

		int k = 0;

		for ( unsigned int j=0; j<nPlugs; j++ ) {

				bool connected = emittersPlug[j].isConnected();

				if(connected) {
		
					emittersPlug[j].connectedTo(emittersArray, true, false);

					MObject emitter = emittersArray[0].node();

					fnDag.setObject(emitter);
					//std::cout << "Emitter " << j << " : "<< fnDag.name() << std::endl;
					fnDag.getPath(path);

					fnTransform.setObject(path);
			
					//MVector pos = fnTransform.getTranslation(MSpace::kWorld);
					//std::cout << "Emitter " << j << " : "<< pos.y << std::endl;

					MTransformationMatrix emitterMatrix = fnTransform.transformation();
					emitterMatrix = emitterMatrix.asMatrix()*fluidMatrix.asMatrixInverse();
					MVector pos = MTransformationMatrix(emitterMatrix).getTranslation(MSpace::kWorld);

					fluidSolver.emitters[k].posX = pos.x;
					fluidSolver.emitters[k].posY = pos.y;
					fluidSolver.emitters[k].posZ = pos.z;

					MPlug densEmitPlug = fnDag.findPlug("fluidDensityEmission",false);
					//double densEmit = densEmitPlug.asDouble();
					fluidSolver.emitters[k].amount = densEmitPlug.asDouble();

					MPlug distEmitPlug = fnDag.findPlug("maxDistance",false);
					//double distEmit = densEmitPlug.asDouble();
					fluidSolver.emitters[k].radius = distEmitPlug.asDouble();
		
					//std::cout << "Emitter " << j << " : "<< densEmit << std::endl;
					k++;
				}
		}

		MPlugArray collidersArray;

		MPlug collidersPlug(this->thisMObject(),aColliders);

		nPlugs = collidersPlug.numElements();
		conPlugs = collidersPlug.numConnectedElements();

		if (fluidSolver.nColliders != conPlugs){
			fluidSolver.nColliders = conPlugs;
			delete fluidSolver.colliders;
			fluidSolver.colliders = new Collider[conPlugs];
		}

		k=0;

		for ( unsigned int j=0; j<nPlugs; j++ ) {

			bool connected = collidersPlug[j].isConnected();

				if(connected) {
		
					collidersPlug[j].connectedTo(collidersArray, true, false);

					MObject collider = collidersArray[0].node();

					fnDag.setObject(collider);
					fnDag.getPath(path);

					fnTransform.setObject(path);

					MTransformationMatrix colliderMatrix = fnTransform.transformation();
					colliderMatrix = colliderMatrix.asMatrix()*fluidMatrix.asMatrixInverse();
					MVector pos = MTransformationMatrix(colliderMatrix).getTranslation(MSpace::kWorld);
			
					//MVector pos = fnTransform.getTranslation(MSpace::kWorld);
					//std::cout << "Emitter " << j << " : "<< pos.y << std::endl;

					if (currentFrame > startFrame) {
						fluidSolver.colliders[k].oldPosX = fluidSolver.colliders[k].posX;
						fluidSolver.colliders[k].oldPosY = fluidSolver.colliders[k].posY;
						fluidSolver.colliders[k].oldPosZ = fluidSolver.colliders[k].posZ;
					} else {
						fluidSolver.colliders[k].oldPosX = pos.x;
						fluidSolver.colliders[k].oldPosY = pos.y;
						fluidSolver.colliders[k].oldPosZ = pos.z;
					}

					fluidSolver.colliders[k].posX = pos.x;
					fluidSolver.colliders[k].posY = pos.y;
					fluidSolver.colliders[k].posZ = pos.z;

					MPlug radiusPlug = fnDag.findPlug("radius",false);
					fluidSolver.colliders[k].radius = radiusPlug.asDouble();

					k++;
				}
		}

		MDataHandle resXHandle = data.inputValue (aResX, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		int newResX = resXHandle.asInt();

		MDataHandle resYHandle = data.inputValue (aResY, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		int newResY = resYHandle.asInt();

		MDataHandle resZHandle = data.inputValue (aResZ, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		int newResZ = resZHandle.asInt();

		if (newResX != fluidSolver.res.width || newResY != fluidSolver.res.height || newResZ != fluidSolver.res.depth) {
			changeFluidRes(newResX, newResY, newResZ);
		}

		MDataHandle inBorderNegXHandle = data.inputValue (aBorderNegX, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		fluidSolver.borderNegX = inBorderNegXHandle.asBool();

		MDataHandle inBorderPosXHandle = data.inputValue (aBorderPosX, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		fluidSolver.borderPosX = inBorderPosXHandle.asBool();

		MDataHandle inBorderNegYHandle = data.inputValue (aBorderNegY, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		fluidSolver.borderNegY = inBorderNegYHandle.asBool();

		MDataHandle inBorderPosYHandle = data.inputValue (aBorderPosY, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		fluidSolver.borderPosY = inBorderPosYHandle.asBool();

		MDataHandle inBorderNegZHandle = data.inputValue (aBorderNegZ, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		fluidSolver.borderNegZ = inBorderNegZHandle.asBool();

		MDataHandle inBorderPosZHandle = data.inputValue (aBorderPosZ, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		fluidSolver.borderPosZ = inBorderPosZHandle.asBool();

		if(MTime::uiUnit() == MTime::kFilm) {
			fluidSolver.fps = 24;
		} else if(MTime::uiUnit() == MTime::kPALFrame) {
			fluidSolver.fps = 25;
		} else if(MTime::uiUnit() == MTime::kNTSCFrame) {
			fluidSolver.fps = 30;
		} else if(MTime::uiUnit() == MTime::kNTSCField) {
			fluidSolver.fps = 60;
		}

		MDataHandle sizeXHandle = data.inputValue (aSizeX, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		fluidSolver.fluidSize.x = sizeXHandle.asFloat();

		MDataHandle sizeYHandle = data.inputValue (aSizeY, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		fluidSolver.fluidSize.y = sizeYHandle.asFloat();

		MDataHandle sizeZHandle = data.inputValue (aSizeZ, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		fluidSolver.fluidSize.z = sizeZHandle.asFloat();


		MPlug mayaFluidPlug(this->thisMObject(),aMayaFluid);

		bool fluidConnected = mayaFluidPlug.isConnected();
		float* fluidDens;

		if (fluidConnected) {
			MPlugArray fluidArray;
			mayaFluidPlug.connectedTo( fluidArray, true, false );
			MObject mayaFluidObject = fluidArray[0].node();
			fluidFn.setObject(mayaFluidObject);
			fluidDens = fluidFn.density();
		}

		MDataHandle inPreviewHandle = data.inputValue (aPreview, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		preview = inPreviewHandle.asBool();

		MDataHandle inDisplaySliceHandle = data.inputValue (aDisplaySlice, &returnStatus);
		CHECK_MSTATUS( returnStatus );
		displaySlice = inDisplaySliceHandle.asBool();


		if (currentFrame <= startFrame) {

			reset3DFluid(&fluidSolver);

			if (preview == false && displaySlice == false) {
				if (fluidConnected)
					memset(fluidDens,0,fluidSolver.domainSize());
					//cu::cudaMemcpy( fluidDens, fluidSolver.dev_dens, fluidSolver.domainSize(), cu::cudaMemcpyDeviceToHost );
			}
		} else {

			MDataHandle substepsHandle = data.inputValue (aSubsteps, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			fluidSolver.substeps = substepsHandle.asInt();

			MDataHandle jacIterHandle = data.inputValue (aJacIter, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			fluidSolver.jacIter = jacIterHandle.asInt();

			MDataHandle densDisHandle = data.inputValue (aDensDis, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			fluidSolver.densDis = densDisHandle.asFloat();

			MDataHandle densBuoyStrHandle = data.inputValue (aDensBuoyStrength, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			fluidSolver.densBuoyStrength = densBuoyStrHandle.asFloat();

			MFloatVector newDir;

			MDataHandle densBuoyDirXHandle = data.inputValue (aDensBuoyDirX, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			newDir.x = densBuoyDirXHandle.asFloat();

			MDataHandle densBuoyDirYHandle = data.inputValue (aDensBuoyDirY, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			newDir.y = densBuoyDirYHandle.asFloat();

			MDataHandle densBuoyDirZHandle = data.inputValue (aDensBuoyDirZ, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			newDir.z = densBuoyDirZHandle.asFloat();

			newDir.normalize();
			fluidSolver.densBuoyDir = cu::make_float3(newDir.x,newDir.y,newDir.z);

			MDataHandle velDampHandle = data.inputValue (aVelDamp, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			fluidSolver.velDamp = velDampHandle.asFloat();

			MDataHandle vortConfHandle = data.inputValue (aVortConf, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			fluidSolver.vortConf = vortConfHandle.asFloat();

			MDataHandle noiseStrHandle = data.inputValue (aNoiseStr, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			fluidSolver.noiseStr = noiseStrHandle.asFloat();

			MDataHandle noiseFreqHandle = data.inputValue (aNoiseFreq, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			fluidSolver.noiseFreq = noiseFreqHandle.asFloat();

			MDataHandle noiseOctHandle = data.inputValue (aNoiseOct, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			fluidSolver.noiseOct = noiseOctHandle.asInt();

			MDataHandle noiseLacunHandle = data.inputValue (aNoiseLacun, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			fluidSolver.noiseLacun = noiseLacunHandle.asFloat();

			MDataHandle noiseSpeedHandle = data.inputValue (aNoiseSpeed, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			fluidSolver.noiseSpeed = noiseSpeedHandle.asFloat();

			MDataHandle noiseAmpHandle = data.inputValue (aNoiseAmp, &returnStatus);
			CHECK_MSTATUS( returnStatus );
			fluidSolver.noiseAmp = noiseAmpHandle.asFloat();

			//float *myVelX, *myVelY, *myVelZ;
			//fluidFn.getVelocity(myVelX, myVelY, myVelZ);

			if (currentFrame == (startFrame+1)) {

				solverTime = 0;
				reset3DFluid(&fluidSolver);
			}

			solve3DFluid(&fluidSolver);

			if (preview == false && displaySlice == false) {

				if (fluidConnected)
					cu::cudaMemcpy( fluidDens, fluidSolver.dev_dens, fluidSolver.domainSize(), cu::cudaMemcpyDeviceToHost );
				}

			//solverTime +=1;


			MDataHandle outTimeHandle = data.outputValue (aOutTime, &returnStatus);
			CHECK_MSTATUS(returnStatus);

			outTimeHandle.set(currentTime);

		}
		
		if(fluidConnected)
			fluidFn.updateGrid();

	 } else {

		return MS::kUnknownParameter;
	}

	return MS::kSuccess;

}


MStatus fluidNode3D::updateFluidName(MString newFluidName) {

	MObject fluidObject;
	MSelectionList list;

		//std::cout << "FluidString :" << newFluid << std::endl;
		//std::cout << "OldFluidString :" << currentFluid << std::endl;
	
	if (newFluidName != currentMayaFluidName) {

		std::cout << "FluidString :" << newFluidName << std::endl;

		MGlobal::getSelectionListByName(newFluidName,list);

		//std::cout << "listlength:"  << list.length() << std::endl;

		if (list.length() == 1) {
			list.getDependNode(0,fluidObject);
			fluidFn.setObject(fluidObject);

			currentMayaFluidName = newFluidName;

		/*	fluidFn.getResolution(mResX, mResY, mResZ);

		std::cout << "FluidName :" << fluidFn.name().asChar() << std::endl;
		std::cout << "Res X :"  << mResX << std::endl;
		std::cout << "Res Y :"  << mResY << std::endl;
		std::cout << "Res Z :"  << mResZ << std::endl;

		 changeFluidRes(mResX,mResY);*/

		} else {
			std::cout << "Wrong Fluid" << std::endl;
			return MS::kFailure;
		}


		
	} else if (currentMayaFluidName == "") {
			std::cout << "No Fluid" << std::endl;
			return MS::kFailure;
	}

	return MS::kSuccess;

}