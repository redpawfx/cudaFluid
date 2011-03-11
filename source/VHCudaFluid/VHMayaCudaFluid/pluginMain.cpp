#include <maya/MFnPlugin.h> 

#include "fluidNode2D.h"
#include "fluidNode3D.h"


MStatus initializePlugin( MObject obj )
{ 
	MStatus   status;
	MFnPlugin plugin( obj, "Foliativ", "0.9", "Any");

	status = plugin.registerNode("vhFluidSolver2D", fluidNode2D::id, fluidNode2D::creator,
		fluidNode2D::initialize, MPxNode::kLocatorNode);

	status = plugin.registerNode("vhFluidSolver3D", fluidNode3D::id, fluidNode3D::creator,
		fluidNode3D::initialize, MPxNode::kLocatorNode);
	
	return status;
}

MStatus uninitializePlugin( MObject obj )
{
	MStatus   status;
	MFnPlugin plugin( obj );

	status = plugin.deregisterNode(fluidNode2D::id);

	status = plugin.deregisterNode(fluidNode3D::id);

	return status;
}