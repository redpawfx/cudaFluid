Maya plugin :

------------------------------------------------
install :

copy VHFluidCuda11X64.mll in a folder called plug-ins in your maya documents folder. (only version is for maya 2011 x64 on windows)
Might not be needed : 
copy the files in the dll folder in the bin directory of your maya installation (where maya.exe is located)

copy AEvhFluidSolver2DTemplate.mel and AEvhFluidSolver3DTemplate.mel in your scripts directory

------------------------------------------------
usage :

check the mel procedures in fluidSolverSetup.mel to get you started.
fluid2DSetup sets up a 2D fluid container with 2 sphere emitters and 1 sphere collider.
fluid3DSetup sets up a 3D fluid container with 1 sphere emitters and 1 sphere collider.

colliders are locators with a radius attribute which controls the size of the sphere.
emitters are maya fluids emitters, but besides there translate and rotate, only the max distance and Density/Voxel/Sec are taken into account for now.

The container can be translated and rotated, but scale has no effect (except on the wire cube).


The transform cSolverEmptyTr is there to force the solver to update every frame when maya is playing, if you hide it the solver won't update.
A maya fluid container is set up with the correct connections and hidden by the script.
If you uncheck the Preview and DisplaySlice (only Preview for a 2D fluid), at each frame the density is copied from the cuda solver to the maya container, which will allow you to shade it (with the settings on the maya fluid node) and render it.
But it is much slower than if the data stays on the GPU between simulation and display.

Most parameters are quite similar to those of a maya Fluid or self explanatory.
For a 3D fluid, the fluid is raymarched in a texture. Display Res defines the size of the texture. The bigger the texture, the more rays are cast (one per pixel), the slower the rendering, especially with shadows on. Shadows are done with a brute force method and are quite slow. Shadow step mul is a multiplier on the size of the base step to speed things up.


-------------------------------------------------

Not much error checking is done, but the plugin shouldn't crash too much.

Tested successfully with a geforce 460 GTX.
For some reason, advection with the 3D solver doesn't work on my laptop card, a geforce 8600M GS, but the 2D solver does work...
Not sure if this happens with other geforces

To build the source code you need the GPU computing SDK 3.2 x64 and the cuda toolkit 3.2 x64.


The standalone versions were for quick testing and have no GUI nor hotkeys...



04/03/2011 : version 0.5 : initial public release.

To do:
Houdini Version
More complex emitters
Collisions with meshes
Fields
Better advection and algorithms


Thanks to Rob Farber for the perlin noise cuda code.


Please credit me and drop me a line if you do something cool with it!

vincent.houze@foliativ.net
http://www.foliativ.net

