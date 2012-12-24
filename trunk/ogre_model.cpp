#include <OGRE/OgreCamera.h>
#include <OGRE/OgreEntity.h>
#include <OGRE/OgreLogManager.h>
#include <OGRE/OgreRoot.h>
#include <OGRE/OgreViewport.h>
#include <OGRE/OgreSceneManager.h>
#include <OGRE/OgreRenderWindow.h>
#include <OGRE/OgreConfigFile.h>


int main(){

	Ogre::Root *root;
	Ogre::Camera* camera;
	Ogre::SceneManager* SceneMgr;
	Ogre::RenderWindow* Window;
	Ogre::Viewport *vp;

	// create root
	root = new Ogre::Root();
	// choose renderer
	if(!root->showConfigDialog())
	{
		return 0;
	}
	// initialise root
	root->initialise(false);
	// create main window
	Window = root->createRenderWindow("Main",320,240,false);
	// create the scene
	SceneMgr = root->createSceneManager(Ogre::ST_GENERIC);
	// add a camera
	camera = SceneMgr->createCamera("MainCam");
	// add viewport
	vp = Window->addViewport(camera);

	root->startRendering();
}
