#include <opencv.hpp>
#include <stdio.h>

#include <OGRE/OgreCamera.h>
#include <OGRE/OgreEntity.h>
#include <OGRE/OgreLogManager.h>
#include <OGRE/OgreRoot.h>
#include <OGRE/OgreViewport.h>
#include <OGRE/OgreSceneManager.h>
#include <OGRE/OgreRenderWindow.h>
#include <OGRE/OgreConfigFile.h>
#include "OGRE/OgreWindowEventUtilities.h"
#include <iostream>

//#include <OGRE/SdkTrays.h>
//#include <OGRE/SdkCameraMan.h>

using namespace std;
using namespace Ogre;

class ogre_model{
private:
	Ogre::ConfigFile cf;
	Ogre::Root *root;
	Ogre::Camera* camera;
	Ogre::SceneManager* SceneMgr;
	Ogre::SceneNode* modelNode;
	Ogre::Viewport *vp;

	void setupResources(void);
	void setupPlugins(void);
	void createDepthRenderTexture();
public:
	Ogre::RenderWindow* Window;
	ogre_model();
	~ogre_model();
	void setup();
	void render();
	void move_model(int);
};

ogre_model::ogre_model(){
	root = new Ogre::Root();

}

ogre_model::~ogre_model(){

}

void ogre_model::setupResources(void)
{

	// Load resource paths from config file
	ConfigFile cf;
#if OGRE_DEBUG_MODE
	cf.load("resources_d.cfg");
#else
	cf.load("resources.cfg");
#endif

	// Go through all sections & settings in the file
	ConfigFile::SectionIterator seci = cf.getSectionIterator();

	String secName, typeName, archName;
	while (seci.hasMoreElements())
	{
		secName = seci.peekNextKey();
		ConfigFile::SettingsMultiMap *settings = seci.getNext();
		ConfigFile::SettingsMultiMap::iterator i;
		for (i = settings->begin(); i != settings->end(); ++i)
		{
			typeName = i->first;
			archName = i->second;
			ResourceGroupManager::getSingleton().addResourceLocation(
					archName, typeName, secName);

		}
	}
}

void ogre_model::setupPlugins(void){
	root->loadPlugin("D:/ogre3d_repos/built/ogre-sdk/bin/debug/RenderSystem_GL_d");
	const Ogre::RenderSystemList& lRenderSystemList = root->getAvailableRenderers();
	if( lRenderSystemList.size() == 0 )
	{
		Ogre::LogManager::getSingleton().logMessage("Sorry, no rendersystem was found.");
	}
	Ogre::RenderSystem *lRenderSystem = lRenderSystemList[0];
	root->setRenderSystem(lRenderSystem);
}

void ogre_model::setup(){
	//Load renderers and stuff


	setupResources();
	setupPlugins();

	root->initialise(false);

	//	//Load meshes
	//	cf.load("resources_d.cfg");
	//	// Go through all sections & settings in the file
	//	Ogre::ConfigFile::SectionIterator seci = cf.getSectionIterator();
	//
	//	Ogre::String secName, typeName, archName;
	//	while (seci.hasMoreElements())
	//	{
	//		secName = seci.peekNextKey();
	//		Ogre::ConfigFile::SettingsMultiMap *settings = seci.getNext();
	//		Ogre::ConfigFile::SettingsMultiMap::iterator i;
	//		for (i = settings->begin(); i != settings->end(); ++i)
	//		{
	//			typeName = i->first;
	//			archName = i->second;
	//			Ogre::ResourceGroupManager::getSingleton().addResourceLocation(
	//					archName, typeName, secName);
	//		}
	//	}


	// create main window
	Window = root->createRenderWindow("Main",640,480,false);
	// create the scene
	SceneMgr = root->createSceneManager(Ogre::ST_GENERIC);
	// add a camera
	camera = SceneMgr->createCamera("MainCam");
	// add viewport
	vp = Window->addViewport(camera);

	ResourceGroupManager::getSingleton().initialiseAllResourceGroups();



	//add mesh
	Ogre::Entity* ogre = SceneMgr->createEntity("ogre", "ninja.mesh");
	ogre->setCastShadows(true);
	// Create a SceneNode and attach the Entity to it
	modelNode = SceneMgr->getRootSceneNode()->createChildSceneNode("OgreNode");
	modelNode->attachObject(ogre);

	//ogre->setMaterialName("Examples/Ninja");

	// I move the SceneNode so that it is visible to the camera.
	modelNode->translate(0, -100, -300.0f);


	//set the light
	SceneMgr->setAmbientLight(Ogre::ColourValue(0.1f, 0.1f, 0.1f));

	Ogre::Light* pointLight = SceneMgr->createLight("pointLight");
	pointLight->setType(Ogre::Light::LT_POINT);
	//pointLight->setPosition(Ogre::Vector3(0, 150, 250));
	pointLight->setDiffuseColour(0.9, 0.9, 0.9);
	pointLight->setSpecularColour(0.9, 0.0, 0.0);

	MaterialPtr material = MaterialManager::getSingleton().getByName("DoF_DepthDebug");
	material->getTechnique(0)->getPass(0)->getTextureUnitState(0)->setTextureName("DoF_DepthDebug");
ogre->setMaterial(material);

	Ogre::ResourceManager::ResourceMapIterator materialIterator = Ogre::MaterialManager::getSingleton().getResourceIterator();
	while (materialIterator.hasMoreElements())
	{
		cout<< (static_cast<Ogre::MaterialPtr>(materialIterator.peekNextValue()))->getName()<<"\n";
		materialIterator.moveNext();
		//count++;
	}

}
void ogre_model::createDepthRenderTexture(){
	//	// Create the depth render texture
	//	Ogre::TexturePtr depthTexture = Ogre::TextureManager::getSingleton().createManual(
	//			"NinjaDepthMap", Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
	//			Ogre::TEX_TYPE_2D, camera->getViewport()->getActualWidth(), camera->getViewport()->getActualHeight(),
	//			0, Ogre::PF_FLOAT16_R, Ogre::TU_RENDERTARGET);
	//
	//	// Get its render target and add a viewport to it
	//	Ogre::RenderTexture *depthTarget = depthTexture->getBuffer()->getRenderTarget();
	//	Ogre::Viewport* depthViewport = depthTarget->addViewport(camera, 40);
	//	depthViewport->setBackgroundColour(Ogre::ColourValue::Black);
	//
	//	// Register 'this' as a render target listener
	//	//depthTarget->addListener(this);
	//
	//	// Get the technique to use when rendering the depth render texture
	//	Ogre::MaterialPtr mDepthMaterial = Ogre::MaterialManager::getSingleton().getByName("DepthMap");
	//	mDepthMaterial->load(); // needs to be loaded manually
	//	Ogre::Technique *mDepthTechnique = mDepthMaterial->getBestTechnique();
	//
	//	// Create a custom render queue invocation sequence for the depth render texture
	//	Ogre::RenderQueueInvocationSequence* invocationSequence =
	//			Ogre::Root::getSingleton().createRenderQueueInvocationSequence("DepthMap");
	//
	//	// Add a render queue invocation to the sequence, and disable shadows for it
	//	Ogre::RenderQueueInvocation* invocation = invocationSequence->add(Ogre::RENDER_QUEUE_MAIN, "main");
	//	invocation->setSuppressShadows(true);
	//
	//	// Set the render queue invocation sequence for the depth render texture viewport
	//	depthViewport->setRenderQueueInvocationSequenceName("DepthMap");
	//	//depthViewport2->setRenderQueueInvocationSequenceName("DepthMap");

}

void ogre_model::render(){
	root->renderOneFrame();
}

void ogre_model::move_model(int deg){
	modelNode->yaw(Ogre::Radian(Ogre::Degree(0.02)),Ogre::Node::TS_WORLD);

}

int main(){
	ogre_model model;
	model.setup();

	while(!model.Window->isClosed())
	{
		float fps = model.Window->getLastFPS();
		model.move_model(1);

		model.render();
		//cout<<"fps="<<fps<<"\n";
		//cout<<"fps="<<fps<<"\n";
		Ogre::WindowEventUtilities::messagePump();
	}
	std::cout<<"end of program";
	return 1;
}

