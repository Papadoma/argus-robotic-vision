#include <opencv.hpp>
#include <stdio.h>
#include <GL/glut.h>
#include <glm/glm.hpp>

#include <OGRE/Ogre.h>
#include <iostream>

class ogre_model{
private:
	Ogre::ConfigFile cf;
	Ogre::Root *root;
	Ogre::Camera* mCamera;
	Ogre::SceneManager* mSceneMgr;
	Ogre::SceneNode* modelNode;
	Ogre::Viewport *vp;


	Ogre::TexturePtr depthTexture;

	Ogre::Technique* mDepthTechnique;
	Ogre::RenderTexture *depthTarget;

	Ogre::Entity* body;

	void setupResources(void);
	void setupRenderer(void);
	void createDepthRenderTexture();
	void createScene(void);

	cv::Mat image_rgb;
	unsigned char *data;
	Ogre::PixelBox pb;

	int left, top, width, height;
	int render_width, render_height;
public:
	Ogre::RenderWindow* mWindow;

	ogre_model();
	~ogre_model();
	void setup();
	void render();
	void move_model();
	void get_opencv_snap();
	cv::Mat* get_rgb();
};

ogre_model::ogre_model(){
	root = new Ogre::Root("plugins_d.cfg");

	render_width = 800;
	render_height = 600;

	cv::namedWindow("test");

	Ogre::PixelFormat format = Ogre::PF_BYTE_BGRA;
	int outBytesPerPixel = Ogre::PixelUtil::getNumElemBytes(format);
	data = new unsigned char [render_width*render_height*outBytesPerPixel];
	Ogre::Box extents(0, 0, render_width, render_height);
	pb = Ogre::PixelBox(extents, format, data);

	image_rgb = cv::Mat(render_height, render_width, CV_8UC4, data);
}

ogre_model::~ogre_model(){
	delete(data);
}

void ogre_model::setupResources(void)
{

	// Load resource paths from config file
	Ogre::ConfigFile cf;
#if OGRE_DEBUG_MODE
	cf.load("resources_d.cfg");
#else
	cf.load("resources.cfg");
#endif

	// Go through all sections & settings in the file
	Ogre::ConfigFile::SectionIterator seci = cf.getSectionIterator();

	Ogre::String secName, typeName, archName;
	while (seci.hasMoreElements())
	{
		secName = seci.peekNextKey();
		Ogre::ConfigFile::SettingsMultiMap *settings = seci.getNext();
		Ogre::ConfigFile::SettingsMultiMap::iterator i;
		for (i = settings->begin(); i != settings->end(); ++i)
		{
			typeName = i->first;
			archName = i->second;
			Ogre::ResourceGroupManager::getSingleton().addResourceLocation(
					archName, typeName, secName);
		}
	}
}

void ogre_model::setupRenderer(void){
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
	setupRenderer();

	root->initialise(false);

	// create main window
	mWindow = root->createRenderWindow("Main",640,480,false);
	// create the scene
	mSceneMgr = root->createSceneManager(Ogre::ST_GENERIC);
	// add a camera
	mCamera = mSceneMgr->createCamera("MainCam");
	mCamera->setNearClipDistance(1);
	mCamera->setFarClipDistance(1000);

	Ogre::ResourceGroupManager::getSingleton().initialiseAllResourceGroups();

	modelNode = mSceneMgr->getRootSceneNode()->createChildSceneNode("OgreNode");

	body = mSceneMgr->createEntity("ogre", "ninja.mesh");
	body->setCastShadows(false);
	//body->setMaterialName("DoF_Depth");

	Ogre::SkeletonInstance *pSkeletonInst = body->getSkeleton();
	Ogre::Bone *pBone = pSkeletonInst->getBone( "Joint12" );
	pBone->setManuallyControlled(true);
	//pBone->setPosition( Ogre::Vector3( 400, 0, 0 ) ); //move it UP
	pBone->pitch(Ogre::Radian(Ogre::Degree(100)),Ogre::Node::TS_WORLD);


	//createDepthRenderTexture();
	modelNode->attachObject(body);


	// I move the SceneNode so that it is visible to the camera.
	modelNode->translate(0, -100, -300.0f);


	//set the light

	//		Ogre::Light* pointLight = mSceneMgr->createLight("pointLight");
	//		pointLight->setType(Ogre::Light::LT_POINT);
	//		//pointLight->setPosition(Ogre::Vector3(0, 150, 250));
	//		pointLight->setDiffuseColour(0.9, 0.9, 0.9);
	//		pointLight->setSpecularColour(0.9, 0.0, 0.0);

	//make a plane just for fun
	Ogre::MeshPtr pMesh = Ogre::MeshManager::getSingleton().createPlane("Plane",
			Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
			Ogre::Plane(Ogre::Vector3(0.0f, 1.0f, 0.0f), 1.0f),
			500.0, 500.0, 100, 100, true, 1, 1, 1, Ogre::Vector3::NEGATIVE_UNIT_Z);
	Ogre::Entity* pPlaneEntity = mSceneMgr->createEntity("PlaneEntity", "Plane");
	//Ogre::SceneNode* pSceneNodePlane = mmSceneMgr->getRootSceneNode()->createChildSceneNode("PlaneNode");
	pPlaneEntity->setMaterialName("Examples/GrassFloor");
	//pPlaneEntity->setMaterialName("DoF_Depth");
	modelNode->attachObject(pPlaneEntity);

}

void ogre_model::createScene(void)
{
	Ogre::Viewport *vp = mCamera->getViewport();
	mCamera->getViewport()->setDimensions(0.0, 0.0, 0.5, 1.0);
	vp->setBackgroundColour(Ogre::ColourValue(1.0, 0.0, 0.0));
	vp = mWindow->addViewport(mCamera, 1, 0.5, 0.0, 0.5, 1.0);
	vp->setBackgroundColour(Ogre::ColourValue(1.0, 1.0, 0.0));

	createDepthRenderTexture();
	Ogre::CompositorInstance* pCompositor = Ogre::CompositorManager::getSingleton().addCompositor(mWindow->getViewport(1), "DepthMap");
	Ogre::CompositorManager::getSingleton().setCompositorEnabled(mWindow->getViewport(1), "DepthMap", true);
	pCompositor->getTechnique()->getTargetPass(0)->setInputMode(Ogre::CompositionTargetPass::IM_PREVIOUS);

	// light creation, irrelevant
	mSceneMgr->setAmbientLight(Ogre::ColourValue(1.0f, 1.0f, 1.0f));

	// Create the scene node
	Ogre::SceneNode* node = mSceneMgr->getRootSceneNode()->createChildSceneNode("CamNode1", Ogre::Vector3(0, 0, 0));
	node->attachObject(mCamera);

	Ogre::MeshPtr pMesh = Ogre::MeshManager::getSingleton().createPlane("Plane",
			Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
			Ogre::Plane(Ogre::Vector3(0.0f, 1.0f, 0.0f), 0.0f),
			500.0, 500.0, 100, 100, true, 1, 1, 1, Ogre::Vector3::NEGATIVE_UNIT_Z);
	Ogre::Entity* pPlaneEntity = mSceneMgr->createEntity("PlaneEntity", "Plane");
	Ogre::SceneNode* pSceneNodePlane = mSceneMgr->getRootSceneNode()->createChildSceneNode("PlaneNode");
	pSceneNodePlane->attachObject(pPlaneEntity);
	pPlaneEntity->setMaterialName("Document");

	Ogre::Entity* pSphereEntity = mSceneMgr->createEntity("SphereEntity", Ogre::SceneManager::PT_SPHERE);
	Ogre::SceneNode* pSceneNodeSphere = mSceneMgr->getRootSceneNode()->createChildSceneNode("SphereNode");
	pSceneNodeSphere->attachObject(pSphereEntity);
	pSceneNodeSphere->translate(0.0f, 60.0f, 0.0f);
	pSphereEntity->setMaterialName("PlanetMap");
}

void ogre_model::createDepthRenderTexture()
{
	// Create the depth render texture
	Ogre::TexturePtr depthTexture = Ogre::TextureManager::getSingleton().createManual(
			"NinjaDepthMap",
			Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
			Ogre::TEX_TYPE_2D,
			mCamera->getViewport()->getActualWidth(),
			mCamera->getViewport()->getActualHeight(),
			0,
			Ogre::PF_FLOAT16_R,
			Ogre::TU_RENDERTARGET);

	// Get its render target and add a viewport to it
	depthTarget = depthTexture->getBuffer()->getRenderTarget();
	Ogre::Viewport* depthViewport = depthTarget->addViewport(mCamera, 40);
	depthViewport->setBackgroundColour(Ogre::ColourValue::Black);

	// Register 'this' as a render target listener
	//depthTarget->addListener(this);

	// Get the technique to use when rendering the depth render texture
	Ogre::MaterialPtr mDepthMaterial = Ogre::MaterialManager::getSingleton().getByName("DepthMap");
	mDepthMaterial->load(); // needs to be loaded manually
	mDepthTechnique = mDepthMaterial->getBestTechnique();

	// Create a custom render queue invocation sequence for the depth render texture
	Ogre::RenderQueueInvocationSequence* invocationSequence = Ogre::Root::getSingleton().createRenderQueueInvocationSequence("DepthMap");

	// Add a render queue invocation to the sequence, and disable shadows for it
	Ogre::RenderQueueInvocation* invocation = invocationSequence->add(Ogre::RENDER_QUEUE_MAIN, "main");
	invocation->setSuppressShadows(true);

	// Set the render queue invocation sequence for the depth render texture viewport
	depthViewport->setRenderQueueInvocationSequenceName("DepthMap");
	//depthViewport2->setRenderQueueInvocationSequenceName("DepthMap");
}

inline void ogre_model::render(){
	root->renderOneFrame();
}

inline void ogre_model::move_model(){
	modelNode->yaw(Ogre::Radian(Ogre::Degree(0.2)),Ogre::Node::TS_WORLD);
}

inline void ogre_model::get_opencv_snap(){

	depthTexture->getBuffer()->getRenderTarget()->update();
	depthTexture->getBuffer()->getRenderTarget()->copyContentsToMemory(pb, Ogre::RenderTarget::FB_AUTO);
	//renderTexture->writeContentsToFile("start.png");
	std::cout<<"texture fps="<<depthTexture->getBuffer()->getRenderTarget()->getAverageFPS()<< std::endl;

}

inline cv::Mat* ogre_model::get_rgb(){
	return &image_rgb;
}

int main(){
	ogre_model model;
	model.setup();

	while(!model.mWindow->isClosed())
	{
		//float fps = model.Window->getLastFPS();
		model.move_model();

		model.render();

		model.get_opencv_snap();
		imshow("test",*model.get_rgb());
		//std::cout<< *model.get_rgb() << std::endl;

		Ogre::WindowEventUtilities::messagePump();
	}
	std::cout<<"end of program";
	return 1;
}

