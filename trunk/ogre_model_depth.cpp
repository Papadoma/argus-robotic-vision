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
	Ogre::Camera* camera;
	Ogre::SceneManager* SceneMgr;
	Ogre::SceneNode* modelNode;
	Ogre::Viewport *vp;

	Ogre::TexturePtr mRenderToTexture;
	Ogre::TexturePtr depthTexture;
	Ogre::RenderTexture *renderTexture;
	Ogre::Entity* body;

	void setupResources(void);
	void setupRenderer(void);
	void createDepthRenderTexture();

	cv::Mat image_rgb;
	unsigned char *data;
	Ogre::PixelBox pb;

	int left, top, width, height;
	int render_width, render_height;
public:
	Ogre::RenderWindow* Window;
	ogre_model();
	~ogre_model();
	void setup();
	void render();
	void move_model(int);
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
	Window = root->createRenderWindow("Main",640,480,false);
	// create the scene
	SceneMgr = root->createSceneManager(Ogre::ST_GENERIC);
	// add a camera
	camera = SceneMgr->createCamera("MainCam");
	camera->setNearClipDistance(1);
	camera->setFarClipDistance(1000);
	// add viewport
	vp = Window->addViewport(camera);

	Ogre::ResourceGroupManager::getSingleton().initialiseAllResourceGroups();

	modelNode = SceneMgr->getRootSceneNode()->createChildSceneNode("OgreNode");

	body = SceneMgr->createEntity("ogre", "ninja.mesh");
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
	SceneMgr->setAmbientLight(Ogre::ColourValue(1.0f, 1.0f, 1.0f));

	//		Ogre::Light* pointLight = SceneMgr->createLight("pointLight");
	//		pointLight->setType(Ogre::Light::LT_POINT);
	//		//pointLight->setPosition(Ogre::Vector3(0, 150, 250));
	//		pointLight->setDiffuseColour(0.9, 0.9, 0.9);
	//		pointLight->setSpecularColour(0.9, 0.0, 0.0);

	//make a plane just for fun
	Ogre::MeshPtr pMesh = Ogre::MeshManager::getSingleton().createPlane("Plane",
			Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
			Ogre::Plane(Ogre::Vector3(0.0f, 1.0f, 0.0f), 1.0f),
			500.0, 500.0, 100, 100, true, 1, 1, 1, Ogre::Vector3::NEGATIVE_UNIT_Z);
	Ogre::Entity* pPlaneEntity = SceneMgr->createEntity("PlaneEntity", "Plane");
	//Ogre::SceneNode* pSceneNodePlane = mSceneMgr->getRootSceneNode()->createChildSceneNode("PlaneNode");
	pPlaneEntity->setMaterialName("Examples/GrassFloor");
	//pPlaneEntity->setMaterialName("DoF_Depth");
	modelNode->attachObject(pPlaneEntity);

	vp->getActualDimensions(left, top, width, height);

	mRenderToTexture = Ogre::TextureManager::getSingleton().createManual("RttTex",
			Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
			Ogre::TEX_TYPE_2D,
			render_width,
			render_height,
			0,
			Ogre::PF_R8G8B8,		//Ogre::PF_R8G8B8
			Ogre::TU_RENDERTARGET | Ogre::TU_DYNAMIC_WRITE_ONLY_DISCARDABLE);

	renderTexture = mRenderToTexture->getBuffer()->getRenderTarget();
	renderTexture->addViewport(camera);
	renderTexture->getViewport(0)->setClearEveryFrame(true);
	renderTexture->getViewport(0)->setBackgroundColour(Ogre::ColourValue::Black);
	renderTexture->getViewport(0)->setOverlaysEnabled(true);
	renderTexture->getViewport(0)->setShadowsEnabled(false);
}

inline void ogre_model::render(){
	root->renderOneFrame();
}

inline void ogre_model::move_model(int deg){


	modelNode->yaw(Ogre::Radian(Ogre::Degree(0.2)),Ogre::Node::TS_WORLD);

}

inline void ogre_model::get_opencv_snap(){

	mRenderToTexture->getBuffer()->getRenderTarget()->update();
	mRenderToTexture->getBuffer()->getRenderTarget()->copyContentsToMemory(pb, Ogre::RenderTarget::FB_AUTO);
	//renderTexture->writeContentsToFile("start.png");
	std::cout<<"texture fps="<<mRenderToTexture->getBuffer()->getRenderTarget()->getAverageFPS()<< std::endl;

}

inline cv::Mat* ogre_model::get_rgb(){
	return &image_rgb;
}

int main(){
	ogre_model model;
	model.setup();

	while(!model.Window->isClosed())
	{
		//float fps = model.Window->getLastFPS();
		model.move_model(10);

		model.render();

		model.get_opencv_snap();
		imshow("test",*model.get_rgb());
		//std::cout<< *model.get_rgb() << std::endl;

		Ogre::WindowEventUtilities::messagePump();
	}
	std::cout<<"end of program";
	return 1;
}

