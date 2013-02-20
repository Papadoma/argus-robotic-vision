#include <opencv.hpp>
#include <stdio.h>
#include <GL/glut.h>
#include <glm/glm.hpp>

#include <OGRE/Ogre.h>
#include <iostream>

#define DEBUG true
#define FLOAT false

class ogre_model{
private:
	Ogre::Root *root;

	Ogre::TexturePtr renderToTexture;

	Ogre::SceneNode* modelNode;
	Ogre::Entity* body;

	Ogre::Technique* mDepthTechnique;
	Ogre::SkeletonInstance *modelSkeleton;

	void setupResources(void);
	void setupRenderer(void);
	void setMaxMinDepth();

	cv::Mat image_rgb;

#if FLOAT
	float *data;
#else
	char *data;
#endif

	Ogre::PixelBox pb;

	int render_width, render_height;
public:
	Ogre::RenderWindow* Window;
	ogre_model();
	~ogre_model();
	void setup();
	void render(){	root->renderOneFrame();};
	void move_model();
	void get_opencv_snap();
	cv::Mat* get_rgb(){	return &image_rgb;};
};

ogre_model::ogre_model()
:render_width(640),
 render_height(480)
{
	root = new Ogre::Root("plugins_d.cfg");

	cv::namedWindow("test");
	Ogre::Box extents(0, 0, render_width, render_height);
#if float
	data = new float [render_width * render_height * Ogre::PixelUtil::getNumElemBytes(Ogre::PF_FLOAT32_R)];
	pb = Ogre::PixelBox(extents, Ogre::PF_FLOAT32_R, data);
	image_rgb = cv::Mat(render_height, render_width, CV_32FC1, data);
#else
	data = new char [render_width * render_height * Ogre::PixelUtil::getNumElemBytes(Ogre::PF_L8)];
	pb = Ogre::PixelBox(extents, Ogre::PF_L8, data);
	image_rgb = cv::Mat(render_height, render_width, CV_8UC1, data);
#endif
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
	Ogre::RenderSystem *lRenderSystem = lRenderSystemList[0]; //Set first renderer, using only OpenGL
	root->setRenderSystem(lRenderSystem);
}

void ogre_model::setup(){
	//Load renderers and stuff
	setupResources();
	setupRenderer();

	root->initialise(false);

	// create main window
	Window = root->createRenderWindow("Main",render_width,render_height,false);
	// create the scene
	Ogre::SceneManager* SceneMgr = root->createSceneManager(Ogre::ST_GENERIC);
	// add a camera
	Ogre::Camera* camera = SceneMgr->createCamera("MainCam");
	camera->yaw(Ogre::Radian(Ogre::Degree(180)));	//Rotate camera because it faces negative z
	camera->setNearClipDistance(1);
	camera->setFarClipDistance(1000);
	// add viewport
	Ogre::Viewport* vp = Window->addViewport(camera);

	Ogre::ResourceGroupManager::getSingleton().initialiseAllResourceGroups();

	modelNode = SceneMgr->getRootSceneNode()->createChildSceneNode("OgreNode");

	body = SceneMgr->createEntity("ogre", "human_model.mesh");
	body->setCastShadows(false);

	modelSkeleton = body->getSkeleton();

	Ogre::Bone *pBone = modelSkeleton->getBone( "Left_Arm" );
	pBone->setManuallyControlled(true);
	//pBone->_getDerivedPosition();
	//pBone->setPosition( Ogre::Vector3( 400, 0, 0 ) ); //move it UP
	pBone->yaw(Ogre::Radian(Ogre::Degree(-90)),Ogre::Node::TS_LOCAL);

	modelNode->attachObject(body);
	//modelNode->showBoundingBox(true);

	// I move the SceneNode so that it is visible to the camera.
	modelNode->translate(0, 0, 300.0f);
	modelNode->yaw(Ogre::Radian(Ogre::Degree(180)),Ogre::Node::TS_WORLD);
	modelNode->setScale(100,100,100);

	//set the light
	//SceneMgr->setAmbientLight(Ogre::ColourValue(1.0f, 1.0f, 1.0f));

#if FLOAT
	renderToTexture = Ogre::TextureManager::getSingleton().createManual("RttTex",
			Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
			Ogre::TEX_TYPE_2D,
			render_width,
			render_height,
			0,
			Ogre::PF_FLOAT32_RGB,
			Ogre::TU_RENDERTARGET | Ogre::TU_DYNAMIC_WRITE_ONLY_DISCARDABLE);
#else
	renderToTexture = Ogre::TextureManager::getSingleton().createManual("RttTex",
			Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
			Ogre::TEX_TYPE_2D,
			render_width,
			render_height,
			0,
			Ogre::PF_R8G8B8,
			Ogre::TU_RENDERTARGET | Ogre::TU_DYNAMIC_WRITE_ONLY_DISCARDABLE);
#endif

	Ogre::RenderTexture* renderTexture = renderToTexture->getBuffer()->getRenderTarget();
	renderTexture->addViewport(camera);
	renderTexture->getViewport(0)->setClearEveryFrame(true);
	renderTexture->getViewport(0)->setBackgroundColour(Ogre::ColourValue::Black);
	renderTexture->getViewport(0)->setOverlaysEnabled(false);
	renderTexture->getViewport(0)->setShadowsEnabled(false);

	Ogre::MaterialPtr mDepthMaterial = Ogre::MaterialManager::getSingleton().getByName("DoF_Depth");
	mDepthMaterial->load(); // needs to be loaded manually
	mDepthTechnique = mDepthMaterial->getBestTechnique();
	body->setMaterial(mDepthMaterial);
}

inline void ogre_model::move_model(){
	modelNode->yaw(Ogre::Radian(Ogre::Degree(0.2)),Ogre::Node::TS_LOCAL);
}

inline void ogre_model::get_opencv_snap(){
	setMaxMinDepth();
	renderToTexture->getBuffer()->getRenderTarget()->update();
	renderToTexture->getBuffer()->getRenderTarget()->copyContentsToMemory(pb, Ogre::RenderTarget::FB_AUTO);

	if(DEBUG)std::cout<<"texture fps="<<renderToTexture->getBuffer()->getRenderTarget()->getAverageFPS()<< std::endl;
}

inline void ogre_model::setMaxMinDepth(){
	const Ogre::Sphere& bodySphere = body->getWorldBoundingSphere();
	Ogre::Vector3 SphereCenter = bodySphere.getCenter();
	Ogre::Real SphereRadius = bodySphere.getRadius();

	float max = ceil(SphereCenter.z+SphereRadius);
	float min = floor(SphereCenter.z-SphereRadius);
	float center = SphereCenter.z;

	Ogre::GpuProgramParametersSharedPtr fragParams = mDepthTechnique->getPass(0)->getFragmentProgramParameters();
	fragParams->setNamedConstant("dofParams",Ogre::Vector4(min, center, max, 1.0));
}

int main(){
	ogre_model model;
	model.setup();

	while(!model.Window->isClosed())
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

