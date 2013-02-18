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

	Ogre::TexturePtr renderToTexture;

	Ogre::RenderTexture *renderTexture;
	Ogre::Entity* body;

	Ogre::MaterialPtr mDepthMaterial;
	Ogre::Technique* mDepthTechnique;

	Ogre::SkeletonInstance *modelSkeleton;

	void setupResources(void);
	void setupRenderer(void);
	void setMaxMinDepth();

	cv::Mat image_rgb;
	float *data;
	Ogre::PixelBox pb;

	int left, top, width, height;
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

ogre_model::ogre_model(){
	root = new Ogre::Root("plugins_d.cfg");

	render_width = 800;
	render_height = 600;

	cv::namedWindow("test");

	Ogre::PixelFormat format = Ogre::PF_FLOAT32_R;
	int outBytesPerPixel = Ogre::PixelUtil::getNumElemBytes(format);
	std::cout<<outBytesPerPixel<<std::endl;
	data = new float [render_width*render_height*outBytesPerPixel];
	Ogre::Box extents(0, 0, render_width, render_height);
	pb = Ogre::PixelBox(extents, format, data);

	image_rgb = cv::Mat(render_height, render_width, CV_32FC1, data);
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
	Window = root->createRenderWindow("Main",render_width,render_height,false);
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

	modelSkeleton = body->getSkeleton();
	Ogre::Bone *pBone = modelSkeleton->getBone( "Joint9" );

		pBone->setManuallyControlled(true);
	//	pBone->_getDerivedPosition();
	//pBone->setPosition( Ogre::Vector3( 400, 0, 0 ) ); //move it UP
	pBone->roll(Ogre::Radian(Ogre::Degree(90)),Ogre::Node::TS_WORLD);

	modelNode->attachObject(body);
	modelNode->showBoundingBox(true);

	// I move the SceneNode so that it is visible to the camera.
	modelNode->translate(0, -100, -300.0f);

	//set the light
	SceneMgr->setAmbientLight(Ogre::ColourValue(1.0f, 1.0f, 1.0f));

	//make a plane just for fun
	Ogre::MeshPtr pMesh = Ogre::MeshManager::getSingleton().createPlane("Plane",
			Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
			Ogre::Plane(Ogre::Vector3(0.0f, 1.0f, 0.0f), 1.0f),
			500.0, 500.0, 100, 100, true, 1, 1, 1, Ogre::Vector3::NEGATIVE_UNIT_Z);
	Ogre::Entity* pPlaneEntity = SceneMgr->createEntity("PlaneEntity", "Plane");

	//pPlaneEntity->setMaterialName("Examples/GrassFloor");
	pPlaneEntity->setMaterialName("DoF_Depth");
	//modelNode->attachObject(pPlaneEntity);

	vp->getActualDimensions(left, top, width, height);

	renderToTexture = Ogre::TextureManager::getSingleton().createManual("RttTex",
			Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
			Ogre::TEX_TYPE_2D,
			render_width,
			render_height,
			0,
			Ogre::PF_FLOAT32_RGB,		//Ogre::PF_R8G8B8
			Ogre::TU_RENDERTARGET | Ogre::TU_DYNAMIC_WRITE_ONLY_DISCARDABLE);

	renderTexture = renderToTexture->getBuffer()->getRenderTarget();
	renderTexture->addViewport(camera);
	renderTexture->getViewport(0)->setClearEveryFrame(true);
	renderTexture->getViewport(0)->setBackgroundColour(Ogre::ColourValue::White);
	renderTexture->getViewport(0)->setOverlaysEnabled(false);
	renderTexture->getViewport(0)->setShadowsEnabled(false);

//	float mNearDepth = 200.0;
//	float mFocalDepth = 300.0;
//	float mFarDepth = 400.0;
//	float mFarBlurCutoff = 1.0;

	mDepthMaterial = Ogre::MaterialManager::getSingleton().getByName("DoF_Depth");
	mDepthMaterial->load(); // needs to be loaded manually
	mDepthTechnique = mDepthMaterial->getBestTechnique();
	body->setMaterial(mDepthMaterial);


	//Ogre::GpuProgramParametersSharedPtr fragParams = mDepthTechnique->getPass(0)->getFragmentProgramParameters();
	//fragParams->setNamedConstant("dofParams",Ogre::Vector4(mNearDepth, mFocalDepth, mFarDepth, mFarBlurCutoff));

}

void ogre_model::setMaxMinDepth(){
	const Ogre::AxisAlignedBox& axisAlignedBox = body->getWorldBoundingBox();

	Ogre::Vector3 max_values = axisAlignedBox.getMaximum();
	Ogre::Vector3 min_values = axisAlignedBox.getMinimum();

	float max = ceil(fabs(max_values.z));
	float min = floor(fabs(min_values.z));

	std::cout<<"max = "<<fabs(max_values.z)<<" normal"<<max<<" min = "<<fabs(min_values.z)<<" normal "<<min<<std::endl;
	std::cout<<"mean="<<min+(max-min)/2<<std::endl;

	Ogre::GpuProgramParametersSharedPtr fragParams = mDepthTechnique->getPass(0)->getFragmentProgramParameters();
	fragParams->setNamedConstant("dofParams",Ogre::Vector4(min, min+(max-min)/2, max, 1.0));
}

inline void ogre_model::move_model(){
	modelNode->yaw(Ogre::Radian(Ogre::Degree(0.2)),Ogre::Node::TS_WORLD);
}

inline void ogre_model::get_opencv_snap(){
	renderToTexture->getBuffer()->getRenderTarget()->update();
	renderToTexture->getBuffer()->getRenderTarget()->copyContentsToMemory(pb, Ogre::RenderTarget::FB_AUTO);
	setMaxMinDepth();
	//renderTexture->writeContentsToFile("start.png");
	std::cout<<"texture fps="<<renderToTexture->getBuffer()->getRenderTarget()->getAverageFPS()<< std::endl;
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

