#include <opencv.hpp>
#include <stdio.h>
#include <GL/glut.h>
#include <glm/glm.hpp>

#include <OGRE/Ogre.h>
#include <iostream>

#define DEBUG_CONSOLE true
#define DEBUG_WINDOW false
#define FLOAT_DEPTH false

struct skeleton_struct{
	Ogre::Bone*	Head;
	Ogre::Bone*	Torso[6];
	Ogre::Bone*	Right_Arm[3];
	Ogre::Bone* Left_Arm[3];
	Ogre::Bone*	Right_Leg[3];
	Ogre::Bone*	Left_Leg[3];
}model_skeleton;

class ogre_model{
private:
	Ogre::Root *root;
	Ogre::TexturePtr renderToTexture;
	Ogre::SceneNode* modelNode;
	Ogre::Entity* body;
	Ogre::Technique* mDepthTechnique;
	Ogre::SkeletonInstance *modelSkeleton;
	Ogre::PixelBox pb;
	Ogre::RenderWindow* Window;
	cv::Mat image_depth;

	void setupResources(void);
	void setupRenderer(void);
	void setup_bones();
	void setMaxMinDepth();
	void render_window(){	root->renderOneFrame();};
	void setup();

	int render_width, render_height;

#if FLOAT_DEPTH
	float *data;
#else
	char *data;
#endif

public:
	ogre_model();
	~ogre_model();

	void move_model();
	void move_bones();
	void get_opencv_snap();
	cv::Mat* get_depth();
};

ogre_model::ogre_model()
:render_width(640),
 render_height(480)
{
	root = new Ogre::Root("plugins_d.cfg");
	cv::namedWindow("test");
	Ogre::Box extents(0, 0, render_width, render_height);

#if FLOAT_DEPTH
	data = new float [render_width * render_height * Ogre::PixelUtil::getNumElemBytes(Ogre::PF_FLOAT32_R)];
	pb = Ogre::PixelBox(extents, Ogre::PF_FLOAT32_R, data);
	image_depth = cv::Mat(render_height, render_width, CV_32FC1, data);
#else
	data = new char [render_width * render_height * Ogre::PixelUtil::getNumElemBytes(Ogre::PF_L8)];
	pb = Ogre::PixelBox(extents, Ogre::PF_L8, data);
	image_depth = cv::Mat(render_height, render_width, CV_8UC1, data);
#endif

	setup();
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

void ogre_model::setup_bones(){
	model_skeleton.Head = modelSkeleton->getBone( "Head" );
	model_skeleton.Torso[0] = modelSkeleton->getBone( "Upper_Torso" );
	model_skeleton.Torso[1] = modelSkeleton->getBone( "Lower_Torso" );
	model_skeleton.Torso[2] = modelSkeleton->getBone( "Right_Shoulder" );
	model_skeleton.Torso[3] = modelSkeleton->getBone( "Left_Shoulder" );
	model_skeleton.Torso[4] = modelSkeleton->getBone( "Right_Hip" );
	model_skeleton.Torso[5] = modelSkeleton->getBone( "Left_Hip" );

	model_skeleton.Right_Arm[0] = modelSkeleton->getBone( "Right_Arm" );
	model_skeleton.Right_Arm[1] = modelSkeleton->getBone( "Right_Forearm" );
	model_skeleton.Right_Arm[2] = modelSkeleton->getBone( "Right_Hand" );

	model_skeleton.Left_Arm[0] = modelSkeleton->getBone( "Left_Arm" );
	model_skeleton.Left_Arm[1] = modelSkeleton->getBone( "Left_Forearm" );
	model_skeleton.Left_Arm[2] = modelSkeleton->getBone( "Left_Hand" );

	model_skeleton.Right_Leg[0] = modelSkeleton->getBone( "Right_Thigh" );
	model_skeleton.Right_Leg[1] = modelSkeleton->getBone( "Right_Calf" );
	model_skeleton.Right_Leg[2] = modelSkeleton->getBone( "Right_Foot" );

	model_skeleton.Left_Leg[0] = modelSkeleton->getBone( "Left_Thigh" );
	model_skeleton.Left_Leg[1] = modelSkeleton->getBone( "Left_Calf" );
	model_skeleton.Left_Leg[2] = modelSkeleton->getBone( "Left_Foot" );

	model_skeleton.Head->setManuallyControlled(true);
	for(int i=0;i<6;i++){
		model_skeleton.Torso[i]->setManuallyControlled(true);
	}
	for(int i=0;i<3;i++){
		model_skeleton.Right_Arm[i]->setManuallyControlled(true);
		model_skeleton.Left_Arm[i]->setManuallyControlled(true);
		model_skeleton.Right_Leg[i]->setManuallyControlled(true);
		model_skeleton.Left_Leg[i]->setManuallyControlled(true);
	}
}

void ogre_model::setup(){
	//Load renderers and stuff
	setupResources();
	setupRenderer();

	root->initialise(false);

	// create main window
#if DEBUG_WINDOW
	Window = root->createRenderWindow("Main",render_width,render_height,false);
#else
	Window = root->createRenderWindow("Main",1,1,false);
	Window->setHidden(true);
#endif

	// create the scene
	Ogre::SceneManager* SceneMgr = root->createSceneManager(Ogre::ST_GENERIC);
	// add a camera
	Ogre::Camera* camera = SceneMgr->createCamera("MainCam");
	camera->yaw(Ogre::Radian(Ogre::Degree(180)));	//Rotate camera because it faces negative z
	camera->setNearClipDistance(1);
	camera->setFarClipDistance(1000);

	// add viewport
#if DEBUG_WINDOW
	Window->addViewport(camera);
#endif

	Ogre::ResourceGroupManager::getSingleton().initialiseAllResourceGroups();

	modelNode = SceneMgr->getRootSceneNode()->createChildSceneNode("OgreNode");

	body = SceneMgr->createEntity("ogre", "human_model.mesh");
	body->setCastShadows(false);
	modelSkeleton = body->getSkeleton();
	setup_bones();

	//	Ogre::Bone *pBone = modelSkeleton->getBone( "Left_Arm" );
	//	pBone->setManuallyControlled(true);
	//	//pBone->_getDerivedPosition();
	//	//pBone->setPosition( Ogre::Vector3( 400, 0, 0 ) ); //move it UP
	//	pBone->yaw(Ogre::Radian(Ogre::Degree(-90)),Ogre::Node::TS_LOCAL);

	modelNode->attachObject(body);
	//modelNode->showBoundingBox(true);

	// I move the SceneNode so that it is visible to the camera.
	modelNode->translate(0, 0, 300.0f);
	modelNode->yaw(Ogre::Radian(Ogre::Degree(180)),Ogre::Node::TS_WORLD);
	modelNode->setScale(100,100,100);

	//set the light
	//SceneMgr->setAmbientLight(Ogre::ColourValue(1.0f, 1.0f, 1.0f));

#if FLOAT_DEPTH
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
			Ogre::PF_L8,
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

inline void ogre_model::move_bones(){

	model_skeleton.Left_Arm[0]->roll(Ogre::Radian(Ogre::Degree(0.1)),Ogre::Node::TS_LOCAL);
	model_skeleton.Left_Arm[0]->pitch(Ogre::Radian(Ogre::Degree(0.05)),Ogre::Node::TS_LOCAL);
	model_skeleton.Left_Arm[1]->pitch(Ogre::Radian(Ogre::Degree(0.1)),Ogre::Node::TS_LOCAL);
	model_skeleton.Left_Arm[2]->yaw(Ogre::Radian(Ogre::Degree(0.1)),Ogre::Node::TS_LOCAL);
}

inline void ogre_model::get_opencv_snap(){

	renderToTexture->getBuffer()->getRenderTarget()->update();
	setMaxMinDepth();
	renderToTexture->getBuffer()->getRenderTarget()->copyContentsToMemory(pb, Ogre::RenderTarget::FB_AUTO);

#if DEBUG_WINDOW
	render_window();
	std::cout << "[Modeler] Main window fps"<<Window->getLastFPS()<<std::endl;
#endif
	//if(DEBUG_CONSOLE)std::cout<<"[Modeler] texture fps="<<renderToTexture->getBuffer()->getRenderTarget()->getAverageFPS()<< std::endl;
}

inline cv::Mat* ogre_model::get_depth(){
	image_depth = 255 - image_depth;
	threshold(image_depth, image_depth, 254, 255, cv::THRESH_TOZERO_INV);
	return &image_depth;
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

void write_file(std::ofstream& fn, float data){
	fn << (int)data <<std::endl;
}



int main(){
	ogre_model model;
	std::ofstream fn ("file_name.xls");

	while(1)
	{
		model.move_model();
		model.move_bones();
		double t = (double)cv::getTickCount();

		model.get_opencv_snap();

		t = (double)cv::getTickCount() - t;

		float fps = 1/(t/cv::getTickFrequency());
		std::cout << "[Modeler] Total fps" << fps << std::endl;//for fps

		write_file(fn, fps);

		imshow("test",*model.get_depth());

		Ogre::WindowEventUtilities::messagePump();

	}

	fn.close();
	if(DEBUG_CONSOLE)std::cout<<"[Modeler] end of program"<<std::endl;
	return 1;
}

