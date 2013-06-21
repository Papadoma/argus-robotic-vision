/*
 * module_input.cpp
 *
 * This file is part of my final year's project for the Department
 * of Electrical and Computer Engineering of Aristotle University
 * of Thessaloniki, 2013.
 *
 * Author:	Miltiadis-Alexios Papadopoulos
 *
 */

#include "ogre_modeler.hpp"
#include <boost/thread.hpp>

ogre_model::ogre_model(int render_width, int render_height)
:render_width(render_width),
 render_height(render_height)
{
	root = new Ogre::Root("plugins_d.cfg");

	depth_list = new cv::Mat[MAX_RENDER_REQUESTS];		//640x480x100x1Byte = 30MB
	extremas_list = new cv::Mat_<cv::Point>[MAX_RENDER_REQUESTS];
	for(int i=0 ; i<MAX_RENDER_REQUESTS ; i++){
		depth_list[i] = cv::Mat(render_height, render_width, CV_8UC1);
		extremas_list[i] = cv::Mat_<cv::Point>::zeros(24,1);
	}

	//#if DEPTH_MODE == 1
	//image_depth = cv::Mat(render_height, render_width, CV_8UC1);
	//#elif DEPTH_MODE == 2
	//	data = new unsigned short [render_width * render_height * Ogre::PixelUtil::getNumElemBytes(Ogre::PF_L16)];
	//	pb = Ogre::PixelBox(extents, Ogre::PF_L16, data);
	//	image_depth = cv::Mat(render_height, render_width, CV_16UC1, data);
	//#else
	//	data = new float [render_width * render_height * Ogre::PixelUtil::getNumElemBytes(Ogre::PF_FLOAT32_R)];
	//	pb = Ogre::PixelBox(extents, Ogre::PF_FLOAT32_R, data);
	//	image_depth = cv::Mat(render_height, render_width, CV_32FC1, data);
	//
	//#endif

	setup();
}

ogre_model::~ogre_model(){
	delete(depth_list);
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
	lRenderSystem->setConfigOption("VSync", "No");
	lRenderSystem->setConfigOption("Full Screen", "Yes");
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
	Window->setVSyncEnabled(false);

	// create the scene
	Ogre::SceneManager* SceneMgr = root->createSceneManager(Ogre::ST_GENERIC);
	// add a camera
	camera = SceneMgr->createCamera("MainCam");
	camera->yaw(Ogre::Radian(Ogre::Degree(180)));	//Rotate camera because it faces negative z
	camera->setNearClipDistance(1000);
	camera->setFarClipDistance(10000);

	Ogre::Matrix4 cam_mat = camera->getProjectionMatrix();
	cam_mat[0][0] = 539.65684452821188 / 640;
	cam_mat[1][1] = 539.28421750515940 / 480;
//	cam_mat[2][2] = -0.100;
//	cam_mat[3][2] = -1.000;
	cam_mat[3][3] = 0.000;
	//camera->setCustomProjectionMatrix(true, cam_mat);
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

	modelNode->attachObject(body);
	//modelNode->showBoundingBox(true);

	// I move the SceneNode so that it is visible to the camera.
	modelNode->setPosition(0, 0, 300.0f);
	modelNode->yaw(Ogre::Radian(Ogre::Degree(180)),Ogre::Node::TS_WORLD);
	modelNode->setScale(100,100,100);
	modelNode->setInitialState();

	//set the light
	//SceneMgr->setAmbientLight(Ogre::ColourValue(1.0f, 1.0f, 1.0f));

#if DEPTH_MODE == 1
	renderToTexture = Ogre::TextureManager::getSingleton().createManual("RttTex",
			Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
			Ogre::TEX_TYPE_2D,
			render_width,
			render_height,
			0,
			Ogre::PF_L8,
			Ogre::TU_RENDERTARGET | Ogre::TU_DYNAMIC_WRITE_ONLY_DISCARDABLE);
#elif DEPTH_MODE == 2
	renderToTexture = Ogre::TextureManager::getSingleton().createManual("RttTex",
			Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
			Ogre::TEX_TYPE_2D,
			render_width,
			render_height,
			0,
			Ogre::PF_FLOAT16_RGB,
			Ogre::TU_RENDERTARGET | Ogre::TU_DYNAMIC_WRITE_ONLY_DISCARDABLE);
#else
	renderToTexture = Ogre::TextureManager::getSingleton().createManual("RttTex",
			Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
			Ogre::TEX_TYPE_2D,
			render_width,
			render_height,
			0,
			Ogre::PF_FLOAT32_RGB,
			Ogre::TU_RENDERTARGET | Ogre::TU_DYNAMIC_WRITE_ONLY_DISCARDABLE);
#endif

	Ogre::RenderTexture* renderTexture = renderToTexture->getBuffer()->getRenderTarget();
	renderTexture->addViewport(camera);
	renderTexture->getViewport(0)->setClearEveryFrame(true);
	renderTexture->getViewport(0)->setBackgroundColour(Ogre::ColourValue::White);
	renderTexture->getViewport(0)->setOverlaysEnabled(false);
	renderTexture->getViewport(0)->setShadowsEnabled(false);

	Ogre::MaterialPtr mDepthMaterial = Ogre::MaterialManager::getSingleton().getByName("DoF_Depth");
	mDepthMaterial->load(); // needs to be loaded manually
	mDepthTechnique = mDepthMaterial->getBestTechnique();
	body->setMaterial(mDepthMaterial);


	//	targettex1 = Ogre::TextureManager::getSingleton().createManual("targettex1",
	//			Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
	//			Ogre::TEX_TYPE_2D,
	//			render_width,
	//			render_height,
	//			0,
	//			Ogre::PF_L8,
	//			Ogre::TU_DEFAULT  );
	//
	//	targettex2 = Ogre::TextureManager::getSingleton().createManual("targettex2",
	//			Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
	//			Ogre::TEX_TYPE_2D,
	//			render_width,
	//			render_height,
	//			0,
	//			Ogre::PF_L8,
	//			Ogre::TU_DEFAULT );
	//
	//	targettex3 = Ogre::TextureManager::getSingleton().createManual("targettex3",
	//			Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
	//			Ogre::TEX_TYPE_2D,
	//			render_width,
	//			render_height,
	//			0,
	//			Ogre::PF_L8,
	//			Ogre::TU_DEFAULT  );
	//
	//	rtt1 = Ogre::TextureManager::getSingleton().createManual("rtt1",
	//			Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
	//			Ogre::TEX_TYPE_2D,
	//			render_width,
	//			render_height,
	//			0,
	//			Ogre::PF_L8,
	//			Ogre::TU_RENDERTARGET);
	//
	//	rtt2 = Ogre::TextureManager::getSingleton().createManual("rtt2",
	//			Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
	//			Ogre::TEX_TYPE_2D,
	//			render_width,
	//			render_height,
	//			0,
	//			Ogre::PF_L8,
	//			Ogre::TU_RENDERTARGET);
	//
	//	rtt3 = Ogre::TextureManager::getSingleton().createManual("rtt3",
	//			Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
	//			Ogre::TEX_TYPE_2D,
	//			render_width,
	//			render_height,
	//			0,
	//			Ogre::PF_L8,
	//			Ogre::TU_RENDERTARGET);
	//
	//	rtt1->getBuffer()->getRenderTarget()->addViewport(camera);
	//	rtt1->getBuffer()->getRenderTarget()->getViewport(0)->setClearEveryFrame(true);
	//	rtt1->getBuffer()->getRenderTarget()->getViewport(0)->setBackgroundColour(Ogre::ColourValue::White);
	//	rtt1->getBuffer()->getRenderTarget()->getViewport(0)->setOverlaysEnabled(false);
	//	rtt1->getBuffer()->getRenderTarget()->getViewport(0)->setShadowsEnabled(false);
	//
	//	rtt2->getBuffer()->getRenderTarget()->addViewport(camera);
	//	rtt2->getBuffer()->getRenderTarget()->getViewport(0)->setClearEveryFrame(true);
	//	rtt2->getBuffer()->getRenderTarget()->getViewport(0)->setBackgroundColour(Ogre::ColourValue::White);
	//	rtt2->getBuffer()->getRenderTarget()->getViewport(0)->setOverlaysEnabled(false);
	//	rtt2->getBuffer()->getRenderTarget()->getViewport(0)->setShadowsEnabled(false);
	//
	//	rtt3->getBuffer()->getRenderTarget()->addViewport(camera);
	//	rtt3->getBuffer()->getRenderTarget()->getViewport(0)->setClearEveryFrame(true);
	//	rtt3->getBuffer()->getRenderTarget()->getViewport(0)->setBackgroundColour(Ogre::ColourValue::White);
	//	rtt3->getBuffer()->getRenderTarget()->getViewport(0)->setOverlaysEnabled(false);
	//	rtt3->getBuffer()->getRenderTarget()->getViewport(0)->setShadowsEnabled(false);
}

inline void ogre_model::reset_bones(){
	model_skeleton.Head->reset();
	for(int i=0;i<6;i++){
		model_skeleton.Torso[i]->reset();
	}
	for(int i=0;i<3;i++){
		model_skeleton.Right_Arm[i]->reset();
		model_skeleton.Left_Arm[i]->reset();
		model_skeleton.Right_Leg[i]->reset();
		model_skeleton.Left_Leg[i]->reset();
	}
}

inline void ogre_model::reset_model(){
	reset_bones();
	modelNode->resetToInitialState();
}

inline void ogre_model::move_model(const cv::Point3f& position = cv::Point3f(0,0,300.0f), const cv::Point3f& rotation = cv::Point3f(0,0,0), const float& scale = 100){
	modelNode->resetToInitialState();
	modelNode->setPosition(position.x, position.y, position.z);
	modelNode->yaw(Ogre::Radian(Ogre::Degree(rotation.x)),Ogre::Node::TS_LOCAL);
	modelNode->pitch(Ogre::Radian(Ogre::Degree(rotation.y)),Ogre::Node::TS_LOCAL);
	modelNode->roll(Ogre::Radian(Ogre::Degree(rotation.z)),Ogre::Node::TS_LOCAL);
	modelNode->setScale(scale,scale,scale);
}

inline void ogre_model::rotate_bones(const cv::Mat& bones_rotation){
	reset_bones();

	model_skeleton.Head -> yaw(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(0,0))),Ogre::Node::TS_LOCAL);//Head
	model_skeleton.Head -> pitch(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(0,1))),Ogre::Node::TS_LOCAL);//Head
	model_skeleton.Head -> roll(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(0,2))),Ogre::Node::TS_LOCAL);//Head

	model_skeleton.Torso[0] -> yaw(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(1,0))),Ogre::Node::TS_LOCAL); //Upper_Torso
	model_skeleton.Torso[0] -> pitch(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(1,1))),Ogre::Node::TS_LOCAL); //Upper_Torso
	model_skeleton.Torso[0] -> roll(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(1,2))),Ogre::Node::TS_LOCAL); //Upper_Torso

	model_skeleton.Torso[1] -> yaw(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(2,0))),Ogre::Node::TS_LOCAL);//Lower_Torso
	model_skeleton.Torso[1] -> pitch(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(2,1))),Ogre::Node::TS_LOCAL);//Lower_Torso
	model_skeleton.Torso[1] -> roll(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(2,2))),Ogre::Node::TS_LOCAL);//Lower_Torso

	model_skeleton.Torso[2] -> yaw(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(3,0))),Ogre::Node::TS_LOCAL);//Right_Shoulder
	model_skeleton.Torso[2] -> pitch(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(3,1))),Ogre::Node::TS_LOCAL);//Right_Shoulder
	model_skeleton.Torso[2] -> roll(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(3,2))),Ogre::Node::TS_LOCAL);//Right_Shoulder

	model_skeleton.Torso[3] -> yaw(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(4,0))),Ogre::Node::TS_LOCAL);//Left_Shoulder
	model_skeleton.Torso[3] -> pitch(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(4,1))),Ogre::Node::TS_LOCAL);//Left_Shoulder
	model_skeleton.Torso[3] -> roll(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(4,2))),Ogre::Node::TS_LOCAL);//Left_Shoulder

	model_skeleton.Torso[4] -> yaw(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(5,0))),Ogre::Node::TS_LOCAL);//Right_Hip
	model_skeleton.Torso[4] -> pitch(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(5,1))),Ogre::Node::TS_LOCAL);//Right_Hip
	model_skeleton.Torso[4] -> roll(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(5,2))),Ogre::Node::TS_LOCAL);//Right_Hip

	model_skeleton.Torso[5] -> yaw(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(6,0))),Ogre::Node::TS_LOCAL);//Left_Hip
	model_skeleton.Torso[5] -> pitch(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(6,1))),Ogre::Node::TS_LOCAL);//Left_Hip
	model_skeleton.Torso[5] -> roll(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(6,2))),Ogre::Node::TS_LOCAL);//Left_Hip

	model_skeleton.Right_Arm[0] -> yaw(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(7,0))),Ogre::Node::TS_LOCAL);//Right_Arm
	model_skeleton.Right_Arm[0] -> pitch(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(7,1))),Ogre::Node::TS_LOCAL);//Right_Arm
	model_skeleton.Right_Arm[0] -> roll(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(7,2))),Ogre::Node::TS_LOCAL);//Right_Arm

	model_skeleton.Right_Arm[1] -> yaw(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(8,0))),Ogre::Node::TS_LOCAL);//Right_Forearm
	model_skeleton.Right_Arm[1] -> pitch(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(8,1))),Ogre::Node::TS_LOCAL);//Right_Forearm
	model_skeleton.Right_Arm[1] -> roll(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(8,2))),Ogre::Node::TS_LOCAL);//Right_Forearm

	model_skeleton.Right_Arm[2] -> yaw(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(9,0))),Ogre::Node::TS_LOCAL);//Right_Hand
	model_skeleton.Right_Arm[2] -> pitch(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(9,1))),Ogre::Node::TS_LOCAL);//Right_Hand
	model_skeleton.Right_Arm[2] -> roll(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(9,2))),Ogre::Node::TS_LOCAL);//Right_Hand

	model_skeleton.Left_Arm[0] -> yaw(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(10,0))),Ogre::Node::TS_LOCAL);//Left_Arm
	model_skeleton.Left_Arm[0] -> pitch(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(10,1))),Ogre::Node::TS_LOCAL);//Left_Arm
	model_skeleton.Left_Arm[0] -> roll(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(10,2))),Ogre::Node::TS_LOCAL);//Left_Arm

	model_skeleton.Left_Arm[1] -> yaw(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(11,0))),Ogre::Node::TS_LOCAL);//Left_Forearm
	model_skeleton.Left_Arm[1] -> pitch(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(11,1))),Ogre::Node::TS_LOCAL);//Left_Forearm
	model_skeleton.Left_Arm[1] -> roll(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(11,2))),Ogre::Node::TS_LOCAL);//Left_Forearm

	model_skeleton.Left_Arm[2] -> yaw(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(12,0))),Ogre::Node::TS_LOCAL);//Left_Hand
	model_skeleton.Left_Arm[2] -> pitch(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(12,1))),Ogre::Node::TS_LOCAL);//Left_Hand
	model_skeleton.Left_Arm[2] -> roll(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(12,2))),Ogre::Node::TS_LOCAL);//Left_Hand

	model_skeleton.Right_Leg[0] -> yaw(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(13,0))),Ogre::Node::TS_LOCAL);//Right_Thigh
	model_skeleton.Right_Leg[0] -> pitch(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(13,1))),Ogre::Node::TS_LOCAL);//Right_Thigh
	model_skeleton.Right_Leg[0] -> roll(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(13,2))),Ogre::Node::TS_LOCAL);//Right_Thigh

	model_skeleton.Right_Leg[1] -> yaw(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(14,0))),Ogre::Node::TS_LOCAL);//Right_Calf
	model_skeleton.Right_Leg[1] -> pitch(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(14,1))),Ogre::Node::TS_LOCAL);//Right_Calf
	model_skeleton.Right_Leg[1] -> roll(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(14,2))),Ogre::Node::TS_LOCAL);//Right_Calf

	model_skeleton.Right_Leg[2] -> yaw(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(15,0))),Ogre::Node::TS_LOCAL);//Right_Foot
	model_skeleton.Right_Leg[2] -> pitch(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(15,1))),Ogre::Node::TS_LOCAL);//Right_Foot
	model_skeleton.Right_Leg[2] -> roll(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(15,2))),Ogre::Node::TS_LOCAL);//Right_Foot

	model_skeleton.Left_Leg[0] -> yaw(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(16,0))),Ogre::Node::TS_LOCAL);//Left_Thigh
	model_skeleton.Left_Leg[0] -> pitch(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(16,1))),Ogre::Node::TS_LOCAL);//Left_Thigh
	model_skeleton.Left_Leg[0] -> roll(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(16,2))),Ogre::Node::TS_LOCAL);//Left_Thigh

	model_skeleton.Left_Leg[1] -> yaw(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(17,0))),Ogre::Node::TS_LOCAL);//Left_Calf
	model_skeleton.Left_Leg[1] -> pitch(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(17,1))),Ogre::Node::TS_LOCAL);//Left_Calf
	model_skeleton.Left_Leg[1] -> roll(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(17,2))),Ogre::Node::TS_LOCAL);//Left_Calf

	model_skeleton.Left_Leg[2] -> yaw(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(18,0))),Ogre::Node::TS_LOCAL);//Left_Foot
	model_skeleton.Left_Leg[2] -> pitch(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(18,1))),Ogre::Node::TS_LOCAL);//Left_Foot
	model_skeleton.Left_Leg[2] -> roll(Ogre::Radian(Ogre::Degree(bones_rotation.at<float>(18,2))),Ogre::Node::TS_LOCAL);//Left_Foot

}

/**
 * 2D position of every joint, in 16bit unsigned 2 channel cv::mat
 */
inline void ogre_model::get_2D_pos(const int& id){
	Ogre::Vector3 head2d = modelNode->_getDerivedPosition() +  modelNode->_getDerivedOrientation() * model_skeleton.Head->_getDerivedPosition()*modelNode->_getDerivedScale();
	Ogre::Vector3 u_torso2d = modelNode->_getDerivedPosition() + modelNode->_getDerivedOrientation() * model_skeleton.Torso[0]->_getDerivedPosition()*modelNode->_getDerivedScale();
	Ogre::Vector3 l_torsod = modelNode->_getDerivedPosition() + modelNode->_getDerivedOrientation() * model_skeleton.Torso[1]->_getDerivedPosition()*modelNode->_getDerivedScale();
	Ogre::Vector3 r_shoulder2d = modelNode->_getDerivedPosition() + modelNode->_getDerivedOrientation() * model_skeleton.Torso[2]->_getDerivedPosition()*modelNode->_getDerivedScale();
	Ogre::Vector3 l_shoulder2d = modelNode->_getDerivedPosition() + modelNode->_getDerivedOrientation() * model_skeleton.Torso[3]->_getDerivedPosition()*modelNode->_getDerivedScale();
	Ogre::Vector3 r_hip2d = modelNode->_getDerivedPosition() + modelNode->_getDerivedOrientation() * model_skeleton.Torso[4]->_getDerivedPosition()*modelNode->_getDerivedScale();
	Ogre::Vector3 l_hip2d = modelNode->_getDerivedPosition() + modelNode->_getDerivedOrientation() * model_skeleton.Torso[5]->_getDerivedPosition()*modelNode->_getDerivedScale();
	Ogre::Vector3 r_arm2d = modelNode->_getDerivedPosition() + modelNode->_getDerivedOrientation() * model_skeleton.Right_Arm[0]->_getDerivedPosition()*modelNode->_getDerivedScale();
	Ogre::Vector3 r_forearm2d = modelNode->_getDerivedPosition() + modelNode->_getDerivedOrientation() * model_skeleton.Right_Arm[1]->_getDerivedPosition()*modelNode->_getDerivedScale();
	Ogre::Vector3 r_hand2d = modelNode->_getDerivedPosition() + modelNode->_getDerivedOrientation() * model_skeleton.Right_Arm[2]->_getDerivedPosition()*modelNode->_getDerivedScale();
	Ogre::Vector3 l_arm2d = modelNode->_getDerivedPosition() + modelNode->_getDerivedOrientation() * model_skeleton.Left_Arm[0]->_getDerivedPosition()*modelNode->_getDerivedScale();
	Ogre::Vector3 l_forearm2d = modelNode->_getDerivedPosition() + modelNode->_getDerivedOrientation() * model_skeleton.Left_Arm[1]->_getDerivedPosition()*modelNode->_getDerivedScale();
	Ogre::Vector3 l_hand2d = modelNode->_getDerivedPosition() + modelNode->_getDerivedOrientation() * model_skeleton.Left_Arm[2]->_getDerivedPosition()*modelNode->_getDerivedScale();
	Ogre::Vector3 r_thigh2d = modelNode->_getDerivedPosition() + modelNode->_getDerivedOrientation() * model_skeleton.Right_Leg[0]->_getDerivedPosition()*modelNode->_getDerivedScale();
	Ogre::Vector3 r_calf2d = modelNode->_getDerivedPosition() + modelNode->_getDerivedOrientation() * model_skeleton.Right_Leg[1]->_getDerivedPosition()*modelNode->_getDerivedScale();
	Ogre::Vector3 r_foot2d = modelNode->_getDerivedPosition() + modelNode->_getDerivedOrientation() * model_skeleton.Right_Leg[2]->_getDerivedPosition()*modelNode->_getDerivedScale();
	Ogre::Vector3 l_thigh2d = modelNode->_getDerivedPosition() + modelNode->_getDerivedOrientation() * model_skeleton.Left_Leg[0]->_getDerivedPosition()*modelNode->_getDerivedScale();
	Ogre::Vector3 l_calf2d = modelNode->_getDerivedPosition() + modelNode->_getDerivedOrientation() * model_skeleton.Left_Leg[1]->_getDerivedPosition()*modelNode->_getDerivedScale();
	Ogre::Vector3 l_foot2d = modelNode->_getDerivedPosition() + modelNode->_getDerivedOrientation() * model_skeleton.Left_Leg[2]->_getDerivedPosition()*modelNode->_getDerivedScale();

	Ogre::Vector3 r_hand_edge2d = modelNode->_getDerivedPosition() + modelNode->_getDerivedOrientation() * modelSkeleton->getBone( "Right_Hand_Edge" )->_getDerivedPosition()*modelNode->_getDerivedScale();
	Ogre::Vector3 l_hand_edge2d = modelNode->_getDerivedPosition() + modelNode->_getDerivedOrientation() * modelSkeleton->getBone( "Left_Hand_Edge" )->_getDerivedPosition()*modelNode->_getDerivedScale();
	Ogre::Vector3 r_foot_edge2d = modelNode->_getDerivedPosition() + modelNode->_getDerivedOrientation() * modelSkeleton->getBone( "Right_Foot_Edge" )->_getDerivedPosition()*modelNode->_getDerivedScale();
	Ogre::Vector3 l_foot_edge2d = modelNode->_getDerivedPosition() + modelNode->_getDerivedOrientation() * modelSkeleton->getBone( "Left_Foot_Edge" )->_getDerivedPosition()*modelNode->_getDerivedScale();
	Ogre::Vector3 head_edge2d = modelNode->_getDerivedPosition() + modelNode->_getDerivedOrientation() * modelSkeleton->getBone( "Head_Edge" )->_getDerivedPosition()*modelNode->_getDerivedScale();


	head2d = camera->getProjectionMatrix()*camera->getViewMatrix()*head2d;
	u_torso2d = camera->getProjectionMatrix()*camera->getViewMatrix()*u_torso2d;
	l_torsod = camera->getProjectionMatrix()*camera->getViewMatrix()*l_torsod;
	r_shoulder2d = camera->getProjectionMatrix()*camera->getViewMatrix()*r_shoulder2d;
	l_shoulder2d = camera->getProjectionMatrix()*camera->getViewMatrix()*l_shoulder2d;
	r_hip2d = camera->getProjectionMatrix()*camera->getViewMatrix()*r_hip2d;
	l_hip2d = camera->getProjectionMatrix()*camera->getViewMatrix()*l_hip2d;
	r_arm2d = camera->getProjectionMatrix()*camera->getViewMatrix()*r_arm2d;
	r_forearm2d = camera->getProjectionMatrix()*camera->getViewMatrix()*r_forearm2d;
	r_hand2d = camera->getProjectionMatrix()*camera->getViewMatrix()*r_hand2d;
	l_arm2d = camera->getProjectionMatrix()*camera->getViewMatrix()*l_arm2d;
	l_forearm2d = camera->getProjectionMatrix()*camera->getViewMatrix()*l_forearm2d;
	l_hand2d = camera->getProjectionMatrix()*camera->getViewMatrix()*l_hand2d;
	r_thigh2d = camera->getProjectionMatrix()*camera->getViewMatrix()*r_thigh2d;
	r_calf2d = camera->getProjectionMatrix()*camera->getViewMatrix()*r_calf2d;
	r_foot2d = camera->getProjectionMatrix()*camera->getViewMatrix()*r_foot2d;
	l_thigh2d = camera->getProjectionMatrix()*camera->getViewMatrix()*l_thigh2d;
	l_calf2d = camera->getProjectionMatrix()*camera->getViewMatrix()*l_calf2d;
	l_foot2d = camera->getProjectionMatrix()*camera->getViewMatrix()*l_foot2d;
	r_hand_edge2d = camera->getProjectionMatrix()*camera->getViewMatrix()*r_hand_edge2d;
	l_hand_edge2d = camera->getProjectionMatrix()*camera->getViewMatrix()*l_hand_edge2d;
	r_foot_edge2d = camera->getProjectionMatrix()*camera->getViewMatrix()*r_foot_edge2d;
	l_foot_edge2d = camera->getProjectionMatrix()*camera->getViewMatrix()*l_foot_edge2d;
	head_edge2d = camera->getProjectionMatrix()*camera->getViewMatrix()*head_edge2d;

	extremas_list[id].at<cv::Point>(0).x = (0.5 + head2d.x/2)*render_width;
	extremas_list[id].at<cv::Point>(0).y = (0.5 - head2d.y/2)*render_height;
	extremas_list[id].at<cv::Point>(1).x = (0.5 + u_torso2d.x/2)*render_width;
	extremas_list[id].at<cv::Point>(1).y = (0.5 - u_torso2d.y/2)*render_height;
	extremas_list[id].at<cv::Point>(2).x = (0.5 + l_torsod.x/2)*render_width;
	extremas_list[id].at<cv::Point>(2).y = (0.5 - l_torsod.y/2)*render_height;
	extremas_list[id].at<cv::Point>(3).x = (0.5 + r_shoulder2d.x/2)*render_width;
	extremas_list[id].at<cv::Point>(3).y = (0.5 - r_shoulder2d.y/2)*render_height;
	extremas_list[id].at<cv::Point>(4).x = (0.5 + l_shoulder2d.x/2)*render_width;
	extremas_list[id].at<cv::Point>(4).y = (0.5 - l_shoulder2d.y/2)*render_height;
	extremas_list[id].at<cv::Point>(5).x = (0.5 + r_hip2d.x/2)*render_width;
	extremas_list[id].at<cv::Point>(5).y = (0.5 - r_hip2d.y/2)*render_height;
	extremas_list[id].at<cv::Point>(6).x = (0.5 + l_hip2d.x/2)*render_width;
	extremas_list[id].at<cv::Point>(6).y = (0.5 - l_hip2d.y/2)*render_height;
	extremas_list[id].at<cv::Point>(7).x = (0.5 + r_arm2d.x/2)*render_width;
	extremas_list[id].at<cv::Point>(7).y = (0.5 - r_arm2d.y/2)*render_height;
	extremas_list[id].at<cv::Point>(8).x = (0.5 + r_forearm2d.x/2)*render_width;
	extremas_list[id].at<cv::Point>(8).y = (0.5 - r_forearm2d.y/2)*render_height;
	extremas_list[id].at<cv::Point>(9).x = (0.5 + r_hand2d.x/2)*render_width;
	extremas_list[id].at<cv::Point>(9).y = (0.5 - r_hand2d.y/2)*render_height;
	extremas_list[id].at<cv::Point>(10).x = (0.5 + l_arm2d.x/2)*render_width;
	extremas_list[id].at<cv::Point>(10).y = (0.5 - l_arm2d.y/2)*render_height;
	extremas_list[id].at<cv::Point>(11).x = (0.5 + l_forearm2d.x/2)*render_width;
	extremas_list[id].at<cv::Point>(11).y = (0.5 - l_forearm2d.y/2)*render_height;
	extremas_list[id].at<cv::Point>(12).x = (0.5 + l_hand2d.x/2)*render_width;
	extremas_list[id].at<cv::Point>(12).y = (0.5 - l_hand2d.y/2)*render_height;
	extremas_list[id].at<cv::Point>(13).x = (0.5 + r_thigh2d.x/2)*render_width;
	extremas_list[id].at<cv::Point>(13).y = (0.5 - r_thigh2d.y/2)*render_height;
	extremas_list[id].at<cv::Point>(14).x = (0.5 + r_calf2d.x/2)*render_width;
	extremas_list[id].at<cv::Point>(14).y = (0.5 - r_calf2d.y/2)*render_height;
	extremas_list[id].at<cv::Point>(15).x = (0.5 + r_foot2d.x/2)*render_width;
	extremas_list[id].at<cv::Point>(15).y = (0.5 - r_foot2d.y/2)*render_height;
	extremas_list[id].at<cv::Point>(16).x = (0.5 + l_thigh2d.x/2)*render_width;
	extremas_list[id].at<cv::Point>(16).y = (0.5 - l_thigh2d.y/2)*render_height;
	extremas_list[id].at<cv::Point>(17).x = (0.5 + l_calf2d.x/2)*render_width;
	extremas_list[id].at<cv::Point>(17).y = (0.5 - l_calf2d.y/2)*render_height;
	extremas_list[id].at<cv::Point>(18).x = (0.5 + l_foot2d.x/2)*render_width;
	extremas_list[id].at<cv::Point>(18).y = (0.5 - l_foot2d.y/2)*render_height;

	extremas_list[id].at<cv::Point>(19).x = (0.5 + head_edge2d.x/2)*render_width;
	extremas_list[id].at<cv::Point>(19).y = (0.5 - head_edge2d.y/2)*render_height;
	extremas_list[id].at<cv::Point>(20).x = (0.5 + r_hand_edge2d.x/2)*render_width;
	extremas_list[id].at<cv::Point>(20).y = (0.5 - r_hand_edge2d.y/2)*render_height;
	extremas_list[id].at<cv::Point>(21).x = (0.5 + l_hand_edge2d.x/2)*render_width;
	extremas_list[id].at<cv::Point>(21).y = (0.5 - l_hand_edge2d.y/2)*render_height;
	extremas_list[id].at<cv::Point>(22).x = (0.5 + r_foot_edge2d.x/2)*render_width;
	extremas_list[id].at<cv::Point>(22).y = (0.5 - r_foot_edge2d.y/2)*render_height;
	extremas_list[id].at<cv::Point>(23).x = (0.5 + l_foot_edge2d.x/2)*render_width;
	extremas_list[id].at<cv::Point>(23).y = (0.5 - l_foot_edge2d.y/2)*render_height;
}

cv::Mat* ogre_model::get_depth(const std::vector<particle_position>& particles_list){

	for(int i=0 ; i<(int)particles_list.size() ; i++){
		move_model(particles_list[i].model_position, particles_list[i].model_rotation, particles_list[i].scale);
		rotate_bones(particles_list[i].bones_rotation);
		renderToTexture->getBuffer()->getRenderTarget()->update();
		get_2D_pos(i);
		renderToTexture->getBuffer()->getRenderTarget()->copyContentsToMemory(Ogre::PixelBox(Ogre::Box(0, 0, render_width, render_height), Ogre::PF_L8, depth_list[i].data), Ogre::RenderTarget::FB_AUTO);
	}
	return depth_list;
	//	move_model(particles_list[0].model_position, particles_list[0].model_rotation, particles_list[0].scale);
	//	rotate_bones(particles_list[0].bones_rotation);
	//	rtt1->getBuffer()->getRenderTarget()->update();
	//	//double t = (double)cv::getTickCount();
	//	targettex1->getBuffer()->blit(rtt1->getBuffer());
	//	//t = ((double)cv::getTickCount() - t)*1000./cv::getTickFrequency();
	//	//std::cout<<"[Modeler]Simple render time: "<<t<<"ms"<<std::endl;
	//
	//	move_model(particles_list[1].model_position, particles_list[1].model_rotation, particles_list[1].scale);
	//	rotate_bones(particles_list[1].bones_rotation);
	//	rtt2->getBuffer()->getRenderTarget()->update();
	//	targettex2->getBuffer()->blit(rtt2->getBuffer());
	//
	//	move_model(particles_list[2].model_position, particles_list[2].model_rotation, particles_list[2].scale);
	//	rotate_bones(particles_list[2].bones_rotation);
	//	rtt3->getBuffer()->getRenderTarget()->update();
	//	targettex3->getBuffer()->blit(rtt3->getBuffer());
	//
	//	//TODO fix limits of rendering
	//	for(int i=3 ; i<(int)particles_list.size()-2 ; i+=3){
	//		move_model(particles_list[i].model_position, particles_list[i].model_rotation, particles_list[i].scale);
	//		rotate_bones(particles_list[i].bones_rotation);
	//		targettex1->getBuffer()->blitToMemory(Ogre::PixelBox(Ogre::Box(0, 0, render_width, render_height), Ogre::PF_L8, depth_list[i-3].data));
	//		rtt1->getBuffer()->getRenderTarget()->update();
	//		targettex1->getBuffer()->blit(rtt1->getBuffer());
	//
	//		move_model(particles_list[i+1].model_position, particles_list[i+1].model_rotation, particles_list[i+1].scale);
	//		rotate_bones(particles_list[i+1].bones_rotation);
	//		targettex2->getBuffer()->blitToMemory(Ogre::PixelBox(Ogre::Box(0, 0, render_width, render_height), Ogre::PF_L8, depth_list[i-2].data));
	//		rtt2->getBuffer()->getRenderTarget()->update();
	//		targettex2->getBuffer()->blit(rtt2->getBuffer());
	//
	//		move_model(particles_list[i+2].model_position, particles_list[i+2].model_rotation, particles_list[i+2].scale);
	//		rotate_bones(particles_list[i+2].bones_rotation);
	//		targettex3->getBuffer()->blitToMemory(Ogre::PixelBox(Ogre::Box(0, 0, render_width, render_height), Ogre::PF_L8, depth_list[i-1].data));
	//		rtt3->getBuffer()->getRenderTarget()->update();
	//		targettex3->getBuffer()->blit(rtt3->getBuffer());
	//	}

	//MADMARX's PROPOSED METHOD
	//	move_model(particles_list[0].model_position, particles_list[0].model_rotation, particles_list[0].scale);
	//	rotate_bones(particles_list[0].bones_rotation);
	//	rtt1->getBuffer()->getRenderTarget()->update();
	//
	//	for(int i=1 ; i<(int)particles_list.size() ; i+=3){
	//		//std::cout<<"i:"<<i<<std::endl;
	//		move_model(particles_list[i].model_position, particles_list[i].model_rotation, particles_list[i].scale);
	//		rotate_bones(particles_list[i].bones_rotation);
	//		rtt2->getBuffer()->getRenderTarget()->update();
	//		double t = (double)cv::getTickCount();
	//		targettex1->getBuffer()->blit(rtt1->getBuffer());
	//		t = ((double)cv::getTickCount() - t)*1000./cv::getTickFrequency();
	//		std::cout<<"[Modeler]Copy time: "<<t<<"ms"<<std::endl;
	//
	//
	//		if(i+1<(int)particles_list.size()){
	//			move_model(particles_list[i+1].model_position, particles_list[i+1].model_rotation, particles_list[i+1].scale);
	//			rotate_bones(particles_list[i+1].bones_rotation);
	//			rtt3->getBuffer()->getRenderTarget()->update();
	//		}
	//		targettex2->getBuffer()->blit(rtt2->getBuffer());
	//
	//		if(i+2<(int)particles_list.size()){
	//			move_model(particles_list[i+2].model_position, particles_list[i+2].model_rotation, particles_list[i+2].scale);
	//			rotate_bones(particles_list[i+2].bones_rotation);
	//			rtt1->getBuffer()->getRenderTarget()->update();
	//		}
	//		if(i+1<(int)particles_list.size())
	//			targettex3->getBuffer()->blit(rtt3->getBuffer());
	//
	//
	//		targettex1->getBuffer()->blitToMemory(Ogre::PixelBox(Ogre::Box(0, 0, render_width, render_height), Ogre::PF_L8, depth_list[i-1].data));
	//		targettex2->getBuffer()->blitToMemory(Ogre::PixelBox(Ogre::Box(0, 0, render_width, render_height), Ogre::PF_L8, depth_list[i].data));
	//		if(i+1<(int)particles_list.size())
	//			targettex3->getBuffer()->blitToMemory(Ogre::PixelBox(Ogre::Box(0, 0, render_width, render_height), Ogre::PF_L8, depth_list[i+1].data));
	//
	//		//		double t = (double)cv::getTickCount();
	//		//		memcpy(static_cast<uchar*>(targettex1->getBuffer()->lock(Ogre::Box(0, 0, render_width, render_height),Ogre::HardwareBuffer::HBL_READ_ONLY).data), depth_list[i-1].data, targettex1-> getBuffer() ->getSizeInBytes());
	//		//		t = ((double)cv::getTickCount() - t)*1000./cv::getTickFrequency();
	//		//		std::cout<<"[Modeler]Copy time: "<<t<<"ms"<<std::endl;
	//		//		memcpy(static_cast<uchar*>(targettex2->getBuffer()->lock(Ogre::Box(0, 0, render_width, render_height),Ogre::HardwareBuffer::HBL_READ_ONLY).data), depth_list[i].data, targettex2-> getBuffer() ->getSizeInBytes());
	//		//		if(i+1<(int)particles_list.size())
	//		//			memcpy(static_cast<uchar*>(targettex3->getBuffer()->lock(Ogre::Box(0, 0, render_width, render_height),Ogre::HardwareBuffer::HBL_READ_ONLY).data), depth_list[i+1].data, targettex3-> getBuffer() ->getSizeInBytes());
	//		//		targettex1->getBuffer()->unlock();
	//		//		targettex2->getBuffer()->unlock();
	//		//		if(i+1<(int)particles_list.size())
	//		//			targettex3->getBuffer()->unlock();
	//	}

#if DEBUG_WINDOW
	render_window();
	std::cout << "[Modeler] Main window fps"<<Window->getLastFPS()<<std::endl;
#endif
	return depth_list;
	//	#if DEPTH_MODE == 1
	//		//	swarm[i].particle_depth = 255 - swarm[i].particle_depth;
	//		//	threshold(swarm[i].particle_depth, swarm[i].particle_depth, 254, 255, cv::THRESH_TOZERO_INV);
	//	#elif DEPTH_MODE == 2
	//		cv::Mat temp;
	//		image_depth = 65535 - image_depth;
	//		image_depth.convertTo(temp, CV_32FC1);
	//		threshold(temp, temp, 65534, 65535, cv::THRESH_TOZERO_INV);
	//		temp.convertTo(image_depth, CV_16UC1);
	//	#else
	//		image_depth = 1 - image_depth;
	//		threshold(image_depth, image_depth, 0.99, 1, cv::THRESH_TOZERO_INV);
	//	#endif
}

float ogre_model::get_fps(){
	return renderToTexture->getBuffer()->getRenderTarget()->getAverageFPS();
}

void ogre_model::set_depth_limits(float min_set = -1, float center_set = -1, float max_set = -1){
	const Ogre::Sphere& bodySphere = body->getWorldBoundingSphere();
	Ogre::Vector3 SphereCenter = bodySphere.getCenter();
	Ogre::Real SphereRadius = bodySphere.getRadius();

	if(min_set==-1 || center_set==-1 || max_set==-1){
		max = ceil(SphereCenter.z+SphereRadius);
		min = floor(SphereCenter.z-SphereRadius);
		center = SphereCenter.z;
	}else{
		max = ceil(max_set);
		min = floor(min_set);
		center = round(center_set);
	}
	setMaxMinDepth();
}

void ogre_model::set_camera_clip(float near_dist,float far_dist){
	camera->setNearClipDistance(near_dist);
	camera->setFarClipDistance(far_dist);
}

void ogre_model::setMaxMinDepth(){
	//std::cout<< min<<" "<< center<<" "<< max<<std::endl;
	Ogre::GpuProgramParametersSharedPtr fragParams = mDepthTechnique->getPass(0)->getFragmentProgramParameters();
	fragParams->setNamedConstant("dofParams",Ogre::Vector4(min, center, max, 1.0));
}

cv::Mat_<cv::Point3f> ogre_model::get_camera_viewspace(){
	const Ogre::Vector3* coords = camera->getWorldSpaceCorners();
	cv::Mat_<cv::Point3f> coordsCV(8,1);
	for (int i=0 ; i<8 ; i++){
		coordsCV(i,0).x = coords[i].x;
		coordsCV(i,0).y = coords[i].y;
		coordsCV(i,0).z = coords[i].z;
	}

	return coordsCV;
}

inline void ogre_model::getMeshInformation(const Ogre::MeshPtr mesh,
		size_t &vertex_count,
		Ogre::Vector3* &vertices,
		size_t &index_count,
		unsigned* &indices,
		const Ogre::Vector3 &position,
		const Ogre::Quaternion &orient,
		const Ogre::Vector3 &scale)
{
	bool added_shared = false;
	size_t current_offset = 0;
	size_t shared_offset = 0;
	size_t next_offset = 0;
	size_t index_offset = 0;

	vertex_count = index_count = 0;

	// Calculate how many vertices and indices we're going to need
	for ( unsigned short i = 0; i < mesh->getNumSubMeshes(); ++i)
	{
		Ogre::SubMesh* submesh = mesh->getSubMesh(i);
		// We only need to add the shared vertices once
		if(submesh->useSharedVertices)
		{
			if( !added_shared )
			{
				vertex_count += mesh->sharedVertexData->vertexCount;
				added_shared = true;
			}
		}
		else
		{
			vertex_count += submesh->vertexData->vertexCount;
		}
		// Add the indices
		index_count += submesh->indexData->indexCount;
	}

	// Allocate space for the vertices and indices
	vertices = new Ogre::Vector3[vertex_count];
	indices = new unsigned[index_count];

	added_shared = false;

	// Run through the submeshes again, adding the data into the arrays
	for (unsigned short i = 0; i < mesh->getNumSubMeshes(); ++i)
	{
		Ogre::SubMesh* submesh = mesh->getSubMesh(i);

		//Ogre::VertexData* vertex_data = submesh->useSharedVertices ? mesh->sharedVertexData : submesh->vertexData;
		Ogre::VertexData* vertex_data = body->_getSkelAnimVertexData();

		if ((!submesh->useSharedVertices) || (submesh->useSharedVertices && !added_shared))
		{
			if(submesh->useSharedVertices)
			{
				added_shared = true;
				shared_offset = current_offset;
			}

			const Ogre::VertexElement* posElem =
					vertex_data->vertexDeclaration->findElementBySemantic(Ogre::VES_POSITION);

			Ogre::HardwareVertexBufferSharedPtr vbuf = vertex_data->vertexBufferBinding->getBuffer(posElem->getSource());
			//Ogre::HardwareVertexBufferSharedPtr vbuf = renderToTexture->getBuffer(posElem->getSource());

			//renderToTexture->getBuffer()

			unsigned char* vertex =
					static_cast<unsigned char*>(vbuf->lock(Ogre::HardwareBuffer::HBL_READ_ONLY));

			// There is _no_ baseVertexPointerToElement() which takes an Ogre::Real or a double
			//  as second argument. So make it float, to avoid trouble when Ogre::Real will
			//  be comiled/typedefed as double:
			//Ogre::Real* pReal;
			float* pReal;

			for( size_t j = 0; j < vertex_data->vertexCount; ++j, vertex += vbuf->getVertexSize())
			{
				posElem->baseVertexPointerToElement(vertex, &pReal);
				Ogre::Vector3 pt(pReal[0], pReal[1], pReal[2]);
				vertices[current_offset + j] = (orient * (pt * scale)) + position;
			}

			vbuf->unlock();
			next_offset += vertex_data->vertexCount;
		}

		Ogre::IndexData* index_data = submesh->indexData;
		size_t numTris = index_data->indexCount / 3;
		Ogre::HardwareIndexBufferSharedPtr ibuf = index_data->indexBuffer;

		bool use32bitindexes = (ibuf->getType() == Ogre::HardwareIndexBuffer::IT_32BIT);

		unsigned long* pLong = static_cast<unsigned long*>(ibuf->lock(Ogre::HardwareBuffer::HBL_READ_ONLY));
		unsigned short* pShort = reinterpret_cast<unsigned short*>(pLong);

		size_t offset = (submesh->useSharedVertices)? shared_offset : current_offset;

		if ( use32bitindexes )
		{
			for ( size_t k = 0; k < numTris*3; ++k)
			{
				indices[index_offset++] = pLong[k] + static_cast<unsigned long>(offset);
			}
		}
		else
		{
			for ( size_t k = 0; k < numTris*3; ++k)
			{
				indices[index_offset++] = static_cast<unsigned long>(pShort[k]) +
						static_cast<unsigned long>(offset);
			}
		}

		ibuf->unlock();
		current_offset = next_offset;
	}
}

cv::Mat ogre_model::get_segmentation(){
	cv::Mat result = cv::Mat::zeros(render_height,render_width, CV_8UC1);
	Ogre::MeshPtr model_mesh = body->getMesh();
	Ogre::Mesh::BoneAssignmentIterator bone_iter = model_mesh->getBoneAssignmentIterator();

	size_t vertex_count;
	size_t index_count;
	Ogre::Vector3 *vertices;
	Ogre::uint32 *indices;
	//body->
	// get the mesh information
	getMeshInformation(body->getMesh(), vertex_count, vertices, index_count, indices,
			body->getParentNode()->_getDerivedPosition(),
			body->getParentNode()->_getDerivedOrientation(),
			body->getParentNode()->_getDerivedScale());

	//	std::cout<<"Vertices in mesh: "<<vertex_count<<std::endl;
	//	std::cout<<"Triangles in mesh: "<<index_count / 3<<std::endl;

	//For every vertex, find out which bone it belongs to
	unsigned short vertex2bone[vertex_count];
	while(bone_iter.hasMoreElements()){
		Ogre::VertexBoneAssignment_s vertex_bone = bone_iter.getNext();
		vertex2bone[vertex_bone.vertexIndex]=vertex_bone.boneIndex;
	}

	//Iterate through mesh triangles, composed of 3 vertices
	cv::Point triangle[index_count/3][3];
	Ogre::Vector3 point0,point1,point2;
	cv::Mat vertices_depth(index_count/3, 1, CV_32FC1);
	cv::Mat vertices_color(index_count/3, 1, CV_8UC1);
	for (size_t i = 0; i < index_count; i += 3)
	{

		point0 = camera->getProjectionMatrix()*camera->getViewMatrix()*vertices[indices[i]];
		point1 = camera->getProjectionMatrix()*camera->getViewMatrix()*vertices[indices[i+1]] ;
		point2 = camera->getProjectionMatrix()*camera->getViewMatrix()*vertices[indices[i+2]];

		vertices_depth.at<float>(i/3,0)=((point0+point1+point2)/3).z;

		triangle[i/3][0] = cv::Point((0.5 + point0.x/2)*render_width,(0.5 - point0.y/2)*render_height);
		triangle[i/3][1] = cv::Point((0.5 + point1.x/2)*render_width,(0.5 - point1.y/2)*render_height);
		triangle[i/3][2] = cv::Point((0.5 + point2.x/2)*render_width,(0.5 - point2.y/2)*render_height);

		unsigned char triangle2bone=0 ;//= vertex2bone[indices[i]];
		if(vertex2bone[indices[i]] == vertex2bone[indices[i+1]])triangle2bone = vertex2bone[indices[i]];
		if(vertex2bone[indices[i+1]] == vertex2bone[indices[i+2]])triangle2bone = vertex2bone[indices[i+1]];
		if(vertex2bone[indices[i]] == vertex2bone[indices[i+2]])triangle2bone = vertex2bone[indices[i]];
		vertices_color.at<uchar>(i/3,0)=triangle2bone;

	}
	cv::Mat cv_indices;
	cv::sortIdx(vertices_depth, cv_indices, cv::SORT_EVERY_COLUMN | cv::SORT_DESCENDING);

	for (int i = 0; i < cv_indices.rows; i ++)
	{
		fillConvexPoly(result, triangle[cv_indices.at<int>(i,0)], 3, cv::Scalar(vertices_color.at<uchar>(cv_indices.at<int>(i,0),0)));
	}
	//std::cout<<vertices[indices[100]]<<std::endl;
	cv::normalize(result, result, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	//	std::cout<<"index_count"<<index_count<<std::endl;
	//	int count=0;
	//
	//	std::cout<<"bone_iter "<<count<<std::endl;

	delete[] vertices;
	delete[] indices;
	return result;
}

#undef DEBUG_CONSOLE
#undef DEBUG_WINDOW
#undef DEPTH_MODE
