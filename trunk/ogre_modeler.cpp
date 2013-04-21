#include "ogre_modeler.hpp"



ogre_model::ogre_model(int render_width, int render_height)
:render_width(render_width),
 render_height(render_height)
{
	root = new Ogre::Root("plugins_d.cfg");

	Ogre::Box extents(0, 0, render_width, render_height);

#if DEPTH_MODE == 1
	data = new char [render_width * render_height * Ogre::PixelUtil::getNumElemBytes(Ogre::PF_L8)];
	pb = Ogre::PixelBox(extents, Ogre::PF_L8, data);
	image_depth = cv::Mat(render_height, render_width, CV_8UC1, data);
#elif DEPTH_MODE == 2
	data = new unsigned short [render_width * render_height * Ogre::PixelUtil::getNumElemBytes(Ogre::PF_L16)];
	pb = Ogre::PixelBox(extents, Ogre::PF_L16, data);
	image_depth = cv::Mat(render_height, render_width, CV_16UC1, data);
#else
	data = new float [render_width * render_height * Ogre::PixelUtil::getNumElemBytes(Ogre::PF_FLOAT32_R)];
	pb = Ogre::PixelBox(extents, Ogre::PF_FLOAT32_R, data);
	image_depth = cv::Mat(render_height, render_width, CV_32FC1, data);

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
	Window->setVSyncEnabled(false);

	// create the scene
	Ogre::SceneManager* SceneMgr = root->createSceneManager(Ogre::ST_GENERIC);
	// add a camera
	camera = SceneMgr->createCamera("MainCam");
	camera->yaw(Ogre::Radian(Ogre::Degree(180)));	//Rotate camera because it faces negative z
	camera->setNearClipDistance(1000);
	camera->setFarClipDistance(10000);

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
	renderTexture->getViewport(0)->setBackgroundColour(Ogre::ColourValue::Black);
	renderTexture->getViewport(0)->setOverlaysEnabled(false);
	renderTexture->getViewport(0)->setShadowsEnabled(false);

	Ogre::MaterialPtr mDepthMaterial = Ogre::MaterialManager::getSingleton().getByName("DoF_Depth");
	mDepthMaterial->load(); // needs to be loaded manually
	mDepthTechnique = mDepthMaterial->getBestTechnique();
	body->setMaterial(mDepthMaterial);

	renderToTexture->getBuffer()->getRenderTarget()->update();
	set_depth_limits(-1,-1,-1);
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

void ogre_model::reset_model(){
	reset_bones();
	modelNode->resetToInitialState();
}

void ogre_model::move_model(cv::Point3f position = cv::Point3f(0,0,300.0f), cv::Point3f rotation = cv::Point3f(0,0,0), float scale = 100){
	modelNode->resetToInitialState();
	modelNode->setPosition(position.x, position.y, position.z);
	modelNode->yaw(Ogre::Radian(Ogre::Degree(rotation.x)),Ogre::Node::TS_LOCAL);
	modelNode->pitch(Ogre::Radian(Ogre::Degree(rotation.y)),Ogre::Node::TS_LOCAL);
	modelNode->roll(Ogre::Radian(Ogre::Degree(rotation.z)),Ogre::Node::TS_LOCAL);
	modelNode->setScale(scale,scale,scale);
	//modelNode->setOrientation(angle_w, rot_vector.x, rot_vector.y, rot_vector.z);
	//modelNode->yaw(Ogre::Radian(Ogre::Degree(0.15)),Ogre::Node::TS_LOCAL);
}

void ogre_model::rotate_bones(cv::Mat bones_rotation){
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

cv::Mat ogre_model::get_2D_pos(){
	cv::Mat pos2D = cv::Mat::zeros(5,2,CV_16UC1);

	Ogre::Vector3 head2d = modelNode->_getDerivedPosition() +  modelNode->_getDerivedOrientation() * model_skeleton.Head->_getDerivedPosition()*modelNode->_getDerivedScale();
	Ogre::Vector3 handL2d = modelNode->_getDerivedPosition() + modelNode->_getDerivedOrientation() * model_skeleton.Left_Arm[2]->_getDerivedPosition()*modelNode->_getDerivedScale();
	Ogre::Vector3 handR2d = modelNode->_getDerivedPosition() + modelNode->_getDerivedOrientation() * model_skeleton.Right_Arm[2]->_getDerivedPosition()*modelNode->_getDerivedScale();
	Ogre::Vector3 footL2d = modelNode->_getDerivedPosition() + modelNode->_getDerivedOrientation() * model_skeleton.Left_Leg[2]->_getDerivedPosition()*modelNode->_getDerivedScale();
	Ogre::Vector3 footR2d = modelNode->_getDerivedPosition() + modelNode->_getDerivedOrientation() * model_skeleton.Right_Leg[2]->_getDerivedPosition()*modelNode->_getDerivedScale();


	head2d = camera->getProjectionMatrix()*camera->getViewMatrix()*head2d;
	handL2d = camera->getProjectionMatrix()*camera->getViewMatrix()*handL2d;
	handR2d = camera->getProjectionMatrix()*camera->getViewMatrix()*handR2d;
	footL2d = camera->getProjectionMatrix()*camera->getViewMatrix()*footL2d;
	footR2d = camera->getProjectionMatrix()*camera->getViewMatrix()*footR2d;

	pos2D.at<ushort>(0,0) = (0.5 + head2d.x/2)*render_width;
	pos2D.at<ushort>(0,1) = (0.5 - head2d.y/2)*render_height;
	pos2D.at<ushort>(1,0) = (0.5 + handR2d.x/2)*render_width;
	pos2D.at<ushort>(1,1) = (0.5 - handR2d.y/2)*render_height;
	pos2D.at<ushort>(2,0) = (0.5 + handL2d.x/2)*render_width;
	pos2D.at<ushort>(2,1) = (0.5 - handL2d.y/2)*render_height;
	pos2D.at<ushort>(3,0) = (0.5 + footR2d.x/2)*render_width;
	pos2D.at<ushort>(3,1) = (0.5 - footR2d.y/2)*render_height;
	pos2D.at<ushort>(4,0) = (0.5 + footL2d.x/2)*render_width;
	pos2D.at<ushort>(4,1) = (0.5 - footL2d.y/2)*render_height;

	return pos2D;
}

void ogre_model::get_opencv_snap(){

	renderToTexture->getBuffer()->getRenderTarget()->update();
	setMaxMinDepth();
	renderToTexture->getBuffer()->getRenderTarget()->copyContentsToMemory(pb, Ogre::RenderTarget::FB_AUTO);

#if DEBUG_WINDOW
	render_window();
	std::cout << "[Modeler] Main window fps"<<Window->getLastFPS()<<std::endl;
#endif
	//if(DEBUG_CONSOLE)std::cout<<"[Modeler] texture fps="<<renderToTexture->getBuffer()->getRenderTarget()->getAverageFPS()<< std::endl;
}

cv::Mat* ogre_model::get_depth(){
	get_opencv_snap();
#if DEPTH_MODE == 1
	image_depth = 255 - image_depth;
	threshold(image_depth, image_depth, 254, 255, cv::THRESH_TOZERO_INV);
#elif DEPTH_MODE == 2
	cv::Mat temp;
	image_depth = 65535 - image_depth;
	image_depth.convertTo(temp, CV_32FC1);
	threshold(temp, temp, 65534, 65535, cv::THRESH_TOZERO_INV);
	temp.convertTo(image_depth, CV_16UC1);
#else
	image_depth = 1 - image_depth;
	threshold(image_depth, image_depth, 0.99, 1, cv::THRESH_TOZERO_INV);
#endif
	return &image_depth;
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

