#include <opencv.hpp>
#include <stdio.h>
#include <GL/glut.h>
#include <glm/glm.hpp>

#include <OGRE/Ogre.h>
#include <iostream>

//#include <OGRE/SdkTrays.h>
//#include <OGRE/SdkCameraMan.h>

cv::Mat img(480, 640, CV_8UC3);


class ogre_model{
private:
	Ogre::ConfigFile cf;
	Ogre::Root *root;
	Ogre::Camera* camera;
	Ogre::SceneManager* SceneMgr;
	Ogre::SceneNode* modelNode;
	Ogre::Viewport *vp;

	Ogre::TexturePtr mRenderToTexture;

	void setupResources(void);
	void setupPlugins(void);
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

	render_width = 640;
	render_height = 480;

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

void ogre_model::setupPlugins(void){
	//root->loadPlugin("C:/Users/papadoma/ogre_mingw/bin/debug/RenderSystem_GL_d");
	//root->loadPlugin("C:/Users/papadoma/Ogre3D/built/ogre-sdk/bin/debug/Plugin_ParticleFX_d");
	//root->loadPlugin("Plugin_CgProgramManager");
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

	Ogre::ResourceGroupManager::getSingleton().initialiseAllResourceGroups();



	//add mesh
	Ogre::Entity* ogre = SceneMgr->createEntity("ogre", "ninja.mesh");
	ogre->setCastShadows(true);
	// Create a SceneNode and attach the Entity to it
	modelNode = SceneMgr->getRootSceneNode()->createChildSceneNode("OgreNode");
	//Ogre::MaterialPtr material = Ogre::MaterialManager::getSingleton().getByName("DoF_DepthDebug");
	//material->getTechnique(0)->getPass(0)->getTextureUnitState(0)->setTextureName("DoF_Depth");
	//ogre->setMaterialName("DoF_Depth");
	//ogre->setMaterialName("DepthMap");
	createDepthRenderTexture();
	modelNode->attachObject(ogre);

	//ogre->setMaterialName("Examples/Ninja");

	// I move the SceneNode so that it is visible to the camera.
	modelNode->translate(0, -100, -300.0f);


	//set the light
	SceneMgr->setAmbientLight(Ogre::ColourValue(1.0f, 1.0f, 1.0f));

	//	Ogre::Light* pointLight = SceneMgr->createLight("pointLight");
	//	pointLight->setType(Ogre::Light::LT_POINT);
	//	//pointLight->setPosition(Ogre::Vector3(0, 150, 250));
	//	pointLight->setDiffuseColour(0.9, 0.9, 0.9);
	//	pointLight->setSpecularColour(0.9, 0.0, 0.0);

	//make a plane just for fun
	 Ogre::MeshPtr pMesh = Ogre::MeshManager::getSingleton().createPlane("Plane",
			 Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
			 Ogre::Plane(Ogre::Vector3(0.0f, 1.0f, 0.0f), 0.0f),
	                          500.0, 500.0, 100, 100, true, 1, 1, 1, Ogre::Vector3::NEGATIVE_UNIT_Z);
	 Ogre::Entity* pPlaneEntity = SceneMgr->createEntity("PlaneEntity", "Plane");
	 //Ogre::SceneNode* pSceneNodePlane = mSceneMgr->getRootSceneNode()->createChildSceneNode("PlaneNode");
	 modelNode->attachObject(pPlaneEntity);

	vp->getActualDimensions(left, top, width, height);

	mRenderToTexture = Ogre::TextureManager::getSingleton().createManual("RttTex",
			Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
			Ogre::TEX_TYPE_2D,
			render_width,
			render_height,
			0,
			Ogre::PF_R8G8B8,
			Ogre::TU_RENDERTARGET | Ogre::TU_DYNAMIC_WRITE_ONLY_DISCARDABLE);

	Ogre::RenderTexture *renderTexture = mRenderToTexture->getBuffer()->getRenderTarget();
	renderTexture->addViewport(camera);
	renderTexture->getViewport(0)->setClearEveryFrame(true);
	renderTexture->getViewport(0)->setBackgroundColour(Ogre::ColourValue::Black);
	renderTexture->getViewport(0)->setOverlaysEnabled(false);

}
void ogre_model::createDepthRenderTexture(){
	 // Create the depth render texture
	            Ogre::TexturePtr depthTexture = Ogre::TextureManager::getSingleton().createManual(
	                    "NinjaDepthMap", Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
	                    Ogre::TEX_TYPE_2D, camera->getViewport()->getActualWidth(), camera->getViewport()->getActualHeight(),
	                            0, Ogre::PF_FLOAT16_R, Ogre::TU_RENDERTARGET);

	    // Get its render target and add a viewport to it
	            Ogre::RenderTexture* depthTarget = depthTexture->getBuffer()->getRenderTarget();
	            Ogre::Viewport* depthViewport = depthTarget->addViewport(camera, 40);
	            depthViewport->setBackgroundColour(Ogre::ColourValue::Black);

	    // Register 'this' as a render target listener
	           // depthTarget->addListener(this);

	    // Get the technique to use when rendering the depth render texture
	            Ogre::MaterialPtr mDepthMaterial = Ogre::MaterialManager::getSingleton().getByName("DepthMap");
	            mDepthMaterial->load(); // needs to be loaded manually
	            Ogre::Technique* mDepthTechnique = mDepthMaterial->getBestTechnique();

	    // Create a custom render queue invocation sequence for the depth render texture
	            Ogre::RenderQueueInvocationSequence* invocationSequence =
	            		Ogre::Root::getSingleton().createRenderQueueInvocationSequence("DepthMap");

	    // Add a render queue invocation to the sequence, and disable shadows for it
	            Ogre::RenderQueueInvocation* invocation = invocationSequence->add(Ogre::RENDER_QUEUE_MAIN, "main");
	            invocation->setSuppressShadows(true);

	    // Set the render queue invocation sequence for the depth render texture viewport
	            depthViewport->setRenderQueueInvocationSequenceName("DepthMap");

}

inline void ogre_model::render(){
	root->renderOneFrame();
}

inline void ogre_model::move_model(int deg){
	modelNode->yaw(Ogre::Radian(Ogre::Degree(0.2)),Ogre::Node::TS_WORLD);

}

inline void ogre_model::get_opencv_snap(){
	//
	//	//use fast 4-byte alignment (default anyway) if possible
	//	glPixelStorei(GL_PACK_ALIGNMENT, (img.step & 3) ? 1 : 4);
	//
	//	//set length of one complete row in destination data (doesn't need to equal img.cols)
	//	glPixelStorei(GL_PACK_ROW_LENGTH, img.step/img.elemSize());
	//
	//	glReadPixels(0, 0, img.cols, img.rows, GL_RGB, GL_UNSIGNED_BYTE, img.data);
	//
	//	cv::flip(img, img, 0);
	//std::cout<<img;

	mRenderToTexture->getBuffer()->getRenderTarget()->update();
	mRenderToTexture->getBuffer()->getRenderTarget()->copyContentsToMemory(pb, Ogre::RenderTarget::FB_AUTO);
	std::cout<<"texture fps="<<mRenderToTexture->getBuffer()->getRenderTarget()->getAverageFPS()<< std::endl;
	//mRenderToTexture->getBuffer()->getRenderTarget()->getDepthBuffer();

	//	for(int x=0 ; x<render_width ; x++){
	//		for(int y=0 ; y<render_height ; y++){
	//			image_rgb.at<cv::Vec3b>(y,x)[0] = data[(x + y * render_width) * 3];
	//			image_rgb.at<cv::Vec3b>(y,x)[1] = data[(x + y * render_width) * 3 + 1];
	//			image_rgb.at<cv::Vec3b>(y,x)[2] = data[(x + y * render_width) * 3 + 2];
	//		}
	//	}
	//

}

inline cv::Mat* ogre_model::get_rgb(){
	return &image_rgb;
}

int main(){
	ogre_model model;
	model.setup();

	while(!model.Window->isClosed())
	{
		float fps = model.Window->getLastFPS();
		model.move_model(10);

		model.render();

		model.get_opencv_snap();
		//imshow("test",*model.get_rgb());

		std::cout<<"fps="<<fps<< std::endl;

		Ogre::WindowEventUtilities::messagePump();
	}
	std::cout<<"end of program";
	return 1;
}

