#ifndef OGRE_MODELER_HPP
#define OGRE_MODELER_HPP

#include <opencv2/opencv.hpp>
#include <stdio.h>
//#include <GL/glut.h>
//#include <glm/glm.hpp>

#include <OGRE/Ogre.h>
#include <iostream>

#define DEBUG_CONSOLE true
#define DEBUG_WINDOW false
#define DEPTH_MODE 1		//1: 1 byte, 2: 2 byte, 3: 4 byte

struct skeleton_struct{
	Ogre::Bone*	Head;
	Ogre::Bone*	Torso[6];
	Ogre::Bone*	Right_Arm[3];
	Ogre::Bone* Left_Arm[3];
	Ogre::Bone*	Right_Leg[3];
	Ogre::Bone*	Left_Leg[3];
};

class ogre_model{
private:
	skeleton_struct model_skeleton;
	Ogre::Root *root;
	Ogre::TexturePtr renderToTexture;
	Ogre::SceneNode* modelNode;
	Ogre::Camera* camera;
	Ogre::Entity* body;
	Ogre::Technique* mDepthTechnique;
	Ogre::SkeletonInstance *modelSkeleton;
	Ogre::PixelBox pb;
	Ogre::RenderWindow* Window;
	cv::Mat image_depth;

	void setupResources(void);
	void setupRenderer(void);
	void setup_bones();
	void reset_bones();
	void setMaxMinDepth();
	void render_window(){	root->renderOneFrame();};
	void setup();
	void get_opencv_snap();


	int render_width, render_height;
	float min,max,center; //depth limits

#if DEPTH_MODE == 1
	char *data;

#elif DEPTH_MODE == 2
	unsigned short *data;
#else
	float *data;
#endif

public:
	ogre_model(int, int);
	~ogre_model();

	void set_depth_limits(float min_set, float center_set, float max_set);
	void set_camera_clip(float, float);
	cv::Mat_<cv::Point3f> get_camera_viewspace();

	void reset_model();
	void move_model(cv::Point3f position, cv::Point3f rot_vector, float angle_w);
	void rotate_bones(cv::Mat);

	float get_fps();

	cv::Mat get_2D_pos();
	cv::Mat* get_depth();
};

#endif
