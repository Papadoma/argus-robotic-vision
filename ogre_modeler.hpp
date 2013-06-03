#ifndef OGRE_MODELER_HPP
#define OGRE_MODELER_HPP

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <OGRE/Ogre.h>
#include <iostream>

#define DEBUG_CONSOLE true
#define DEBUG_WINDOW false
#define DEPTH_MODE 1		//1: 1 byte, 2: 2 byte, 3: 4 byte
#define MAX_RENDER_REQUESTS 100

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
	Ogre::RenderWindow* Window;
	cv::Mat* depth_list;
	cv::Mat_<cv::Point>* extremas_list;

//	Ogre::TexturePtr rtt1;
//	Ogre::TexturePtr rtt2;
//	Ogre::TexturePtr rtt3;
//	Ogre::TexturePtr targettex1;
//	Ogre::TexturePtr targettex2;
//	Ogre::TexturePtr targettex3;

	void setupResources(void);
	void setupRenderer(void);
	void setup_bones();
	void reset_bones();
	void setMaxMinDepth();
	void render_window(){	root->renderOneFrame();};
	void setup();
	void getMeshInformation(const Ogre::MeshPtr mesh,
			size_t &vertex_count,
			Ogre::Vector3* &vertices,
			size_t &index_count,
			unsigned* &indices,
			const Ogre::Vector3 &position,
			const Ogre::Quaternion &orient,
			const Ogre::Vector3 &scale);

	void reset_model();
	void move_model(const cv::Point3f& position, const cv::Point3f& rot_vector, const float& angle_w);
	void rotate_bones(const cv::Mat&);
	void get_2D_pos(const int&);

	int render_width, render_height;
	float min,max,center; //depth limits
public:
	ogre_model(int, int);
	~ogre_model();

	struct particle_position{
		cv::Mat bones_rotation;
		cv::Point3f model_position;
		cv::Point3f model_rotation;
		float scale;
	};

	void set_depth_limits(float min_set, float center_set, float max_set);
	void set_camera_clip(float, float);
	cv::Mat_<cv::Point3f> get_camera_viewspace();

	float get_fps();

	cv::Mat* get_depth(const std::vector<particle_position>&);
	cv::Mat_<cv::Point>* get_extremas(){return extremas_list;};
	cv::Mat get_segmentation();
};


#endif
