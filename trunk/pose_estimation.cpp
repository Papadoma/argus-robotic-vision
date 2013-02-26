#include "ogre_modeler.hpp"
#include <opencv2/opencv.hpp>

#define rand_num ((double) rand() / (RAND_MAX))

const float w = 0.5;
const float c1 = 1.5;
const float c2 = 1.5;

const int swarm_size = 5;

struct particle_position{
	cv::Mat bones_rotation;
	cv::Point3f model_position;
	cv::Point3f model_rotation;
	float scale;
};

struct particle{
	cv::Mat particle_silhouette;
	cv::Mat particle_depth;

	particle_position current_position;
	particle_position next_position;
	particle_position best_position;
	float best_score;
};

class pose_estimator{
private:

	void load_param();
	void set_modeler_depth();
	cv::Mat calculate_depth_limit();

	void init_particles();
	void calc_evolution();
	void calc_next_position(particle&);
	float calc_score(particle&);

	cv::Mat get_random_bones_rot();
	cv::Point3f get_random_model_rot();

	particle swarm[swarm_size];
	particle_position best_global_position;
	float best_global_score;

	cv::Mat cameraMatrix, Q;
	cv::Mat bone_max, bone_min;
	int startX, startY, endX, endY;		//2D search space for positioning the model
	int frame_width, frame_height;
public:
	ogre_model* model;



	cv::Mat input_frame;
	cv::Mat* output_modeler;
	pose_estimator(int, int);
	~pose_estimator();
};

pose_estimator::pose_estimator(int frame_width, int frame_height)
:frame_width(frame_width),
 frame_height(frame_height)
{
	std::cout << "[Pose Estimator] New Pose estimator: " << frame_width << "x" << frame_height << std::endl;
	load_param();

	input_frame = cv::Mat::zeros(frame_height, frame_width, CV_8UC1);
	cv::Mat input_frameROI = cv::imread("woman_dancing.png",0);

	//Fix the input_frame size to the pose_estimator's size
	startX = (frame_width - input_frameROI.cols)/2;
	startY = (frame_height - input_frameROI.rows)/2;
	endX = startX + input_frameROI.cols;
	endY = startY + input_frameROI.rows;
	input_frameROI.copyTo(input_frame(cv::Rect(startX,startY,input_frameROI.cols,input_frameROI.rows)));

	model = new ogre_model(frame_width,frame_height);
	set_modeler_depth();


	init_particles();
}



pose_estimator::~pose_estimator(){
	cv::destroyAllWindows();
	delete(model);
}

/**
 * Load camera parameters
 */
void pose_estimator::load_param(){
	std::cout << "[Pose Estimator] Loading settings"<< std::endl;
	std::string intrinsics="intrinsics_eye.yml";
	std::string extrinsics="extrinsics_eye.yml";
	std::string bone_limits="bone_limits.yml";

	cv::Size imageSize(frame_width,frame_height);

	cv::FileStorage fs(intrinsics, CV_STORAGE_READ);
	if( fs.isOpened() )
	{
		fs["M1"] >> cameraMatrix;
		fs.release();
	}
	else
		std::cout << "Error: can not load the intrinsic parameters" << std::endl;

	fs.open(extrinsics, CV_STORAGE_READ);
	if( fs.isOpened() )
	{
		fs["Q"] >> Q;
		fs.release();
	}
	else
		std::cout << "Error: can not load the extrinsics parameters" << std::endl;

	fs.open(bone_limits, CV_STORAGE_READ);
	if( fs.isOpened() )
	{
		fs["MinLimits"] >> bone_min;
		fs["MaxLimits"] >> bone_max;
		fs.release();
	}
	else
		std::cout << "Error: can not load the bone limits" << std::endl;
}

/**
 * Sets the modelers depth, so the disparity map of the modeler matches
 * the real disparity map. This means that the 2 world coordinate
 * systems match perfectly.
 */
inline void pose_estimator::set_modeler_depth(){
	std::cout << "[Pose Estimator] Setting model depth"<< std::endl;
	cv::Mat limits = calculate_depth_limit();
	//model->set_depth_limits(limits.at<double>(2),limits.at<double>(1),limits.at<double>(0));
	std::cout<<"Depth set to "<<limits.at<double>(2)<<","<<limits.at<double>(1)<<","<<limits.at<double>(0)<<std::endl;
}

/**
 * Calculates and returns the min, max and center depth value calculated by the stereo rig
 * on the center of the left camera. This is done with respect to the real camera's world
 * coordinate system.
 */
cv::Mat pose_estimator::calculate_depth_limit(){
	std::cout << "[Pose Estimator] Calculating depth limits"<< std::endl;
	cv::Mat result;
	cv::Mat values = cv::Mat(3,1,CV_64FC1);

	cv::Mat disparity_limit = cv::Mat(4,3,CV_8UC1);
	disparity_limit.at<uchar>(0,0) = frame_width/2;
	disparity_limit.at<uchar>(0,1) = frame_width/2;
	disparity_limit.at<uchar>(0,2) = frame_width/2;
	disparity_limit.at<uchar>(1,0) = frame_height/2;
	disparity_limit.at<uchar>(1,1) = frame_height/2;
	disparity_limit.at<uchar>(1,2) = frame_height/2;
	disparity_limit.at<uchar>(2,0) = 1;
	disparity_limit.at<uchar>(2,1) = 16;
	disparity_limit.at<uchar>(2,2) = 32;
	disparity_limit.at<uchar>(3,0) = 1;
	disparity_limit.at<uchar>(3,1) = 1;
	disparity_limit.at<uchar>(3,2) = 1;
	disparity_limit.convertTo(disparity_limit,CV_64FC1);

	result = Q * disparity_limit;
	values.at<double>(0) = result.at<double>(2,0)/result.at<double>(3,0);
	values.at<double>(1) = result.at<double>(2,1)/result.at<double>(3,1);
	values.at<double>(2) = result.at<double>(2,2)/result.at<double>(3,2);
	return values;
}

/**
 * Initializes swarm particles and the best position
 */
void pose_estimator::init_particles(){
	std::cout << "[Pose Estimator] Initializing particles"<< std::endl;
	srand(time(0));
	for(int i = 0; i<swarm_size; i++){
		swarm[i].particle_silhouette = cv::Mat::zeros(frame_height,frame_width,CV_16UC1);

		swarm[i].current_position.bones_rotation = get_random_bones_rot();
		swarm[i].current_position.model_position = cv::Point3f(0,0,300);
		swarm[i].current_position.model_rotation = get_random_model_rot();
		swarm[i].current_position.scale = 10 + rand_num*(100);

		swarm[i].next_position.bones_rotation = get_random_bones_rot();
		swarm[i].next_position.model_position = cv::Point3f(0,0,300);
		swarm[i].next_position.model_rotation = get_random_model_rot();
		swarm[i].next_position.scale = 10 + rand_num*(100);

		swarm[i].best_position.bones_rotation = get_random_bones_rot();
		swarm[i].best_position.model_position = cv::Point3f(0,0,300);
		swarm[i].best_position.model_rotation = get_random_model_rot();
		swarm[i].best_position.scale = 10 + rand_num*(100);

		swarm[i].best_score = 0;
	}
	best_global_position.bones_rotation = cv::Mat::zeros(19,3,CV_32FC1);
	best_global_position.model_position = cv::Point3f(0,0,300);
	best_global_position.model_rotation = cv::Point3f(0,0,0);
	best_global_position.scale = 100;

	best_global_score = 0;
}

/**
 * Calculates random values for the bones based on loaded movement
 * restrictions.
 */
inline cv::Mat pose_estimator::get_random_bones_rot(){
	cv::Mat result = cv::Mat::ones(19,3,CV_32FC1);
	for(int i=0 ; i<19 ; i++){
		for(int j=0 ; j<3 ; j++){
			result.at<float>(i,j) = rand_num * result.at<float>(i,j);
		}
	}
	result = bone_max.mul(result)+bone_min.mul((1-result));
	return result;
}

/**
 * Calculates random values for model global rotation
 */
inline cv::Point3f pose_estimator::get_random_model_rot(){
	float yaw = 0 + rand_num*(360);
	float pitch = 0 + rand_num*(360);
	float roll = 0 + rand_num*(360);
	return cv::Point3f(yaw, pitch, roll);
}

/**
 *Refreshes the scores and thus evolving the swarm
 */
void pose_estimator::calc_evolution(){
	float score;
}

/**
 *Calculates the best next position for the given particle, based on
 *random movement, its best position and the global best one
 */
void pose_estimator::calc_next_position(particle& particle_inst){

}

/**
 *Calculates fitness score
 */
float pose_estimator::calc_score(particle&){

	return 0;
}

int main(){
	pose_estimator instance(640,480);
	cv::namedWindow("input frame");
	cv::namedWindow("modeler output");
	imshow("input frame", instance.input_frame);


	while(1)
	{
		int key = cv::waitKey(1) & 255;;
		if ( key == 27 ) break;	//Esc
		//instance.model->move_model(cv::Point3f(0,0,1000));

		cv::Mat jet_depth_map2;
		cv::applyColorMap(*instance.model->get_depth(), jet_depth_map2, cv::COLORMAP_JET );
		imshow("modeler output", jet_depth_map2);
		imshow("input frame", instance.input_frame);
	}
}

