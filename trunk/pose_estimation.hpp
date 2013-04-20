#ifndef POSE_ESTIMATION_HPP
#define POSE_ESTIMATION_HPP

#include "ogre_modeler.hpp"
//#include <boost/thread.hpp>
#include <math.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

#define MAX_THREADS		2
#define rand_num ((double) rand() / (RAND_MAX))
#define A 0.8	//0.8
#define N 20
#define MAX_SCORE_CHANGE_COUNT 30	//How many times will the best score stays unchanged before stopping
#define MAX_EVOLS 300				//How many evolutions will cause the search to stop
#define DEBUG_WIND_POSE true
#define DEBUG_COUT_POSE true

const int swarm_size = 50;
const float w = 0.5;
const float c1 = 1.5;
const float c2 = 1.5;

cv::Point input_head(140,120);
cv::Point input_hand_r(55,137);
cv::Point input_hand_l(224,122);
cv::Point input_foot_r(117,288);
cv::Point input_foot_l(166,282);


struct particle_position{
	cv::Mat bones_rotation;
	cv::Point3f model_position;
	cv::Point3f model_rotation;
	float scale;
};

struct particle{
	int id;
	cv::Mat particle_silhouette;		//Silhouette of particle
	cv::Mat particle_depth;				//Viewable disparity of particle
	cv::Mat extremas;					//Limbs and head 2D position

	cv::Mat bones_violation;
	cv::Mat position_violation;
	cv::Mat rotation_violation;

	particle_position current_position;	//Current particle position
	particle_position next_position;	//Next particle position (velocity)
	particle_position best_position;	//Best position of particle
	double best_score;
	float x;

	bool obsolete;						//Marks whether the particle is reseted completely at least once during evolution
};

class pose_estimator{
private:
	cv::RNG rand_gen;

	void load_param();
	void set_modeler_depth();
	cv::Mat calculate_depth_limit();

	cv::Point3f estimate_starting_position();
	void calc_next_position(particle&);
	void round_position(particle_position&);
	double calc_score(particle&);
	void paint_particle(particle&);

	cv::Mat get_random_bones_rot(bool);
	cv::Mat get_random_19x3_mat(bool);
	cv::Point3f get_random_model_position(bool, float, float);
	cv::Point3f get_random_model_rot(bool);
	cv::Mat get_MinMax_XY(float);

	particle swarm[swarm_size];
	particle_position best_global_position;

	cv::Mat window_boundaries;		//Window boundaries, to avoid the model being moved off the screen

	cv::Mat cameraMatrix, Q;

	cv::Mat bone_max, bone_min;
	cv::Point3f rotation_max, rotation_min;
	float z_max, z_min;

	cv::Point3f human_position;		//At first, the estimated human position;

	bool enable_bones;
	int evolution_num;					//Total number of evolutions
	int score_change_count;				//No of evolutions since last best score calculation
	int startX, startY, endX, endY;		//2D search space for positioning the model
	int frame_width, frame_height;
	int numberOfDisparities;

	cv::Mat_<cv::Point3f> camera_viewspace; //TRN, TLN, BLN, BRN, TRF, TLF, BLF, BRF

	cv::Mat best_global_silhouette;
	cv::Mat best_global_depth;

	enum init_type { POSITION, ROTATION, BONES, SCALE, ALL };

	cv::Rect bounding_box(cv::Mat);
	void init_particles(bool);				//Reset particles by initializing them. Everything is reseted.
	void init_single_particle(particle&, int, bool);
	void calc_evolution();			//Calculate evolution of particles. Returns final score.

public:
	double best_global_score;
	ogre_model* model;

	cv::Mat input_silhouette;			//Silhouette of input image
	cv::Rect input_boundingbox;			//BOunding box of input image
	cv::Mat input_depth;				//Viewable disparity of input image

	pose_estimator(int, int, int);
	~pose_estimator();

	particle_position find_pose(cv::Mat, bool reset);		//Evolve particles based on new frame. If reset flag is set, then it resets everything.
	cv::Mat get_silhouette(){return best_global_silhouette;};
	cv::Mat get_depth(){return best_global_depth;};
	void get_instructions();
	void show_best_solution();
};

pose_estimator::pose_estimator(int frame_width, int frame_height, int noDisparities)
: rotation_max(90,45,45),	//yaw, pitch, roll
  rotation_min(-90,-45,-45),
  frame_width(frame_width),
  frame_height(frame_height),
  numberOfDisparities(noDisparities)
{
	std::cout << "[Pose Estimator] New Pose estimator: " << frame_width << "x" << frame_height << std::endl;
	load_param();

	input_depth = cv::Mat::zeros(frame_height, frame_width, CV_8UC1);
	input_silhouette = cv::Mat::zeros(frame_height, frame_width, CV_8UC1);
	window_boundaries = cv::Mat::zeros(frame_height, frame_width, CV_8UC1);
	cv::rectangle(window_boundaries,cv::Point(1,1),cv::Point(frame_width-2,frame_height-2),cv::Scalar(255),2);

	input_head+= cv::Point(startX,startY);
	input_hand_r+=cv::Point(startX,startY);
	input_hand_l+=cv::Point(startX,startY);
	input_foot_r+=cv::Point(startX,startY);
	input_foot_l+=cv::Point(startX,startY);

	model = new ogre_model(frame_width,frame_height);
	set_modeler_depth();
	camera_viewspace = model->get_camera_viewspace();

	//init_particles(true);
#if DEBUG_WIND_POSE
	cv::namedWindow("input frame");
	cv::namedWindow("best global silhouette");
	cv::namedWindow("best global depth");
	cv::namedWindow("best global diff");
	cv::namedWindow("test");
#endif
}

pose_estimator::~pose_estimator()
{
	cv::destroyAllWindows();
	delete(model);
}

/**
 * Load camera parameters
 */
void pose_estimator::load_param()
{
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
inline void pose_estimator::set_modeler_depth()
{
	std::cout << "[Pose Estimator] Setting model depth"<< std::endl;
	cv::Mat limits = calculate_depth_limit();

	model->set_depth_limits(limits.at<double>(2),limits.at<double>(1),limits.at<double>(0));
	model->set_camera_clip(limits.at<double>(2)/2,2*limits.at<double>(0));
	z_max = limits.at<double>(0);
	z_min = limits.at<double>(2);
	std::cout<<"Depth set to "<<limits.at<double>(2)<<","<<limits.at<double>(1)<<","<<limits.at<double>(0)<<std::endl;
}

/**
 * Calculates and returns the min, max and center depth value calculated by the stereo rig
 * on the center of the left camera. This is done with respect to the real camera's world
 * coordinate system, so its in 'mm'.
 */
cv::Mat pose_estimator::calculate_depth_limit()
{
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
	disparity_limit.at<uchar>(2,2) = numberOfDisparities;
	disparity_limit.at<uchar>(3,0) = 1;
	disparity_limit.at<uchar>(3,1) = 1;
	disparity_limit.at<uchar>(3,2) = 1;
	disparity_limit.convertTo(disparity_limit,CV_64FC1);

	result = Q * disparity_limit;
	values.at<double>(0) = ceil(result.at<double>(2,0)/result.at<double>(3,0));
	values.at<double>(1) = round(result.at<double>(2,1)/result.at<double>(3,1));
	values.at<double>(2) = floor(result.at<double>(2,2)/result.at<double>(3,2));
	return values;
}


/**
 * Initializes swarm particles and the best position. If start_over is true, then everything resets to random.
 * If start_over is false, then only velocities are initiated
 */
void pose_estimator::init_particles(bool start_over)
{
	std::cout << "[Pose Estimator] Initializing/Reseting particles"<< std::endl;
	srand(time(0));
	cv::theRNG().state = time(0);
	evolution_num = 0;
	score_change_count = 0;

	if(start_over){
		enable_bones = false;
		human_position = estimate_starting_position();
		std::cout<< "[Pose Estimator] estimated starting position" << human_position <<std::endl;
		best_global_position.bones_rotation = cv::Mat::zeros(19,3,CV_32FC1);
		best_global_position.model_position = cv::Point3f(human_position.x,human_position.y,human_position.z);
		best_global_position.model_rotation = cv::Point3f(0,0,0);
		best_global_position.scale = 700;
		best_global_score = 0;
		std::cout<< "[Pose Estimator] best solution reseted"<<std::endl;
	}else{
		enable_bones = false;
		particle previous_best_particle;
		previous_best_particle.particle_depth = best_global_depth.clone();			//Last best disparity
		previous_best_particle.particle_silhouette = best_global_silhouette.clone();//Last best silhouette
		best_global_score = calc_score(previous_best_particle);				//Calculate score of last best position on new given frame
	}

	for(int i = 0; i<swarm_size; i++){
		if(start_over)swarm[i].id = i;
		swarm[i].obsolete = false;
		init_single_particle(swarm[i], ALL, !start_over);
	}
}

/**
 * Initializes a single particle. Usefull when initializing for the first time
 * or reseting individual particles.
 * @param	type	POSITION,ROTATION,BONES,SCALE,ALL marks the type of initialization
 * @param	fine	Boolean flag. If true, only the next position gets initialized.
 */
void pose_estimator::init_single_particle(particle& singleParticle, int type, bool fine)
{
	//singleParticle.x=0;					//'x' variable is responsible for forcing the particle to converge to a local solution
	if( type==POSITION || type==ALL ){
		if(!fine)singleParticle.current_position.model_position = get_random_model_position(false, human_position.z*0.85,human_position.z*1.15);
		singleParticle.next_position.model_position = get_random_model_position(true, human_position.z*0.85,human_position.z*1.15);
		if(!fine)singleParticle.best_position.model_position = singleParticle.current_position.model_position;
		singleParticle.position_violation = cv::Mat::ones(3,1,CV_32FC1);
	}

	if( type==ROTATION || type==ALL ){
		if(!fine)singleParticle.current_position.model_rotation = get_random_model_rot(false);
		singleParticle.next_position.model_rotation = get_random_model_rot(true);
		if(!fine)singleParticle.best_position.model_rotation =  singleParticle.current_position.model_rotation;
		singleParticle.rotation_violation = cv::Mat::ones(3,1,CV_32FC1);
	}

	if( type==BONES || type==ALL ){
		if(!fine)singleParticle.current_position.bones_rotation = get_random_bones_rot(false).clone();
		singleParticle.next_position.bones_rotation = get_random_bones_rot(true).clone();
		if(!fine)singleParticle.best_position.bones_rotation = singleParticle.current_position.bones_rotation.clone();
		singleParticle.bones_violation = cv::Mat::ones(19,3,CV_32FC1);
	}

	if( type==SCALE || type==ALL ){
		if(!fine)singleParticle.current_position.scale = round(100 + rand_num*(1000));
		singleParticle.next_position.scale = round(-50 + rand_num*(100));
		if(!fine)singleParticle.best_position.scale = singleParticle.current_position.scale;
	}
	if( type==ALL ){
		singleParticle.x=0;
		singleParticle.best_score = 0;
	}
}

/**
 * Estimate where the human is relative to the camera
 */
cv::Point3f pose_estimator::estimate_starting_position()
{
	cv::Mat result, disparity;
	input_depth.convertTo(disparity, CV_32FC1,numberOfDisparities/255.);
	cv::reprojectImageTo3D(disparity, result, Q);
	cv::Scalar mean_depth = mean(result, input_silhouette);
	return cv::Point3f(-mean_depth.val[0], -mean_depth.val[1], mean_depth.val[2]);
}

/**
 * Calculates random values for the bones based on loaded movement
 * restrictions.
 */
inline cv::Mat pose_estimator::get_random_bones_rot(bool velocity)
{
	if(enable_bones){
		cv::Mat result;
		result = get_random_19x3_mat(velocity).clone();
		if(velocity){
			result = (bone_max-bone_min).mul(result);
		}else{
			result = bone_max.mul(result) + bone_min.mul(1-result);
		}
		return result;
	}else{
		return cv::Mat::zeros(19,3,CV_32FC1);
	}
}

inline cv::Mat pose_estimator::get_random_19x3_mat(bool velocity){
	cv::Mat result = cv::Mat::zeros(19,3,CV_32FC1);
	for(int i=0 ; i<19 ; i++){
		for(int j=0 ; j<3 ; j++){
			if(velocity){
				result.at<float>(i,j) = (float)rand_gen.gaussian(0.08);
				//result.at<float>(i,j) = 2*rand_num - 1;
			}else{
				result.at<float>(i,j) = (float)rand_num;		//uniform distribution
			}
		}
	}
	return result;
}

/**
 * Calculates random values for model position.
 * @param	velocity	Boolean flag. If true, then a gaussian distribution is used instead of a uniform one.
 * @param	min_z		Minimum z distance that the model may be found.
 * @param 	max_z		Maximum z distance that the model may be found.
 */
inline cv::Point3f pose_estimator::get_random_model_position(bool velocity, float min_z = 0 ,float max_z = 0){

	if(min_z == 0 || max_z == 0){
		min_z=camera_viewspace(0,0).z;
		max_z=camera_viewspace(4,0).z;
	}

	cv::Point3f position;
	if(velocity){
		position.z = rand_gen.gaussian(0.3) * (max_z - min_z);
	}else{
		position.z = min_z + rand_num * (max_z - min_z);
	}
	//0:TRN, 1:TLN, 2:BLN, 3:BRN, 4:TRF, 5:TLF, 6:BLF, 7:BRF

	cv::Mat minmax_XY = get_MinMax_XY(position.z);
	float min_x = minmax_XY.at<float>(0,0);
	float max_x = minmax_XY.at<float>(1,0);
	float min_y = minmax_XY.at<float>(2,0);
	float max_y = minmax_XY.at<float>(3,0);

	if(velocity){
		position.x = rand_gen.gaussian(0.1) * (max_x - min_x);	//gaussian distribution
		position.y = rand_gen.gaussian(0.1) * (max_y - min_y);
	}else{
		position.x = min_x + rand_num * (max_x - min_x);	//uniform distribution
		position.y = min_y + rand_num * (max_y - min_y);
	}
	return position;
}

/**
 * Calculates min and max valuse of x and y for given value of z
 */
inline cv::Mat pose_estimator::get_MinMax_XY(float z){
	cv::Mat result = cv::Mat(4,1,CV_32FC1);

	result.at<float>(0,0) = (camera_viewspace(4,0).x - camera_viewspace(0,0).x)/(camera_viewspace(4,0).z-camera_viewspace(0,0).z)*z + (camera_viewspace(0,0).x*camera_viewspace(4,0).z - camera_viewspace(0,0).z*camera_viewspace(4,0).x)/(camera_viewspace(4,0).z-camera_viewspace(0,0).z);	//min_x
	result.at<float>(1,0) = (camera_viewspace(5,0).x - camera_viewspace(1,0).x)/(camera_viewspace(4,0).z-camera_viewspace(0,0).z)*z + (camera_viewspace(1,0).x*camera_viewspace(4,0).z - camera_viewspace(0,0).z*camera_viewspace(5,0).x)/(camera_viewspace(4,0).z-camera_viewspace(0,0).z);	//max_x
	result.at<float>(2,0) = (camera_viewspace(6,0).y - camera_viewspace(2,0).y)/(camera_viewspace(4,0).z-camera_viewspace(0,0).z)*z + (camera_viewspace(2,0).y*camera_viewspace(4,0).z - camera_viewspace(0,0).z*camera_viewspace(6,0).y)/(camera_viewspace(4,0).z-camera_viewspace(0,0).z);	//min_y
	result.at<float>(3,0) = (camera_viewspace(5,0).y - camera_viewspace(1,0).y)/(camera_viewspace(4,0).z-camera_viewspace(0,0).z)*z + (camera_viewspace(1,0).y*camera_viewspace(4,0).z - camera_viewspace(0,0).z*camera_viewspace(5,0).y)/(camera_viewspace(4,0).z-camera_viewspace(0,0).z);	//max_y

	return result;
}

/**
 * Calculates random values for model global rotation
 */
inline cv::Point3f pose_estimator::get_random_model_rot(bool velocity){
	float yaw, pitch, roll;
	if(velocity){
		yaw = rand_gen.gaussian(0.1)*(rotation_max.x - rotation_min.x);
		pitch = rand_gen.gaussian(0.1)*(rotation_max.y - rotation_min.y);
		roll = rand_gen.gaussian(0.1)*(rotation_max.z - rotation_min.z);

	}else{
		yaw = rotation_min.x + rand_num*(rotation_max.x - rotation_min.x);
		pitch =  rotation_min.y + rand_num*(rotation_max.y - rotation_min.y);
		roll =  rotation_min.z + rand_num*(rotation_max.z - rotation_min.z);
	}
	return cv::Point3f(yaw, pitch, roll);
}

/**
 *Refreshes the scores and thus evolving the swarm
 */
void pose_estimator::calc_evolution(){
	bool best_solution_updated;
	score_change_count = 0;
	evolution_num = 0;
	while(evolution_num<MAX_EVOLS && score_change_count<MAX_SCORE_CHANGE_COUNT){
		evolution_num++;
		best_solution_updated = false;
		for(int i=0;i<swarm_size;i++){
			paint_particle(swarm[i]);
		}
		for(int i=0;i<swarm_size;i++){
			double score = calc_score(swarm[i]);
			if(score > swarm[i].best_score){
				swarm[i].best_score = score;
				if(enable_bones)swarm[i].best_position.bones_rotation = swarm[i].current_position.bones_rotation.clone();
				swarm[i].best_position.model_position = swarm[i].current_position.model_position;
				swarm[i].best_position.model_rotation = swarm[i].current_position.model_rotation;
				swarm[i].best_position.scale = swarm[i].current_position.scale;

			}
			if(swarm[i].best_score > best_global_score){
				best_solution_updated = true;
				score_change_count = 0;
				best_global_score = swarm[i].best_score;
				best_global_position = swarm[i].best_position;
			}
		}
		if(!best_solution_updated)score_change_count++;
		int counter = 0;
		for(int i=0;i<swarm_size;i++){
			calc_next_position(swarm[i]);
			if(swarm[i].obsolete)counter++;
		}
		if(counter==swarm_size){
			enable_bones = true;
			for(int i = 0; i<swarm_size; i++){
				init_single_particle(swarm[i], BONES, true);
			}
		}

		if(best_solution_updated){
			model->move_model(best_global_position.model_position,best_global_position.model_rotation, best_global_position.scale);
			model->rotate_bones(best_global_position.bones_rotation);
			best_global_depth = model->get_depth()->clone();
			threshold(best_global_depth,best_global_silhouette,1,255,cv::THRESH_BINARY);
		}
		std::cout<< "[Pose Estimator] best global score: " <<best_global_score << " best global position: " << best_global_position.model_position <<" best global scale: "<<best_global_position.scale <<std::endl;
		//show_best_solution();
#if DEBUG_WIND_POSE
		get_instructions();
		show_best_solution();
#endif
	}
}


/**
 * Call the modeler and get the depth. Also get the silhouette.
 */
inline void pose_estimator::paint_particle(particle& particle_inst){
	model->move_model(particle_inst.current_position.model_position,particle_inst.current_position.model_rotation, particle_inst.current_position.scale);
	model->rotate_bones(particle_inst.current_position.bones_rotation);
	particle_inst.extremas = model->get_2D_pos();
	particle_inst.particle_depth = model->get_depth()->clone();
	threshold(particle_inst.particle_depth,particle_inst.particle_silhouette,1,255,cv::THRESH_BINARY);
}

/**
 *Calculates the best next position for the given particle, based on
 *random movement, its best position and the global best one
 */
void pose_estimator::calc_next_position(particle& particle_inst){
	round_position(particle_inst.current_position);
	cv::Mat particle_inst_old_bones = particle_inst.current_position.bones_rotation.clone();
	cv::Point3f particle_inst_old_position = particle_inst.current_position.model_position;
	cv::Point3f particle_inst_old_rotation = particle_inst.current_position.model_rotation;
	float particle_inst_old_scale = particle_inst.current_position.scale;

	float w1 = A/exp(particle_inst.x);

	particle_inst.next_position.model_position.x = w1*particle_inst.next_position.model_position.x + c1*rand_num*(particle_inst.best_position.model_position.x-particle_inst.current_position.model_position.x) + c2*rand_num*(best_global_position.model_position.x-particle_inst.current_position.model_position.x);
	particle_inst.next_position.model_position.y = w1*particle_inst.next_position.model_position.y + c1*rand_num*(particle_inst.best_position.model_position.y-particle_inst.current_position.model_position.y) + c2*rand_num*(best_global_position.model_position.y-particle_inst.current_position.model_position.y);
	particle_inst.next_position.model_position.z = w1*particle_inst.next_position.model_position.z + c1*rand_num*(particle_inst.best_position.model_position.z-particle_inst.current_position.model_position.z) + c2*rand_num*(best_global_position.model_position.z-particle_inst.current_position.model_position.z);
	particle_inst.next_position.model_rotation.x = w1*particle_inst.next_position.model_rotation.x + c1*rand_num*(particle_inst.best_position.model_rotation.x-particle_inst.current_position.model_rotation.x) + c2*rand_num*(best_global_position.model_rotation.x-particle_inst.current_position.model_rotation.x);
	particle_inst.next_position.model_rotation.y = w1*particle_inst.next_position.model_rotation.y + c1*rand_num*(particle_inst.best_position.model_rotation.y-particle_inst.current_position.model_rotation.y) + c2*rand_num*(best_global_position.model_rotation.y-particle_inst.current_position.model_rotation.y);
	particle_inst.next_position.model_rotation.z = w1*particle_inst.next_position.model_rotation.z + c1*rand_num*(particle_inst.best_position.model_rotation.z-particle_inst.current_position.model_rotation.z) + c2*rand_num*(best_global_position.model_rotation.z-particle_inst.current_position.model_rotation.z);
	if(enable_bones)particle_inst.next_position.bones_rotation = w1*particle_inst.next_position.bones_rotation + c1*get_random_19x3_mat(false).mul(particle_inst.best_position.bones_rotation-particle_inst.current_position.bones_rotation) + c2*get_random_19x3_mat(false).mul(best_global_position.bones_rotation-particle_inst.current_position.bones_rotation);
	particle_inst.next_position.scale = w1*particle_inst.next_position.scale + c1*rand_num*(particle_inst.best_position.scale-particle_inst.current_position.scale) + c2*rand_num*(best_global_position.scale-particle_inst.current_position.scale);

	particle_inst.current_position.model_position += particle_inst.next_position.model_position;
	particle_inst.current_position.model_rotation += particle_inst.next_position.model_rotation;
	if(enable_bones)add(particle_inst.current_position.bones_rotation,particle_inst.next_position.bones_rotation,particle_inst.current_position.bones_rotation);
	particle_inst.current_position.scale += particle_inst.next_position.scale;

	//Place the new position within limits
	cv::Mat minmax_XY = get_MinMax_XY(particle_inst.current_position.model_position.z);
	if(particle_inst.current_position.model_position.x < minmax_XY.at<float>(0,0)){
		particle_inst.current_position.model_position.x = minmax_XY.at<float>(0,0);
		particle_inst.next_position.model_position.x = -0.4*particle_inst.next_position.model_position.x;
		particle_inst.position_violation.at<float>(0,0) = 0;
	}else if(particle_inst.current_position.model_position.x > minmax_XY.at<float>(1,0)){
		particle_inst.current_position.model_position.x = minmax_XY.at<float>(1,0);
		particle_inst.next_position.model_position.x = -0.4*particle_inst.next_position.model_position.x;
		particle_inst.position_violation.at<float>(0,0) = 0;
	}else{
		particle_inst.position_violation.at<float>(0,0) = 1;
	}
	if(particle_inst.current_position.model_position.y < minmax_XY.at<float>(2,0)){
		particle_inst.current_position.model_position.y = minmax_XY.at<float>(2,0);
		particle_inst.next_position.model_position.y = -0.4*particle_inst.next_position.model_position.y;
		particle_inst.position_violation.at<float>(1,0) = 0;
	}else if(particle_inst.current_position.model_position.y > minmax_XY.at<float>(3,0)){
		particle_inst.current_position.model_position.y = minmax_XY.at<float>(3,0);
		particle_inst.next_position.model_position.y = -0.4*particle_inst.next_position.model_position.y;
		particle_inst.position_violation.at<float>(1,0) = 0;
	}else{
		particle_inst.position_violation.at<float>(1,0) = 1;
	}

	if(particle_inst.current_position.model_position.z < z_min){
		particle_inst.current_position.model_position.z = z_min;
		particle_inst.next_position.model_position.z = -0.4*particle_inst.next_position.model_position.z;
		particle_inst.position_violation.at<float>(2,0) = 0;
	}else if(particle_inst.current_position.model_position.z > z_max){
		particle_inst.current_position.model_position.z = z_max;
		particle_inst.next_position.model_position.z = -0.4*particle_inst.next_position.model_position.z;
		particle_inst.position_violation.at<float>(2,0) = 0;
	}else{
		particle_inst.position_violation.at<float>(2,0) = 1;
	}

	//Place the new rotation within limits
	if(particle_inst.current_position.model_rotation.x > rotation_max.x){	//yaw
		particle_inst.current_position.model_rotation.x = rotation_max.x;
		particle_inst.next_position.model_rotation.x = -0.4*particle_inst.next_position.model_rotation.x;
		particle_inst.rotation_violation.at<float>(0,0) = 0;
	}else if(particle_inst.current_position.model_rotation.x < rotation_min.x){
		particle_inst.current_position.model_rotation.x = rotation_min.x;
		particle_inst.next_position.model_rotation.x = -0.4*particle_inst.next_position.model_rotation.x;
		particle_inst.rotation_violation.at<float>(0,0) = 0;
	}else{
		particle_inst.rotation_violation.at<float>(0,0) = 1;
	}

	if(particle_inst.current_position.model_rotation.y > rotation_max.y){	//pitch
		particle_inst.current_position.model_rotation.y = rotation_max.y;
		particle_inst.next_position.model_rotation.y = -0.4*particle_inst.next_position.model_rotation.y;
		particle_inst.rotation_violation.at<float>(1,0) = 0;
	}else if(particle_inst.current_position.model_rotation.y < rotation_min.y){
		particle_inst.current_position.model_rotation.y = rotation_min.y;
		particle_inst.next_position.model_rotation.y = -0.4*particle_inst.next_position.model_rotation.y;
		particle_inst.rotation_violation.at<float>(1,0) = 0;
	}else{
		particle_inst.rotation_violation.at<float>(1,0) = 1;
	}

	if(particle_inst.current_position.model_rotation.z > rotation_max.z){	//roll
		particle_inst.current_position.model_rotation.z = rotation_max.z;
		particle_inst.next_position.model_rotation.z = -0.4*particle_inst.next_position.model_rotation.z;
		particle_inst.rotation_violation.at<float>(2,0) = 0;
	}else if(particle_inst.current_position.model_rotation.z < rotation_min.z){
		particle_inst.current_position.model_rotation.z = rotation_min.z;
		particle_inst.next_position.model_rotation.z = -0.4*particle_inst.next_position.model_rotation.z;
		particle_inst.rotation_violation.at<float>(2,0) = 0;
	}else{
		particle_inst.rotation_violation.at<float>(2,0) = 1;
	}

	//Place the new bones rotation within limits
	cv::Mat result = particle_inst.current_position.bones_rotation.clone();
	cv::min(particle_inst.current_position.bones_rotation,bone_max,particle_inst.current_position.bones_rotation);
	cv::max(particle_inst.current_position.bones_rotation,bone_min,particle_inst.current_position.bones_rotation);
	particle_inst.bones_violation = (result == particle_inst.current_position.bones_rotation);	//Check if before and after limiting
	particle_inst.bones_violation.convertTo(particle_inst.bones_violation, CV_32FC1,(float)1.3/255);

	//Test if there where any violations/ FOR DEBUGGING PURPOSES
	//	if (57-cv::countNonZero(particle_inst.bones_violation)){
	//		std::cout << "[Pose Estimator]: Bone violation fixed on particle: "<<particle_inst.id<<" for No of values: "<< 57-cv::countNonZero(particle_inst.bones_violation)<<std::endl;
	//	}
	//	if (3-cv::countNonZero(particle_inst.position_violation)){
	//		std::cout << "[Pose Estimator]: Position violation fixed on particle: "<<particle_inst.id<<" for No of values: "<< 3-cv::countNonZero(particle_inst.position_violation)<<std::endl;
	//	}
	//	if (3-cv::countNonZero(particle_inst.rotation_violation)){
	//		std::cout << "[Pose Estimator]: Rotation violation fixed on particle: "<<particle_inst.id<<" for No of values: "<< 3-cv::countNonZero(particle_inst.rotation_violation)<<std::endl;
	//	}
	particle_inst.bones_violation=particle_inst.bones_violation-0.3;
	particle_inst.next_position.bones_rotation = particle_inst.next_position.bones_rotation.mul(particle_inst.bones_violation);

	//Round current_position, since its float
	round_position(particle_inst.current_position);

	//Reset particles, if they have settled on some dimensions and on the best_global_position
	if( particle_inst.current_position.model_position == particle_inst_old_position){
#if DEBUG_COUT_POSE
		std::cout<< "Particle " << particle_inst.id <<" resets: POSITION"<< std::endl;
#endif
		init_single_particle(particle_inst,POSITION,true);
	}
	if( particle_inst.current_position.model_rotation == particle_inst_old_rotation){
#if DEBUG_COUT_POSE
		std::cout<< "Particle " << particle_inst.id <<" resets: ROTATION"<< std::endl;
#endif
		init_single_particle(particle_inst,ROTATION,true);
	}
	if(enable_bones && (cv::countNonZero(particle_inst.current_position.bones_rotation == particle_inst_old_bones)==57)){
#if DEBUG_COUT_POSE
		std::cout<< "Particle " << particle_inst.id <<" resets: BONES"<< std::endl;
#endif
		init_single_particle(particle_inst,BONES,true);
	}
	if( particle_inst.current_position.scale == particle_inst_old_scale){
#if DEBUG_COUT_POSE
		std::cout<< "Particle " << particle_inst.id <<" resets: SCALE"<< std::endl;
#endif
		init_single_particle(particle_inst,SCALE,true);
	}

	if(particle_inst.x<log(10*A)){
		particle_inst.x+=log(10*A)/N;
	}else{
#if DEBUG_COUT_POSE
		std::cout<< "Particle " << particle_inst.id <<" resets: ALL "<< std::endl;
#endif
		init_single_particle(particle_inst,ALL,true);
		particle_inst.obsolete = true;
	}
}

inline void pose_estimator::round_position(particle_position& position){
	position.model_position.x = floor(position.model_position.x + 0.5);
	position.model_position.y = floor(position.model_position.y + 0.5);
	position.model_position.z = floor(position.model_position.z + 0.5);
	position.model_rotation.x = floor(position.model_rotation.x + 0.5);
	position.model_rotation.y = floor(position.model_rotation.y + 0.5);
	position.model_rotation.z = floor(position.model_rotation.z + 0.5);
	position.bones_rotation.convertTo(position.bones_rotation, CV_32SC1);
	position.bones_rotation.convertTo(position.bones_rotation, CV_32FC1);
	position.scale = floor(position.scale + 0.5);
}

/**
 *Calculates fitness score
 */
inline double pose_estimator::calc_score(particle& particle_inst){
	cv::Mat result_or,result_and,result_xor;
	bitwise_and(input_silhouette, particle_inst.particle_silhouette, result_and);
	bitwise_or(input_silhouette, particle_inst.particle_silhouette, result_or);
	bitwise_xor(input_silhouette, particle_inst.particle_silhouette, result_xor);

	//	float dist1 = 1./(1 + sqrt(pow((particle_inst.extremas.at<ushort>(0,0)-input_head.x),2) + pow((particle_inst.extremas.at<ushort>(0,1)-input_head.y),2)));
	//	float dist2 = 1./(1 + sqrt(pow((particle_inst.extremas.at<ushort>(1,0)-input_hand_r.x),2) + pow((particle_inst.extremas.at<ushort>(1,1)-input_hand_r.y),2)));
	//	float dist3 = 1./(1 + sqrt(pow((particle_inst.extremas.at<ushort>(2,0)-input_hand_l.x),2) + pow((particle_inst.extremas.at<ushort>(2,1)-input_hand_l.y),2)));
	//		float dist4 = 1./(1 + sqrt(pow((particle_inst.extremas.at<ushort>(3,0)-input_foot_r.x),2) + pow((particle_inst.extremas.at<ushort>(3,1)-input_foot_r.y),2)));
	//		float dist5 = 1./(1 + sqrt(pow((particle_inst.extremas.at<ushort>(4,0)-input_foot_l.x),2) + pow((particle_inst.extremas.at<ushort>(4,1)-input_foot_l.y),2)));
	cv::Mat window_boundaries_violation;
	bitwise_and(particle_inst.particle_silhouette,window_boundaries,window_boundaries_violation);

	if(countNonZero(result_and) && (countNonZero(window_boundaries_violation)==0)){
		cv::Mat result;
		cv::absdiff(particle_inst.particle_depth,input_depth,result);
		//				cv::Mat A1_A2 ;
		//				cv::Mat A1 = cv::Mat::zeros(frame_height,frame_width,CV_8UC1);
		//				cv::Mat A2 = cv::Mat::zeros(frame_height,frame_width,CV_8UC1);
		//				cv::rectangle(A1,input_boundingbox,cv::Scalar(255),CV_FILLED);
		//				cv::rectangle(A2,bounding_box(particle_inst.particle_silhouette),cv::Scalar(255),CV_FILLED);
		//				cv::bitwise_xor(A1,A2,A1_A2);
		//
		//		cv::bitwise_xor(A1,A2,A1_A2);
		//		imshow("A1",A1);
		//		imshow("A2",A2);
		//		cv::waitKey(0);
		cv::Scalar mean,stddev;
		meanStdDev(result, mean, stddev);
		//cv::bitwise_and(result_and,result,result);
		//return 0.4* (0.2*dist1 + 0.2*dist2 + 0.2*dist3 + 0.2*dist4 + 0.2*dist5) + 0.6 * 1./(1+cv::mean(result).val[0])  * countNonZero(result_and)/fmaxf(countNonZero(input_silhouette),countNonZero(particle_inst.particle_silhouette));
		//return 0.5* 1./(1+cv::mean(result).val[0])+ 0.5* (float)countNonZero(result_and)/countNonZero(result_or);
		//return  1./(1+cv::mean(result,particle_inst.particle_silhouette).val[0]) *(float)countNonZero(result_and)/countNonZero(result_or);
		//return 1./(1+mean.val[0]*stddev.val[0]) *(double)countNonZero(result_and)/countNonZero(result_or);
		return 3*1./(1+mean.val[0]*stddev.val[0]) *(double)countNonZero(result_and)/(1+countNonZero(result_xor))*(double)countNonZero(result_and)/countNonZero(result_or);

	}else{
		return 0;
	}
}

/**
 * Returns the bounding box of a silhouette.
 */
cv::Rect pose_estimator::bounding_box(cv::Mat input){
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours(input.clone(), contours, hierarchy, CV_RETR_EXTERNAL,  CV_CHAIN_APPROX_SIMPLE);
	return  cv::boundingRect( contours[0] );
}

void pose_estimator::show_best_solution(){
	imshow("input frame", input_depth);
	cv::Mat result, resullt2 = cv::Mat::zeros(640,480,CV_8UC1);
	bitwise_xor(input_silhouette, best_global_silhouette, result);
	imshow("best global silhouette", best_global_silhouette);
	cv::Mat jet_depth_map2;
	cv::applyColorMap(best_global_depth, jet_depth_map2, cv::COLORMAP_JET );

	cv::Mat extremas = model->get_2D_pos();
	for(int i=0;i<5;i++){
		cv::circle(jet_depth_map2,cv::Point(extremas.at<ushort>(i,0),extremas.at<ushort>(i,1)),3,cv::Scalar(0,255,255),2);
		cv::circle(jet_depth_map2,cv::Point(extremas.at<ushort>(i,0),extremas.at<ushort>(i,1)),1,cv::Scalar(0,0,255),2);
	}
	imshow("best global depth", best_global_depth);
	imshow("best global diff", result);
	imshow("test",swarm[0].particle_silhouette);
}

void pose_estimator::get_instructions(){
	int key = cv::waitKey(1) & 255;
	if ( key == 27 ) exit(1);	//Esc
	if ( key == 114 ) { //R
		init_particles(true);
	}
}

particle_position pose_estimator::find_pose(cv::Mat disparity_frame, bool reset)
{
	disparity_frame.copyTo(input_depth);
	threshold(disparity_frame,input_silhouette,1,255,cv::THRESH_BINARY);
	init_particles(reset);
	calc_evolution();
	return best_global_position;
}
#endif
