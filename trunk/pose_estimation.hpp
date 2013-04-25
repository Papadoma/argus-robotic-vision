#ifndef POSE_ESTIMATION_HPP
#define POSE_ESTIMATION_HPP

#define NUM_THREADS		4
#if NUM_THREADS > 1
#include <boost/thread.hpp>
#endif

#include "ogre_modeler.hpp"

#include <math.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

#define rand_num ((double) rand() / (RAND_MAX))
#define A 0.8	//0.8
#define N 10
#define MAX_SCORE_CHANGE_COUNT 15	//How many times will the best score stays unchanged before stopping
#define MAX_EVOLS 300				//How many evolutions will cause the search to stop
#define DEBUG_WIND_POSE true
#define DEBUG_COUT_POSE false

const int swarm_size = 50;
const float w = 0.5;
const float c1 = 1.5;
const float c2 = 1.5;

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
#if NUM_THREADS > 1
	boost::mutex best_solution_mutex;
	boost::thread_group thread_group;
#endif

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
	cv::Point3f get_random_model_position(bool);
	cv::Point3f get_random_model_rot(bool);
	cv::Mat get_MinMax_XY(float);

	particle swarm[swarm_size];
	particle_position best_global_position;

	cv::Mat window_boundaries;		//Window boundaries, to avoid the model being moved off the screen

	cv::Mat cameraMatrix, Q;

	cv::Mat bone_max, bone_min;
	cv::Point3f rotation_max, rotation_min;
	float z_max, z_min;

	cv::Point3f human_position;			//Estimated human position;

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
	void calc_evolution();					//Calculate evolution of particles. Returns final score.
	void evaluate_particles(int , int );	//Evaluates particles according to their score

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

#endif
