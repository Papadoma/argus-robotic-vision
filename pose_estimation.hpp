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

#pragma once
#ifndef POSE_ESTIMATION_HPP
#define POSE_ESTIMATION_HPP

#define NUM_THREADS		4//4
#define USE_GPU			false
#if NUM_THREADS > 1
#include <boost/thread.hpp>
#endif

#include "ogre_modeler.hpp"
#include <opencv2/ocl/ocl.hpp>
#include "edges_distance.hpp"
#include "pose_predictor.hpp"

#include <math.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

//#define rand_num ((double) rand() / (RAND_MAX))
#define A 0.8f	//0.8
#define N 12
#define MAX_EVOLS 40				//How many evolutions will cause the search to stop
#define DEBUG_WIND_POSE true
#define DEBUG_COUT_POSE false

const int swarm_size = 20;
const float w = 0.5;
const float c1 = 1.5;//1.5
const float c2 = 1.5;//1.5

struct particle{
	int id;
	//cv::Mat particle_silhouette;		//Silhouette of particle
	cv::Mat particle_depth;				//Viewable disparity of particle
	bool reset_array[4];

	cv::Mat bones_violation;
//	cv::Mat position_violation;
//	cv::Mat rotation_violation;
	cv::Mat_<cv::Point> extremas;

	ogre_model::particle_position current_position;	//Current particle position
	ogre_model::particle_position next_position;	//Next particle position (velocity)
	ogre_model::particle_position best_position;	//Best position of particle
	double best_score;
	float x;

	bool obsolete;						//Marks whether the particle is reseted completely at least once during evolution
};

class pose_estimator{
private:
#if NUM_THREADS > 1
	//boost::mutex best_solution_mutex;
	boost::mutex obsolete_counter_mutex;
	boost::mutex human_position_mutex;
	boost::thread_group thread_group;
#endif

	cv::RNG rand_gen;
	edge_similarity edge_estimator;
	pose_prediction* pose_predictor;

	void load_param();
	void set_modeler_depth();
	cv::Mat calculate_depth_limit();

	cv::Point3f estimate_starting_position();
	void calc_next_position(particle&);
	void round_position(ogre_model::particle_position&);
	double calc_score(const particle&);

	cv::Mat get_random_bones_rot(bool);
	cv::Mat get_random_19x3_mat(bool);
	cv::Point3f get_random_model_position(bool);
	cv::Point3f get_random_model_rot(bool);
	cv::Mat get_MinMax_XY(float);

	particle swarm[swarm_size];
	ogre_model::particle_position best_global_position;
	ogre_model::particle_position best_global_position_predicted;

	cv::Mat window_boundaries;		//Window boundaries, to avoid the model being moved off the screen

	cv::Mat cameraMatrix, Q;

	cv::Mat bone_max, bone_min;
	cv::Point3f rotation_max, rotation_min;
	float z_max, z_min;

	cv::Point3f human_position;			//Estimated human position;
	cv::Point	left_hand;
	cv::Point	right_hand;
	cv::Rect user_bounds;

	bool enable_bones;

	int evolution_num;					//Total number of evolutions
	int score_change_count;				//No of evolutions since last best score calculation
	int startX, startY, endX, endY;		//2D search space for positioning the model
	int frame_width, frame_height;
	int numberOfDisparities;
	int obsolete_counter;

	cv::Mat_<cv::Point3f> camera_viewspace; //TRN, TLN, BLN, BRN, TRF, TLF, BLF, BRF

	//cv::Mat best_global_silhouette;
	cv::Mat best_global_depth;
	cv::Mat_<cv::Point> best_global_extremas;

	enum init_type { POSITION, ROTATION, BONES, SCALE, ALL };

	void init_particles();				//Reset particles by initializing them. Everything is reseted.
	void init_single_particle(particle&, int, bool);
	void calc_evolution();					//Calculate evolution of particles. Returns final score.
	void evaluate_particles(int , int );	//Evaluates particles according to their score

	int MAX_SCORE_CHANGE_COUNT;	//How many times will the best score stays unchanged before stopping
public:
	double best_global_score;
	ogre_model* model;

	cv::Rect input_boundingbox;			//BOunding box of input image
	cv::Mat input_depth;				//Viewable disparity of input image
	cv::Mat input_frame;				//Input frame
	cv::Mat distance_penalty;			//Distance transform based penalty

	pose_estimator(int, int, int);
	~pose_estimator();
	enum operation_mode {FIRST_RUN, SEARCH, TRACK};
	pose_estimator::operation_mode mode;

	particle find_pose(cv::Mat, pose_estimator::operation_mode , cv::Rect, cv::Point left_marker=cv::Point(-1,-1), cv::Point right_marker=cv::Point(-1,-1));		//Evolve particles based on new frame. If reset flag is set, then it resets everything.
	//cv::Mat get_silhouette(){return best_global_silhouette;};
	cv::Mat get_depth(){return best_global_depth;};
	void get_instructions();
	void show_best_solution();
	void set_human_position(cv::Point3f est_pos){human_position=est_pos;};
	cv::Rect get_model_bound_rect(){
		cv::Rect box = cv::boundingRect(best_global_extremas);
//		box.x -= box.width/6;
//		box.width += box.width/3;
//		box.y -= box.height/6;
//		box.height += box.height/3;
//		box &= cv::Rect(0,0,frame_width, frame_height);
		return box;
	};
};

#endif
