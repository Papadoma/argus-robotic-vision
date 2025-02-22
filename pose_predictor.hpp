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

#ifndef POSE_PREDICTOR_HPP
#define POSE_PREDICTOR_HPP

#include <opencv2/opencv.hpp>
#include "ogre_modeler.hpp"

class pose_prediction{
private:
	int choice;
	ogre_model::particle_position position_old;
	bool history_known;

	ogre_model::particle_position naive(const ogre_model::particle_position&);
	ogre_model::particle_position kalman(const ogre_model::particle_position&);

	cv::KalmanFilter* KF;

public:
	pose_prediction(int);
	ogre_model::particle_position predict(const ogre_model::particle_position&);
	void reset_predictor();
};

#endif
