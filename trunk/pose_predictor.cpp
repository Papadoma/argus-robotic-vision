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

#include "pose_predictor.hpp"

pose_prediction::pose_prediction(int choice){
	history_known = false;
	this->choice = choice;

	if(choice == 1){
		KF = new cv::KalmanFilter(126, 63, 0);
		for(int i=0;i<126;i++){
			KF->transitionMatrix.at<float>(i,i)=1;
			if(i+63<126)KF->transitionMatrix.at<float>(i,i+63)=1;
		}
		setIdentity(KF->measurementMatrix);
		setIdentity(KF->processNoiseCov, cv::Scalar::all(0.0001));
		setIdentity(KF->measurementNoiseCov, cv::Scalar::all(0.01));
		setIdentity(KF->errorCovPost, cv::Scalar::all(0.05));
	}
}

inline ogre_model::particle_position pose_prediction::naive(const ogre_model::particle_position& cur_pos){
	if(!history_known){
		//set old pose as current one and return current pose as predicted
		position_old.model_position = cur_pos.model_position;
		position_old.model_rotation = cur_pos.model_rotation;
		position_old.bones_rotation = cur_pos.bones_rotation.clone();
		position_old.scale = cur_pos.scale;
		history_known = true;
		return cur_pos;
	}else{
		//predict new pose using a naive method
		ogre_model::particle_position pos_pred;
		pos_pred.model_position = (cur_pos.model_position - position_old.model_position) + cur_pos.model_position;
		pos_pred.model_rotation = (cur_pos.model_rotation - position_old.model_rotation) + cur_pos.model_rotation;
		pos_pred.bones_rotation = (cur_pos.bones_rotation - position_old.bones_rotation) + cur_pos.bones_rotation;
		pos_pred.scale = (cur_pos.scale - position_old.scale) + cur_pos.scale;
		position_old = cur_pos;
		return pos_pred;
	}
}

inline ogre_model::particle_position pose_prediction::kalman(const ogre_model::particle_position& cur_pos){
	cv::Mat input = cv::Mat::zeros(63,1,CV_32FC1);
	if(!history_known){
		//Pass current pose as history
		position_old.model_position = cur_pos.model_position;
		position_old.model_rotation = cur_pos.model_rotation;
		position_old.bones_rotation = cur_pos.bones_rotation.clone();
		position_old.scale = cur_pos.scale;
		history_known = true;

		//Correct kalman's prediction
		input.at<float>(0)=cur_pos.model_position.x;
		input.at<float>(1)=cur_pos.model_position.y;
		input.at<float>(2)=cur_pos.model_position.z;
		input.at<float>(3)=cur_pos.model_rotation.x;
		input.at<float>(4)=cur_pos.model_rotation.y;
		input.at<float>(5)=cur_pos.model_rotation.z;
		cur_pos.bones_rotation.reshape(1,57).copyTo(input(cv::Rect(0,6,1,57)));
		KF->correct(input);

		//return current pose, as we have no other prediction
		return cur_pos;
	}else{
		//calculate a posible pose, using the naive method
		ogre_model::particle_position pos_corr;
		pos_corr.model_position = (cur_pos.model_position - position_old.model_position) + cur_pos.model_position;
		pos_corr.model_rotation = (cur_pos.model_rotation - position_old.model_rotation) + cur_pos.model_rotation;
		pos_corr.bones_rotation = (cur_pos.bones_rotation - position_old.bones_rotation) + cur_pos.bones_rotation;
		pos_corr.scale = (cur_pos.scale - position_old.scale) + cur_pos.scale;
		position_old = cur_pos;

		//Correct kalman filter, after predicting the new pose
		input.at<float>(0)=pos_corr.model_position.x;
		input.at<float>(1)=pos_corr.model_position.y;
		input.at<float>(2)=pos_corr.model_position.z;
		input.at<float>(3)=pos_corr.model_rotation.x;
		input.at<float>(4)=pos_corr.model_rotation.y;
		input.at<float>(5)=pos_corr.model_rotation.z;
		pos_corr.bones_rotation.reshape(1,57).copyTo(input(cv::Rect(0,6,1,57)));

		KF->predict();								//first predict
		cv::Mat estimation = KF->correct(input);	//then correct

		//return kalman's pose prediction
		ogre_model::particle_position output;
		output.model_position.x = estimation.at<float>(0);
		output.model_position.y = estimation.at<float>(1);
		output.model_position.z = estimation.at<float>(2);
		output.model_rotation.x = estimation.at<float>(3);
		output.model_rotation.y = estimation.at<float>(4);
		output.model_rotation.z = estimation.at<float>(5);
		output.bones_rotation = estimation((cv::Rect(0,6,1,57))).reshape(1,19).clone();
		return output;
	}
}

ogre_model::particle_position pose_prediction::predict(const ogre_model::particle_position& cur_pos){
	switch(choice){
	case 0:
		return naive(cur_pos);
	case 1:
		return kalman(cur_pos);
	}
}

void pose_prediction::reset_predictor(){
	history_known = false;
}

