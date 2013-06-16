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
		position_old.model_position = cur_pos.model_position;
		position_old.model_rotation = cur_pos.model_rotation;
		position_old.bones_rotation = cur_pos.bones_rotation.clone();
		position_old.scale = cur_pos.scale;
		history_known = true;
		return cur_pos;
	}else{
		ogre_model::particle_position pos_pred;
		pos_pred.model_position = (cur_pos.model_position - position_old.model_position) + cur_pos.model_position;
		pos_pred.model_rotation = (cur_pos.model_rotation - position_old.model_rotation) + cur_pos.model_rotation;
		pos_pred.bones_rotation = (cur_pos.bones_rotation - position_old.bones_rotation) + cur_pos.bones_rotation;
		pos_pred.scale = (cur_pos.scale - position_old.scale) + cur_pos.scale;
		position_old = cur_pos;
		return pos_pred;
	}
}

inline ogre_model::particle_position pose_prediction::kalman(const ogre_model::particle_position& new_pos){
	cv::Mat input = cv::Mat::zeros(63,1,CV_32FC1);
	ogre_model::particle_position output;

	input.at<float>(0)=new_pos.model_position.x;
	input.at<float>(1)=new_pos.model_position.y;
	input.at<float>(2)=new_pos.model_position.z;
	input.at<float>(3)=new_pos.model_rotation.x;
	input.at<float>(4)=new_pos.model_rotation.y;
	input.at<float>(5)=new_pos.model_rotation.z;
	cv::Mat bones_array = new_pos.bones_rotation.reshape(1,57).clone();
	bones_array.copyTo(input(cv::Rect(0,6,1,57)));

	cv::Mat prediction = KF->predict();
	KF->correct(input);

	output.model_position.x = prediction.at<float>(0);
	output.model_position.y = prediction.at<float>(1);
	output.model_position.z = prediction.at<float>(2);
	output.model_rotation.x = prediction.at<float>(3);
	output.model_rotation.y = prediction.at<float>(4);
	output.model_rotation.z = prediction.at<float>(5);
	output.bones_rotation = prediction((cv::Rect(0,6,1,57))).clone();
	output.bones_rotation = output.bones_rotation.reshape(1,19).clone();
	return output;
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

