#include "pose_estimation.hpp"

int main(){

	pose_estimator* instance;
	instance = new pose_estimator(640,480, 32);
	cv::Mat input_frame = cv::imread("woman_dancing.png",0);
	//cv::Mat input_frame = cv::imread("man_standing.png",0);

	instance->find_pose(input_frame, true);
	//input_frame = cv::imread("woman_dancing2.png",0);
	//instance->find_pose(input_frame, false);

	std::cout<< "[Pose Estimator] Pose estimator stopped!"<<std::endl;
	cv::waitKey(0);
	return 0;
}

