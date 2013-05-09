#include "pose_estimation.hpp"

int main(){

	pose_estimator* instance;
	instance = new pose_estimator(640,480, 32);
	//cv::Mat input_frame = cv::imread("woman_dancing.png",0);
	//cv::Mat input_frame = cv::imread("test_human2.png",0);
	cv::Mat input_frame = cv::imread("snap_depth2.png",0);
	//cv::Mat input_frame = cv::imread("man_standing.png",0);

	//instance->find_pose(input_frame, false, cv::Point(461,200), cv::Point(291,215));
	//instance->find_pose(input_frame, false);
	//instance->find_pose(input_frame, false, cv::Point(529,157), cv::Point(343,188));
	instance->find_pose(input_frame, false, cv::Point(433,72), cv::Point(230,71));
	//input_frame = cv::imread("woman_dancing2.png",0);
	//instance->find_pose(input_frame, true, cv::Point(459,202), cv::Point(296,219));

	std::cout<< "[Pose Estimator] Pose estimator stopped!"<<std::endl;
	cv::waitKey(0);
	return 0;
}

