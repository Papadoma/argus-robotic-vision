#include "pose_estimation.hpp"

int main(){

	pose_estimator* instance;
	instance = new pose_estimator(640,480, 32);
	//cv::Mat input_frame = cv::imread("woman_dancing.png",0);
	cv::Mat input_frame = cv::imread("snap_depth3.png",0);
	//cv::Mat input_frame = cv::imread("test_human2.png",0);
	//cv::Mat input_frame = cv::imread("snap_depth2.png",0);
	//cv::Mat input_frame = cv::imread("man_standing.png",0);

	int key = 'r';
	while(key == 'r'){
		//instance->find_pose(input_frame, false, cv::Point(461,200), cv::Point(291,215));
		//instance->find_pose(input_frame, false, cv::Point(433,79), cv::Point(226,83));

		instance->find_pose(input_frame, false, cv::Rect(194,36,263,419));
		//instance->find_pose(input_frame, true, cv::Rect(194,36,263,419));
		//instance->find_pose(input_frame, false, cv::Point(435,88) , cv::Point(223,92) );
		//instance->find_pose(input_frame, true, cv::Point(424,149) , cv::Point(222,158) );
		//instance->find_pose(input_frame, true, cv::Point(424,149) , cv::Point(222,158) );
		//instance->find_pose(input_frame, true, cv::Point(433,67), cv::Point(230,67));
		//instance->find_pose(input_frame, true, cv::Point(433,67), cv::Point(230,67));

		//instance->find_pose(input_frame, false, cv::Point(529,157), cv::Point(343,188));

		//input_frame = cv::imread("woman_dancing2.png",0);
		//instance->find_pose(input_frame, true, cv::Point(459,202), cv::Point(296,219));
		key = cv::waitKey(0) & 255;
	}

	std::cout<< "[Pose Estimator] Pose estimator stopped!"<<std::endl;
	return 0;
}

