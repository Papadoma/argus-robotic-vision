#include "pose_estimation.hpp"

int main(){
	pose_estimator* instance = new pose_estimator(640,480, 32);

//	cv::namedWindow("input frame");
//	cv::namedWindow("best global silhouette");
//	cv::namedWindow("best global depth");
//	cv::namedWindow("best global diff");
//	cv::namedWindow("test");

	//		cv::cvtColor(test,test,CV_GRAY2RGB);
	//		cv::circle(test,input_head,3,cv::Scalar(0,0,255),2);
	//		cv::circle(test,input_hand_r,3,cv::Scalar(0,0,255),2);
	//		cv::circle(test,input_hand_l,3,cv::Scalar(0,0,255),2);
	//		cv::circle(test,input_foot_r,3,cv::Scalar(0,0,255),2);
	//		cv::circle(test,input_foot_l,3,cv::Scalar(0,0,255),2);

	cv::Mat input_frame = cv::imread("woman_dancing.png",0);
	instance->find_pose(input_frame, true);
	input_frame = cv::imread("woman_dancing2.png",0);
	instance->find_pose(input_frame, false);

	std::cout<< "[Pose Estimator] Pose estimator stopped!"<<std::endl;
	cv::waitKey(0);
delete(instance);
	return 0;
}

