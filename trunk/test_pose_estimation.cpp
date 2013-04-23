#include "pose_estimation.hpp"



class test{
private:
	pose_estimator* instance;
public:
	test();
	~test();
	void use_it();
};
test::test(){
	instance = new pose_estimator(640,480, 32);
}

test::~test(){
	delete(instance);
}


void test::use_it(){

	cv::Mat input_frame = cv::imread("woman_dancing.png",0);

		instance->find_pose(input_frame, true);
		input_frame = cv::imread("woman_dancing2.png",0);
		instance->find_pose(input_frame, false);
}

int main(){

	test a;
	cv::waitKey(10000);
	a.use_it();
//	cv::namedWindow("input frame");
cv::waitKey(5000);
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



	std::cout<< "[Pose Estimator] Pose estimator stopped!"<<std::endl;
	cv::waitKey(0);
	return 0;
}

