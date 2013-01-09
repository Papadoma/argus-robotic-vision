#include <opencv.hpp>
#include "module_input.hpp"

class test_depth{
private:
	module_eye input_module;
	cv::StereoSGBM sgbm;
	//
	cv::Mat cameraMatrix[2], distCoeffs[2];
	cv::Mat R, T, E, F, Q;
	cv::Mat R1, R2, P1, P2;
	cv::Mat rmap[2][2];
	cv::Rect roi1, roi2;
	int numberOfDisparities;

public:
	test_depth();
	void load_param();
	void refresh_frame();
	void refresh_depth();
	int width,height;

	cv::Mat depth;
	cv::Mat BW_rect_mat_left,BW_rect_mat_right;
};
test_depth::test_depth(){
	cv::Size framesize = input_module.getSize();
	height=framesize.height;
	width=framesize.width;


	this->load_param();

	numberOfDisparities=48;
	sgbm.preFilterCap = 31; //previously 31
	sgbm.SADWindowSize = 1;
	int cn = 1;
	sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.minDisparity = 0;
	sgbm.numberOfDisparities = numberOfDisparities;
	sgbm.uniquenessRatio = 5;
	sgbm.speckleWindowSize = 100;//previously 50
	sgbm.speckleRange = 32;
	sgbm.disp12MaxDiff = -1;
	sgbm.fullDP = true;


}


void test_depth::load_param(){

	bool flag1=false;
	bool flag2=false;

	std::string intrinsics="intrinsics_eye.yml";
	std::string extrinsics="extrinsics_eye.yml";

	cv::Size imageSize(width,height);

	cv::FileStorage fs(intrinsics, CV_STORAGE_READ);
	if( fs.isOpened() )
	{
		fs["M1"] >> cameraMatrix[0];
		fs["D1"] >> distCoeffs[0] ;
		fs["M2"] >> cameraMatrix[1];
		fs["D2"] >> distCoeffs[1] ;
		fs.release();
		flag1=true;
	}
	else
		std::cout << "Error: can not load the intrinsic parameters" << std::endl;

	fs.open(extrinsics, CV_STORAGE_READ);
	if( fs.isOpened() )
	{
		fs["R"] >> R;
		fs["T"] >> T;
		fs["R1"] >> R1;
		fs["R2"] >> R2;
		fs["P1"] >> P1;
		fs["P2"] >> P2;
		fs["Q"] >> Q;
		fs.release();
		flag2=true;
	}
	else
		std::cout << "Error: can not load the extrinsics parameters" << std::endl;

	if(flag1&&flag2){
		cv::Mat Q_local=Q.clone();
		cv::stereoRectify( cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1], imageSize, R, T, R1, R2, P1, P2, Q_local, cv::CALIB_ZERO_DISPARITY, -1, imageSize, &roi1, &roi2 );
		cv::initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
		cv::initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);
	}
}

void test_depth::refresh_frame(){
	cv::Mat mat_left,mat_right,rect_mat_left,rect_mat_right;
	input_module.getFrame(mat_left,mat_right);

	remap(mat_left, rect_mat_left, rmap[0][0], rmap[0][1], CV_INTER_LINEAR);
	remap(mat_right, rect_mat_right, rmap[1][0], rmap[1][1], CV_INTER_LINEAR);

	cvtColor(rect_mat_left,BW_rect_mat_left,CV_RGB2GRAY);
	cvtColor(rect_mat_right,BW_rect_mat_right,CV_RGB2GRAY);
}



void test_depth::refresh_depth(){
	//medianBlur(BW_rect_mat_left, BW_rect_mat_left, 5);
	//medianBlur(BW_rect_mat_right, BW_rect_mat_right, 5);

	//	cv::Mat local_left,local_right;
	//	BW_rect_mat_left.copyTo(local_left);
	//	BW_rect_mat_right.copyTo(local_right);
	//	bilateralFilter(local_left,BW_rect_mat_left,9,10,10);
	//	bilateralFilter(local_right,BW_rect_mat_right,9,10,10);
	sgbm.SADWindowSize = 1;
	sgbm(BW_rect_mat_left,BW_rect_mat_right,depth);
	depth.convertTo(depth, CV_8UC1, 255/(numberOfDisparities*16.));

	cv::Mat jet_depth_map2(height,width,CV_8UC3);
	applyColorMap(depth, jet_depth_map2, cv::COLORMAP_JET );

	//medianBlur(depth, depth, 5);
}
int main(){

	test_depth test;
	while(1){
		int key_pressed = cvWaitKey(1) & 255;
		if ( key_pressed == 27 ) break;
		test.refresh_frame();
		cv::imshow("Left",test.BW_rect_mat_left);
		test.refresh_depth();

		cv::Mat jet_depth_map2(test.height,test.width,CV_8UC3);
		applyColorMap(test.depth, jet_depth_map2, cv::COLORMAP_JET );
		imshow( "depth2", jet_depth_map2 );

	}

	return 0;
}
