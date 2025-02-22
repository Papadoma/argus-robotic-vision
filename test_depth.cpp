/**
	Argus System
    test_depth.cpp
    Purpose: Tests depth calculation using stereo rectified video input

    @author Miltiadis-Alexios Papadopoulos
 */

#include <opencv2/opencv.hpp>
#include "module_input.hpp"

#define DEPTH_ALG 0 //0:SGBM, 1:BM, 2:VAR

class test_depth{
private:
	module_eye* input_module;

#if DEPTH_ALG == 0
	cv::StereoSGBM sgbm;
#elif DEPTH_ALG == 1
	cv::StereoBM bm;
#else
	cv::StereoVar var;
#endif

	cv::Mat cameraMatrix[2], distCoeffs[2];
	cv::Mat R, T, E, F, Q;
	cv::Mat R1, R2, P1, P2;
	cv::Mat rmap[2][2];
	cv::Rect roi1, roi2, *clear_roi;
	int numberOfDisparities;
	cv::Mat mat_left,mat_right,rect_mat_left,rect_mat_right;
	cv::Mat BW_rect_mat_left,BW_rect_mat_right;

public:
	test_depth();
	~test_depth();
	void load_param();
	void refresh_frame();
	void refresh_depth();
	void show_video();
	void smooth_depth();
	void take_snapshot();
	int width,height;

	cv::Mat point_cloud;
	cv::Rect marker;
	cv::MatND histogram;
	cv::Mat mask;
	cv::Point center;

	cv::Mat depth;

};


test_depth::test_depth(){
	//input_module=new module_eye("left.mpg","right.mpg");
	input_module=new module_eye();
	cv::Size framesize = input_module->getSize();
	height=framesize.height;
	width=framesize.width;

	numberOfDisparities=32;
	int cn = 1;
#if DEPTH_ALG == 0
	sgbm.preFilterCap = 63; //previously 31
	sgbm.SADWindowSize = 3;
	sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.minDisparity = 0;
	sgbm.numberOfDisparities = numberOfDisparities;
	sgbm.uniquenessRatio = 30;
	sgbm.speckleWindowSize = 100;//previously 50
	sgbm.speckleRange = 32;
	sgbm.disp12MaxDiff = 2;
	sgbm.fullDP = false;

	cv::namedWindow("parameters",CV_WINDOW_NORMAL );
	cv::createTrackbar("preFilterCap", "parameters", &sgbm.preFilterCap, 200);
	cv::createTrackbar("uniquenessRatio", "parameters", &sgbm.uniquenessRatio, 200);
	cv::createTrackbar("speckleWindowSize", "parameters", &sgbm.speckleWindowSize, 1000);
	cv::createTrackbar("speckleRange", "parameters", &sgbm.speckleRange, 1000);
	cv::createTrackbar("disp12MaxDiff", "parameters", &sgbm.disp12MaxDiff, 1000);
	cv::createTrackbar("P1", "parameters", &sgbm.P1, 1000);
	cv::createTrackbar("P2", "parameters", &sgbm.P2, 1000);

#elif DEPTH_ALG == 1
	bm.init(cv::StereoBM::BASIC_PRESET ,numberOfDisparities,7);
#else
	var.levels = 3;                                 // ignored with USE_AUTO_PARAMS
	var.pyrScale = 0.5;                             // ignored with USE_AUTO_PARAMS
	var.nIt = 25;
	var.minDisp = -numberOfDisparities;
	var.maxDisp = 0;
	var.poly_n = 3;
	var.poly_sigma = 0.0;
	var.fi = 15.0f;
	var.lambda = 0.03f;
	var.penalization = var.PENALIZATION_TICHONOV;   // ignored with USE_AUTO_PARAMS
	var.cycle = var.CYCLE_V;                        // ignored with USE_AUTO_PARAMS
	var.flags = var.USE_SMART_ID | var.USE_AUTO_PARAMS | var.USE_INITIAL_DISPARITY | var.USE_MEDIAN_FILTERING ;
#endif

	this->load_param();
	marker=cv::Rect(250,100,120,350);
	center = cv::Point(320,240);
}

test_depth::~test_depth(){
	delete(input_module);
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
		clear_roi=new cv::Rect(numberOfDisparities,0,width,height);
		*clear_roi = roi1 & roi2 & (*clear_roi);
	}
}

void test_depth::refresh_frame(){

	input_module->getFrame(mat_left,mat_right);

	remap(mat_left, rect_mat_left, rmap[0][0], rmap[0][1], CV_INTER_LINEAR);
	remap(mat_right, rect_mat_right, rmap[1][0], rmap[1][1], CV_INTER_LINEAR);

	//	BW_rect_mat_left=rect_mat_left;
	//	BW_rect_mat_right=rect_mat_right;
	cvtColor(rect_mat_left,BW_rect_mat_left,CV_RGB2GRAY);
	cvtColor(rect_mat_right,BW_rect_mat_right,CV_RGB2GRAY);
}

void test_depth::smooth_depth(){
	cv::Mat depth2,mask;
	cv::threshold(depth, mask, 0, 255, cv::THRESH_BINARY_INV);

	cv::inpaint(depth, mask, depth2, 1, cv::INPAINT_NS);
	cv::imshow("test",depth2);
}

void test_depth::refresh_depth(){
	double t= cv::getTickCount();

#if	DEPTH_ALG == 0
	sgbm(BW_rect_mat_left,BW_rect_mat_right,depth);							//Compute disparity
#elif DEPTH_ALG == 1
	bm(BW_rect_mat_left,BW_rect_mat_right,depth);
#else
	var(BW_rect_mat_left, BW_rect_mat_right, depth);
#endif
	t = cv::getTickCount() - t;
	std::cout<<t*1000/cv::getTickFrequency()<<std::endl;

	cv::Mat local_depth;
	depth.convertTo(local_depth,CV_32FC1, 1./16);
	reprojectImageTo3D(local_depth, point_cloud, Q, false, -1);	//Get the point cloud in WCS

#if DEPTH_ALG == 0 || DEPTH_ALG == 1
	depth.convertTo(depth, CV_8UC1, 255/(numberOfDisparities*16.));
#else
	depth.convertTo(depth, CV_8UC1);
#endif
	cv::Mat jet_depth_map2(height,width,CV_8UC3);
	applyColorMap(depth, jet_depth_map2, cv::COLORMAP_JET );


	//medianBlur(depth, depth, 5);
}


void test_depth::show_video(){
	cv::rectangle(rect_mat_left,*clear_roi,cv::Scalar(0,255,0));
	cv::rectangle(rect_mat_right,*clear_roi,cv::Scalar(0,255,0));
	cv::Mat imgResult(height,2*width,CV_8UC3); // Your final image
	cv::Mat roiImgResult_Left = imgResult(cv::Rect(0,0,width,height));
	cv::Mat roiImgResult_Right = imgResult(cv::Rect(width,0,width,height));
	cv::Mat roiImg1 = rect_mat_left(cv::Rect(0,0,width,height));
	cv::Mat roiImg2 = rect_mat_right(cv::Rect(0,0,width,height));
	roiImg1.copyTo(roiImgResult_Left);
	roiImg2.copyTo(roiImgResult_Right);

	cv::imshow("Both",imgResult);
	cv::Mat overlay;
	addWeighted(rect_mat_left(*clear_roi), 0.5, rect_mat_right(*clear_roi), 0.5, 0, overlay);
	cv::imshow("Combined",overlay);
}

void test_depth::take_snapshot(){
	std::cout<<"Snapshot taken!" << std::endl;
	imwrite("snap_depth.png", depth);
	imwrite("snap_color.png", rect_mat_left);
	//imwrite("person.png", person_left);
}

int main(){
	int key_pressed = 255;
	bool loop=false;
	test_depth test;
	while(1){
		do{
			key_pressed = cvWaitKey(1) & 255;
			if ( key_pressed == 32 )loop=!loop;
			if ( key_pressed == 115 )test.take_snapshot();
			if ( key_pressed == 27 ) break;
		}while (loop);

		if ( key_pressed == 27 ) break;
		test.refresh_frame();
		//cv::imshow("Left",test.BW_rect_mat_left);
		//cv::imshow("Right",test.BW_rect_mat_right);
		test.refresh_depth();
		//test.smooth_depth();
		test.show_video();


		cv::Mat jet_depth_map2(test.height,test.width,CV_8UC3);
		applyColorMap(test.depth, jet_depth_map2, cv::COLORMAP_JET );
		imshow( "depth2", jet_depth_map2 );

	}

	return 0;
}
