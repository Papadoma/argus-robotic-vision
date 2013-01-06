#include "module_eye.hpp"
#include <iostream>

module_eye::module_eye()
#ifndef WIN32
: capture_left(0),
  capture_right(1),
  width(640),
  height(480)
#endif
{
	using std::cout;
	using std::endl;

#ifdef WIN32
	int numCams = CLEyeGetCameraCount();
	if(numCams == 0)
	{
		printf("No PS3Eye cameras detected\n");
		exit(1);
	}
	printf("Found %d cameras\n", numCams);
	GUID left_id=CLEyeGetCameraUUID(1);
	GUID right_id=CLEyeGetCameraUUID(0);

	capture_left = CLEyeCreateCamera(left_id, CLEYE_COLOR_RAW , CLEYE_VGA, (float)60);
	capture_right = CLEyeCreateCamera(right_id, CLEYE_COLOR_RAW , CLEYE_VGA, (float)60);

	CLEyeCameraGetFrameDimensions(capture_left, width, height);

	pCapBufferLeft=NULL;
	pCapBufferRight=NULL;

	pCapImageLeft = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U   , 4);
	pCapImageRight = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U   , 4);

	CLEyeSetCameraParameter(capture_left,CLEYE_AUTO_GAIN,true);
	CLEyeSetCameraParameter(capture_left,CLEYE_AUTO_EXPOSURE,true);
	CLEyeSetCameraParameter(capture_left,CLEYE_AUTO_WHITEBALANCE,true);

	CLEyeSetCameraParameter(capture_right,CLEYE_AUTO_GAIN,true);
	CLEyeSetCameraParameter(capture_right,CLEYE_AUTO_EXPOSURE,true);
	CLEyeSetCameraParameter(capture_right,CLEYE_AUTO_WHITEBALANCE,true);

	if(CLEyeCameraStart(capture_left)){
		cout << "Left Camera initiated recording" << endl;
	}else{
		cout << "Left Camera failed" << endl;
	}
	if(CLEyeCameraStart(capture_right)){
		cout << "Right Camera initiated recording" << endl;
	}else{
		cout << "Right Camera failed" << endl;
	}
#else
	if(capture_left.isOpened()){
		cout << "Right Camera initiated recording" << endl;
	}else{
		cout << "Right Camera failed" << endl;
	}
	if(capture_right.isOpened()){
		cout<<"Left Camera initiated recording" << endl;
	}else{
		cout<<"Left Camera failed" << endl;
	}
#endif
}

module_eye::~module_eye(){
#ifdef WIN32
	CLEyeCameraStop(capture_left);
	CLEyeCameraStop(capture_right);
	CLEyeDestroyCamera(capture_left);
	CLEyeDestroyCamera(capture_right);
#endif
}

void module_eye::getFrame(cv::Mat& mat_left,cv::Mat& mat_right){
#ifdef WIN32
	CLEyeCameraGetFrame(capture_left,	pCapBufferLeft, 100);
	CLEyeCameraGetFrame(capture_right,	pCapBufferRight, 100);
	cvGetImageRawData(pCapImageLeft, &pCapBufferLeft);
	cvGetImageRawData(pCapImageRight, &pCapBufferRight);
	cvtColor((Mat)pCapImageLeft,*mat_left,COLOR_RGBA2RGB);
	cvtColor((Mat)pCapImageRight,*mat_right,COLOR_RGBA2RGB);
#else
	capture_left.grab();
	capture_right.grab();
	capture_left.retrieve(mat_left);
	capture_right.retrieve(mat_right);
#endif
}

int main(){
	module_eye test;
	cv::Mat left, right;

	test.getFrame(left, right);
	cv::imshow("left",left);
	cv::imshow("right",right);

	int i = 0;
	std::cin >> i;

	return 0;
}
