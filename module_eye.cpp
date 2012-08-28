#include <opencv.hpp>
#include <stdio.h>
#include "CLEyeMulticam.h"

#include "module_eye.hpp"

module_eye::module_eye(){
	int numCams = CLEyeGetCameraCount();
	if(numCams == 0)
	{
		printf("No PS3Eye cameras detected\n");
		exit(1);
	}
	printf("Found %d cameras\n", numCams);
	GUID left_id=CLEyeGetCameraUUID(1);
	GUID right_id=CLEyeGetCameraUUID(0);

	capture_left = CLEyeCreateCamera(left_id, CLEYE_MONO_RAW , CLEYE_VGA, (float)60);
	capture_right = CLEyeCreateCamera(right_id, CLEYE_MONO_RAW , CLEYE_VGA, (float)60);

	CLEyeCameraGetFrameDimensions(capture_left, width, height);

	pCapBufferLeft=NULL;
	pCapBufferRight=NULL;

	pCapImageLeft = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U   , 1);
	pCapImageRight = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U   , 1);

	CLEyeSetCameraParameter(capture_left,CLEYE_AUTO_GAIN,true);
	CLEyeSetCameraParameter(capture_left,CLEYE_AUTO_EXPOSURE,true);
	CLEyeSetCameraParameter(capture_left,CLEYE_AUTO_WHITEBALANCE,true);

	CLEyeSetCameraParameter(capture_right,CLEYE_AUTO_GAIN,true);
	CLEyeSetCameraParameter(capture_right,CLEYE_AUTO_EXPOSURE,true);
	CLEyeSetCameraParameter(capture_right,CLEYE_AUTO_WHITEBALANCE,true);

	if(CLEyeCameraStart(capture_left)){
		cout<<"Left Camera initiated recording\n";
	}else{
		cout<<"Left Camera failed\n";
	}
	if(CLEyeCameraStart(capture_right)){
		cout<<"Right Camera initiated recording\n";
	}else{
		cout<<"Right Camera failed\n";
	}

	//
	//	temp_mat_left=new Mat(height,width,CV_8UC4);
	//	temp_mat_right=new Mat(height,width,CV_8UC4);
}

module_eye::~module_eye(){
	CLEyeCameraStop(capture_left);
	CLEyeCameraStop(capture_right);
	CLEyeDestroyCamera(capture_left);
	CLEyeDestroyCamera(capture_right);
}

Size module_eye::getSize(){
	Size temp_size(width,height);
	return temp_size;
}

void module_eye::getFrame(Mat* mat_left,Mat* mat_right){

	CLEyeCameraGetFrame(capture_left,	pCapBufferLeft, 100);
	CLEyeCameraGetFrame(capture_right,	pCapBufferRight, 100);

	cvGetImageRawData(pCapImageLeft, &pCapBufferLeft);
	cvGetImageRawData(pCapImageRight, &pCapBufferRight);
	*mat_left=pCapImageLeft;
	*mat_right=pCapImageRight;

	//	cvtColor(*temp_mat_left,*temp_mat_left,COLOR_RGBA2RGB);
	//	cvtColor(*temp_mat_right,*temp_mat_right,COLOR_RGBA2RGB);
	//	temp_mat_left->convertTo(*temp_mat_left,CV_8UC3);
	//	temp_mat_right->convertTo(*temp_mat_right,CV_8UC3);
	//
	//	cvtColor(*temp_mat_left,*mat_left,CV_RGB2GRAY);
	//	cvtColor(*temp_mat_right,*mat_right,CV_RGB2GRAY);

}
