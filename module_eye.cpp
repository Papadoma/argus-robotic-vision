#include <cv.hpp>
#include <stdio.h>
#include "CLEyeMulticam.h"

#include "module_eye.hpp"

module_eye::module_eye(){

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
}

module_eye::~module_eye(){
	delete(pCapImageLeft);
	delete(pCapImageRight);
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

}
