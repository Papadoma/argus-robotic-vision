#include "module_eye.hpp"

module_eye::module_eye(){
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
		cout<<"Left Camera initiated recording\n";
	}else{
		cout<<"Left Camera failed\n";
	}
	if(CLEyeCameraStart(capture_right)){
		cout<<"Right Camera initiated recording\n";
	}else{
		cout<<"Right Camera failed\n";
	}
#else
	width=640;
	height=480;


	capture_left.open(0);
	capture_right.open(1);

	if(capture_left.isOpened()){
		cout<<"Right Camera initiated recording\n";
	}else{
		cout<<"Right Camera failed\n";
	}
	if(capture_right.isOpened()){
		cout<<"Left Camera initiated recording\n";
	}else{
		cout<<"Left Camera failed\n";
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

Size module_eye::getSize(){
	Size temp_size(width,height);
	return temp_size;
}

void module_eye::getFrame(Mat* mat_left,Mat* mat_right){
#ifdef WIN32
	CLEyeCameraGetFrame(capture_left,	pCapBufferLeft, 100);
	CLEyeCameraGetFrame(capture_right,	pCapBufferRight, 100);
	cvGetImageRawData(pCapImageLeft, &pCapBufferLeft);
	cvGetImageRawData(pCapImageRight, &pCapBufferRight);
	cvtColor((Mat)pCapImageLeft,*mat_left,COLOR_RGBA2RGB);
	cvtColor((Mat)pCapImageRight,*mat_right,COLOR_RGBA2RGB);
#else
	VideoCapture capture_left;
	VideoCapture capture_right;
	capture_left.grab();
	capture_right.grab();
	capture_left.retrieve(frame_left);
	capture_right.retrieve(frame_right);
	mat_left=&frame_left;
	mat_right=&frame_right;
#endif
}
