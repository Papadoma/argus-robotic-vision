#ifndef MODULE_EYE_HPP
#define MODULE_EYE_HPP


#include <opencv2/opencv.hpp>
#include <opencv/cv.h>

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#include <CLEyeMulticam.h>
#endif

class module_eye{
private:
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
	typedef CLEyeCameraInstance capture_type;
#else
	typedef cv::VideoCapture capture_type;
#endif

	capture_type capture_left;
	capture_type capture_right;
	bool EoF;
	bool use_camera;
	int width;	//Frame width
	int height; //Frame height

	cv::VideoCapture file_left;
	cv::VideoCapture file_right;

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
	PBYTE pCapBufferLeft;
	PBYTE pCapBufferRight;
	IplImage *pCapImageLeft;
	IplImage *pCapImageRight;
#endif

public:
	module_eye();
	module_eye(cv::String, cv::String);
	~module_eye();
	void getFrame(cv::Mat&, cv::Mat&);
	cv::Size getSize();

};

inline module_eye::module_eye()
:use_camera(true)
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
,
width(640),
height(480)
#endif

{
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
	int numCams = CLEyeGetCameraCount();
	if(numCams == 0)
	{
		printf("No PS3Eye cameras detected\n");
		exit(1);
	}
	printf("Found %d cameras\n", numCams);
	GUID left_id=CLEyeGetCameraUUID(1);
	GUID right_id=CLEyeGetCameraUUID(0);

	capture_left = CLEyeCreateCamera(left_id, CLEYE_BAYER_RAW , CLEYE_VGA, (float)60);
	capture_right = CLEyeCreateCamera(right_id, CLEYE_BAYER_RAW , CLEYE_VGA, (float)60);

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
		std::cout << "Left Camera initiated recording" << std::endl;
	}else{
		std::cout << "Left Camera failed" << std::endl;
	}
	if(CLEyeCameraStart(capture_right)){
		std::cout << "Right Camera initiated recording" << std::endl;
	}else{
		std::cout << "Right Camera failed" << std::endl;
	}
#else
	capture_left.open(0);
	capture_right.open(1);
	if(capture_left.isOpened()){
		std::cout << "Right Camera initiated recording" << std::endl;
	}else{
		std::cout << "Right Camera failed" << std::endl;
		exit(1);
	}
	if(capture_right.isOpened()){
		std::cout<<"Left Camera initiated recording" << std::endl;
	}else{
		std::cout<<"Left Camera failed" << std::endl;
		exit(1);
	}
#endif
}

inline module_eye::module_eye(std::string filename_left, std::string filename_right)
:EoF(false),
 use_camera(false)
{
	std::cout << "Opening video files" << std::endl;
	file_left.open(filename_left);
	file_right.open(filename_right);

	if((file_left.isOpened())&&(file_right.isOpened())){
		std::cout << "Video files opened!" << std::endl;
	}else{
		std::cout << "Video files could not be opened/not found..." << std::endl;
		exit(1);
	}

	width = file_left.get(CV_CAP_PROP_FRAME_WIDTH);
	height = file_left.get(CV_CAP_PROP_FRAME_HEIGHT);
}

inline module_eye::~module_eye(){
	if(use_camera){
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
		CLEyeCameraStop(capture_left);
		CLEyeCameraStop(capture_right);
		CLEyeDestroyCamera(capture_left);
		CLEyeDestroyCamera(capture_right);
		cvReleaseImage(&pCapImageLeft);
		cvReleaseImage(&pCapImageLeft);
#else
		capture_left.release();
		capture_right.release();
#endif
	}else{
		file_left.release();
		file_right.release();
	}
}

inline void module_eye::getFrame(cv::Mat& mat_left,cv::Mat& mat_right){
	if (use_camera){
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
		CLEyeCameraGetFrame(capture_left,	pCapBufferLeft, 50);
		CLEyeCameraGetFrame(capture_right,	pCapBufferRight, 50);

		cvGetImageRawData(pCapImageLeft, &pCapBufferLeft);
		cvGetImageRawData(pCapImageRight, &pCapBufferRight);

		cv::cvtColor((cv::Mat)pCapImageLeft,mat_left,cv::COLOR_BayerGR2RGB);
		cv::cvtColor((cv::Mat)pCapImageRight,mat_right,cv::COLOR_BayerGR2RGB);
#else
		capture_left.grab();
		capture_right.grab();
		capture_left.retrieve(mat_left);
		capture_right.retrieve(mat_right);
#endif
	}else{
		if(!file_left.grab())EoF=true;
		if(!file_right.grab())EoF=true;

		if(EoF==false){
			file_left.retrieve(mat_left,3);
			file_right.retrieve(mat_right,3);
		}
	}
}

inline cv::Size module_eye::getSize()
{
	return cv::Size(width,height);
}

#endif
