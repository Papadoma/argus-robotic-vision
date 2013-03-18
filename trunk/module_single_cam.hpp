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

	capture_type capture;
	bool EoF;
	bool use_camera;
	int width;	//Frame width
	int height; //Frame height

	cv::VideoCapture file;

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
	PBYTE pCapBuffer;
	IplImage *pCapImage;
#endif

public:
	module_eye();
	module_eye(std::string);
	~module_eye();
	void getFrame(cv::Mat&);
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
		printf("No PS3Eye camera detected\n");
		exit(1);
	}
	printf("Found %d cameras\n", numCams);
	GUID right_id=CLEyeGetCameraUUID(0);

	capture = CLEyeCreateCamera(right_id, CLEYE_BAYER_RAW , CLEYE_VGA, (float)60);

	CLEyeCameraGetFrameDimensions(capture, width, height);

	pCapBuffer=NULL;

	pCapImage = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U   , 1);


	CLEyeSetCameraParameter(capture,CLEYE_AUTO_GAIN,true);
	CLEyeSetCameraParameter(capture,CLEYE_AUTO_EXPOSURE,true);
	CLEyeSetCameraParameter(capture,CLEYE_AUTO_WHITEBALANCE,true);

	if(CLEyeCameraStart(capture)){
		std::cout << "[Module Input] Left Camera initiated recording" << std::endl;
	}else{
		std::cout << "[Module Input] Left Camera failed" << std::endl;
	}

#else
	capture.open(0);
	if(capture.isOpened()){
		std::cout<<"[Module Input] Left Camera initiated recording" << std::endl;
	}else{
		std::cout<<"[Module Input] Left Camera failed" << std::endl;
		exit(1);
	}
#endif
}

inline module_eye::module_eye(std::string filename)
:EoF(false),
 use_camera(false)
{
	std::cout << "[Module Input] Opening video files" << std::endl;
	file.open(filename);


	if(file.isOpened()){
		std::cout << "[Module Input] Video files opened!" << std::endl;
	}else{
		std::cout << "[Module Input] Video files could not be opened/not found..." << std::endl;
		exit(1);
	}

	width = file.get(CV_CAP_PROP_FRAME_WIDTH);
	height = file.get(CV_CAP_PROP_FRAME_HEIGHT);
}

inline module_eye::~module_eye(){
	if(use_camera){
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
		CLEyeCameraStop(capture);
		CLEyeDestroyCamera(capture);
		cvReleaseImage(&pCapImage);
#else
		capture.release();
#endif
	}else{
		file.release();
	}
}

inline void module_eye::getFrame(cv::Mat& mat){
	if (use_camera){
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
		CLEyeCameraGetFrame(capture,	pCapBuffer, 2000);

		cvGetImageRawData(pCapImage, &pCapBuffer);

		cv::cvtColor((cv::Mat)pCapImage,mat,cv::COLOR_BayerGR2RGB);
#else
		capture.grab();
		capture.retrieve(mat);
#endif
	}else{
		if(!file.grab())EoF=true;

		if(EoF==false){
			file.retrieve(mat,3);
		}
	}
}

inline cv::Size module_eye::getSize()
{
	return cv::Size(width,height);
}

#endif
