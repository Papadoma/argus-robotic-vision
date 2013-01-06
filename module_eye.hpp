#ifndef MODULE_EYE_HPP
#define MODULE_EYE_HPP

#include <opencv.hpp>

#ifdef WIN32
#include "CLEyeMulticam.h"
#endif


class module_eye{
private:
#ifdef WIN32
	typedef CLEyeCameraInstance capture_type;
#else
	typedef cv::VideoCapture capture_type;
#endif

	capture_type capture_left;
	capture_type capture_right;

#ifdef WIN32
	PBYTE pCapBufferLeft;
	PBYTE pCapBufferRight;
	IplImage *pCapImageLeft;
	IplImage *pCapImageRight;
#endif

	int width;	//Frame width
	int height; //Frame height

public:
	module_eye();
	~module_eye();
	void getFrame(cv::Mat&, cv::Mat&);
	cv::Size getSize() const;

};

inline cv::Size module_eye::getSize() const
{
	return cv::Size(width,height);
}

#endif
