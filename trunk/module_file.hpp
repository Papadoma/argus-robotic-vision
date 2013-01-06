#ifndef MODULE_FILE_HPP
#define MODULE_FILE_HPP

#include <opencv.hpp>
//#include "CLEyeMulticam.h"


class module_file{
private:
	cv::String filename_left;
	cv::String filename_right;

	cv::VideoCapture* capture_left;
	cv::VideoCapture* capture_right;

	int width;
	int height;
	bool EoF;

public:
	module_file();
	~module_file();
	void getFrame(cv::Mat*, cv::Mat*);
	cv::Size getSize() const;

};

inline cv::Size module_file::getSize() const
{
	return cv::Size(width,height);
}


#endif
