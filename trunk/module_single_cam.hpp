#ifndef module_cam_HPP
#define module_cam_HPP

#include <opencv2/opencv.hpp>
#include <opencv/cv.h>

class module_cam{
private:

	cv::VideoCapture capture;

	bool EoF;
	bool use_camera;
	int width;	//Frame width
	int height; //Frame height

	cv::VideoCapture file;

public:
	module_cam();
	module_cam(cv::String);
	~module_cam();
	void getFrame(cv::Mat&);
	cv::Size getSize();

};

inline module_cam::module_cam()
:use_camera(true)
{
	capture.open(0);

	if(capture.isOpened()){
		std::cout << "Camera initiated recording" << std::endl;
	}else{
		std::cout << "Camera failed" << std::endl;
		exit(1);
	}
	width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
}

inline module_cam::module_cam(std::string filename)
:EoF(false),
 use_camera(false)
{
	std::cout << "Opening video files" << std::endl;
	file.open(filename);

	if((file.isOpened())){
		std::cout << "Video files opened!" << std::endl;
	}else{
		std::cout << "Video files could not be opened/not found..." << std::endl;
		exit(1);
	}

	width = file.get(CV_CAP_PROP_FRAME_WIDTH);
	height = file.get(CV_CAP_PROP_FRAME_HEIGHT);
}

inline module_cam::~module_cam(){
	if(use_camera){
		capture.release();
	}else{
		file.release();
	}
}

inline void module_cam::getFrame(cv::Mat& mat_frame){
	if (use_camera){
		capture.grab();
		capture.retrieve(mat_frame);
	}else{
		if(!file.grab())EoF=true;
		if(EoF==false){
			file.retrieve(mat_frame,3);
		}
	}
}

inline cv::Size module_cam::getSize()
{
	return cv::Size(width,height);
}

#endif
