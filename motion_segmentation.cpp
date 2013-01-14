#include <stdio.h>
#include <opencv.hpp>
#include "module_input.hpp"

const double MHI_DURATION = 1;
const double MAX_TIME_DELTA = 0.3;

class motion_segmentation{
private:
	module_eye* input_instance;

	cv::Mat frame_grayscale;
	cv::Mat frame_left, frame_right;
	cv::Mat prev_frame;
	cv::Mat MHI;
	cv::Mat motion_seg;
public:
	motion_segmentation();
	~motion_segmentation();
	void get_frame();
	void refresh_output();
	void scene_segmentation();
};

motion_segmentation::motion_segmentation(){
	input_instance = new module_eye();
	cv::Size framesize = input_instance->getSize();
	prev_frame = cv::Mat::zeros(framesize,CV_8UC1);
	motion_seg = cv::Mat::zeros(framesize,CV_8UC1);
	MHI = cv::Mat::zeros(framesize,CV_32FC1);
}

motion_segmentation::~motion_segmentation(){
	delete(input_instance);
}
void motion_segmentation::refresh_output(){
	cv::imshow("feed",frame_left);
}

void motion_segmentation::get_frame(){
	input_instance->getFrame(frame_left,frame_right);
	cv::cvtColor(frame_left, frame_grayscale,cv::COLOR_RGB2GRAY);
}

void motion_segmentation::scene_segmentation(){
	double timestamp = (double)cv::getTickCount()/cv::getTickFrequency();

	cv::Mat frame_diff;
	absdiff(frame_grayscale, prev_frame, frame_diff);


	threshold( frame_diff, frame_diff, 80, 1, CV_THRESH_BINARY );
	updateMotionHistory(frame_diff, MHI, timestamp, MHI_DURATION);
	frame_diff=frame_diff*255;
	imshow("test",frame_diff);

	imshow("MHI",MHI);
	//MHI.convertTo(MHI,CV_8UC1,255./MHI_DURATION,(MHI_DURATION - timestamp)*255./MHI_DURATION);
	cv::Mat segmask;
	std::vector <cv::Rect> boundingRects;
	cv::segmentMotion(MHI, segmask, boundingRects, timestamp, MAX_TIME_DELTA);

	int rect_size = boundingRects.size();
	for(int i = 0; i<rect_size;i++){
		cv::rectangle(frame_left,boundingRects[i],cv::Scalar((6*i*255/rect_size)%255,8*(255-i*255/rect_size)%255,i*255*10/rect_size%255));
	}
	imshow("Segmentation mask",segmask);


	frame_grayscale.copyTo(prev_frame);
}



int main(){
	motion_segmentation test;

	while(1){
		int key_pressed = cvWaitKey(1) & 255;
		if ( key_pressed == 27 ) break;
		test.get_frame();
		test.scene_segmentation();

		test.refresh_output();
	}
}
