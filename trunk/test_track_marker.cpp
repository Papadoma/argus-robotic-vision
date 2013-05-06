#include "module_input.hpp"
#include "track_marker.hpp"

int main(){
	//	module_eye input;
	module_eye input("left.mpg","right.mpg");

	cv::Mat input_frame,l;

	cv::Mat marker_img = cv::imread("green_marker.jpg");
	marker_tracker tracker("green_histogram.yml");
	while(1){
		int key_pressed = cvWaitKey(1) & 255;
		if ( key_pressed == 27 ) break;			//ESC

		input.getFrame(input_frame,l);
		cv::Point mark = tracker.get_marker_center(input_frame);
		cv::circle(input_frame,mark,2,cv::Scalar(0,255,0),2);
		cv::imshow("output",input_frame);
	}
	return 0;
}
