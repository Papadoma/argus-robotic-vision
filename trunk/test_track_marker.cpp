#include "module_input.hpp"
#include "track_marker.hpp"

int main(){
	//	module_eye input;
	module_eye input("left1.mpg","right1.mpg");

	cv::Mat input_frame,l;

	marker_tracker tracker_green("green_histogram.yml");
	marker_tracker tracker_red("red_histogram.yml");
	while(1){
		int key_pressed = cvWaitKey(1) & 255;
		if ( key_pressed == 27 ) break;			//ESC

		input.getFrame(input_frame,l);
		//cv::Point mark_green = tracker_green.get_marker_center(input_frame);
		cv::Point mark_red = tracker_red.get_marker_center(input_frame);

		//if(tracker_green.is_visible())cv::circle(input_frame,mark_green,2,cv::Scalar(0,255,0),2);
		if(tracker_red.is_visible())cv::circle(input_frame,mark_red,2,cv::Scalar(0,0,255),2);

		cv::imshow("output",input_frame);
	}
	return 0;
}
