//#include "module_input.hpp"
#include "module_single_cam.hpp"

int main(){
	module_cam test;
	cv::Mat left, right;

	while(1){
		int key_pressed = cvWaitKey(1) & 255;
		if ( key_pressed == 27 ) break;
		test.getFrame(left);

		cv::imshow("left",left);
		cv::imshow("test",left);
//
	}
	return 0;
}
