#include "module_input.hpp"

int main(){
	module_eye test;
	cv::Mat left, right;

	while(1){
		int key_pressed = cvWaitKey(1) & 255;
		if ( key_pressed == 27 ) break;
		test.getFrame(left, right);
		cv::imshow("left",left);
		cv::imshow("right",right);

	}
	return 0;
}
