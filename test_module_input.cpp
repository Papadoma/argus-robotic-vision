#include "module_input.hpp"

int main(){
	module_eye test("left.mpg","right.mpg");
	cv::Mat left, right;

	while(1){
		int key_pressed = cvWaitKey(1) & 255;
		if ( key_pressed == 27 ) break;
		test.getFrame(left, right);

		cv::imshow("right",right);
		cv::imshow("left",left);
		cv::imshow("test",left);
//
	}
	return 0;
}
