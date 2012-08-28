#include <opencv.hpp>
#include <stdio.h>
#include "CLEyeMulticam.h"

using namespace std;
using namespace cv;

class module_file{
private:
	String filename_left;
	String filename_right;

	VideoCapture* capture_left;
	VideoCapture* capture_right;

	int width;
	int height;

public:
	module_file();
	~module_file();
	void getFrame(Mat*, Mat*);
	Size getSize();

};
