#include <opencv.hpp>
#include <stdio.h>

#ifdef WIN32
#include "CLEyeMulticam.h"
#endif

using namespace std;
using namespace cv;

class module_eye{
private:

#ifdef WIN32
	CLEyeCameraInstance capture_left;
	CLEyeCameraInstance capture_right;
	PBYTE pCapBufferLeft;
	PBYTE pCapBufferRight;
	IplImage *pCapImageLeft;
	IplImage *pCapImageRight;
#else
	VideoCapture capture_left;
	VideoCapture capture_right;
	Mat frame_left;
	Mat frame_right;
#endif

	int width;	//Frame width
	int height; //Frame height

public:
	module_eye();
	~module_eye();
	void getFrame(Mat*, Mat*);
	Size getSize();

};
