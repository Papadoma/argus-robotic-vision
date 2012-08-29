#include <cv.hpp>
#include <stdio.h>
#include "CLEyeMulticam.h"

using namespace std;
using namespace cv;

class module_eye{
private:
	CLEyeCameraInstance capture_left;
	CLEyeCameraInstance capture_right;
	PBYTE pCapBufferLeft;
	PBYTE pCapBufferRight;
	IplImage *pCapImageLeft;
	IplImage *pCapImageRight;

	//	Mat* temp_mat_left;
	//	Mat* temp_mat_right;

	int width;	//Frame width
	int height; //Frame height
	bool EoF; 	//End of file flag

public:
	module_eye();
	~module_eye();
	void getFrame(Mat*, Mat*);
	Size getSize();

};
