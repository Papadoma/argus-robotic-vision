#include "cv.h"
#include "highgui.h"
#include <stdio.h>

class eye_stereo_calibrate{
private:
	CvCapture* capture_left;
	CvCapture* capture_right;

    int width = 640;
    int height = 480;
    int fps = 15;
public:

};

//Constructor
eye_stereo_calibrate::eye_stereo_calibrate(){
	capture_left = cvCaptureFromCAM( 0 );
	if ( !capture_left ) {
		fprintf( stderr, "ERROR: could not initialize camera 0 \n" );
		getchar();
		return -1;
	}
	capture_right = cvCaptureFromCAM( 1 );
	if ( !capture_right ) {
		fprintf( stderr, "ERROR: could not initialize camera 1 \n" );
		getchar();
		return -1;
	}

	cvSetCaptureProperty( capture_left, CV_CAP_PROP_FRAME_WIDTH, width );
	cvSetCaptureProperty( capture_left, CV_CAP_PROP_FRAME_HEIGHT, height );
	cvSetCaptureProperty( capture_left, CV_CAP_PROP_FPS, fps );

	cvSetCaptureProperty( capture_right, CV_CAP_PROP_FRAME_WIDTH, width );
	cvSetCaptureProperty( capture_right, CV_CAP_PROP_FRAME_HEIGHT, height );
	cvSetCaptureProperty( capture_right, CV_CAP_PROP_FPS, fps );


}

//Destructor
eye_stereo_calibrate::~eye_stereo_calibrate(){

}
