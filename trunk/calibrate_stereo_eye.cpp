#include "cv.h"
#include "highgui.h"
#include <stdio.h>

class eye_stereo_calibrate{
private:
	CvCapture* capture_left;
	CvCapture* capture_right;
	IplImage* frame_left;
	IplImage* frame_right;

	int width;
	int height;
	int fps;

public:
	eye_stereo_calibrate();
	~eye_stereo_calibrate();
	void save_snapshot();
	void refresh_frame();
	void refresh_window();
};

//Constructor
eye_stereo_calibrate::eye_stereo_calibrate(){
	width = 640;
	height = 480;
	fps = 15;

	capture_left = cvCaptureFromCAM( 0 );
	if ( !capture_left ) {
		fprintf( stderr, "ERROR: could not initialize camera 0 \n" );
		getchar();
		exit;
	}
	capture_right = cvCaptureFromCAM( 1 );
	if ( !capture_right ) {
		fprintf( stderr, "ERROR: could not initialize camera 1 \n" );
		getchar();
		exit;
	}
	cvSetCaptureProperty( capture_left, CV_CAP_PROP_FRAME_WIDTH, width );
	cvSetCaptureProperty( capture_left, CV_CAP_PROP_FRAME_HEIGHT, height );
	cvSetCaptureProperty( capture_left, CV_CAP_PROP_FPS, fps );
	cvSetCaptureProperty( capture_right, CV_CAP_PROP_FRAME_WIDTH, width );
	cvSetCaptureProperty( capture_right, CV_CAP_PROP_FRAME_HEIGHT, height );
	cvSetCaptureProperty( capture_right, CV_CAP_PROP_FPS, fps );

	cvNamedWindow( "camera_left", CV_WINDOW_AUTOSIZE );
	cvNamedWindow( "camera_right", CV_WINDOW_AUTOSIZE );
}

//Destructor
eye_stereo_calibrate::~eye_stereo_calibrate(){
	cvReleaseCapture( &capture_right );
	cvReleaseCapture( &capture_left );
	cvDestroyWindow( "camera_left" );
	cvDestroyWindow( "camera_right" );
}

void eye_stereo_calibrate::refresh_frame(){
	cvGrabFrame( capture_left );
	cvGrabFrame( capture_right );
	frame_left = cvRetrieveFrame( capture_left );
	frame_right = cvRetrieveFrame( capture_right );
}

void eye_stereo_calibrate::refresh_window(){
	cvShowImage( "camera_left", frame_left );
	cvShowImage( "camera_right", frame_right );
}


void eye_stereo_calibrate::save_snapshot(){

}

int main(){

	eye_stereo_calibrate *eye_stereo = new eye_stereo_calibrate();
	while(1){
		eye_stereo->refresh_frame();
		eye_stereo->refresh_window();
		if ( (cvWaitKey(10) & 255) == 27 ) break;
	}
	return 0;
}
