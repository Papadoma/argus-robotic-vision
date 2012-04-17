/*
 *  stereo_match.cpp
 *  calibration
 *
 *  Created by Victor  Eruhimov on 1/18/10.
 *  Copyright 2010 Argus Corp. All rights reserved.
 *
 */

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <stdio.h>

using namespace cv;


class eye_depth{
	CvCapture* capture_left;
	CvCapture* capture_right;

	Mat* frame_left;
	Mat* frame_right;

	int width = 640;
	int height = 480;
	int fps = 15;

public:

	int grab_frames();
	void calculate_depth();
};

//Constructor
eye_depth::eye_depth(){
	frame_left=new Mat(height,width,CV_8UC1);
	frame_right=new Mat(height,width,CV_8UC1);
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
eye_depth::~eye_depth(){
    cvReleaseCapture( &capture_right );
    cvReleaseCapture( &capture_left );
}



int eye_depth::grab_frames(){

    cvGrabFrame( capture_left );
    cvGrabFrame( capture_right );

    frame_left = cvRetrieveFrame( capture_left );
    frame_right = cvRetrieveFrame( capture_right );
}

void calculate_depth(){


}

void print_help()
{
	printf("\nDemo stereo matching converting L and R images into disparity and point clouds\n");
	printf("\nUsage: stereo_match <left_image> <right_image> [--algorithm=bm|sgbm|hh|var] [--blocksize=<block_size>]\n"
			"[--max-disparity=<max_disparity>] [--scale=scale_factor>] [-i <intrinsic_filename>] [-e <extrinsic_filename>]\n"
			"[--no-display] [-o <disparity_image>] [-p <point_cloud_file>]\n");
}




int main(int argc, char** argv)
{

}
