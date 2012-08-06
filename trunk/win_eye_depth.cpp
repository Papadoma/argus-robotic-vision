#include "opencv2/core/core_c.h"
#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include <stdio.h>
#include <cv.h>
#include <highgui.h>


#include "CLEyeMulticam.h"


using namespace std;
using namespace cv;


class eye_stereo_match{
private:
	CLEyeCameraInstance capture_left;
	CLEyeCameraInstance capture_right;
	PBYTE pCapBufferLeft;
	PBYTE pCapBufferRight;
	IplImage *pCapImageLeft;
	IplImage *pCapImageRight;


	Mat* mat_left;
	Mat* mat_right;
	Mat* rect_mat_left;
	Mat* rect_mat_right;
	Mat* depth_map;
	Mat* previous_depth_map;
	Mat* depth_map2;

	Mat* thres_mask;


	Rect roi1, roi2;
	Mat rmap[2][2];

	StereoBM bm;
	StereoSGBM sgbm;
	StereoVar var;
	BackgroundSubtractorMOG2* BSMOG;

	Rect* clearview_mask;

	int numberOfDisparities;
	int width;
	int height;

	void load_param();
	void smooth_depth_map();

public:
	eye_stereo_match();
	~eye_stereo_match();

	Mat imHist(Mat, float, float);

	void remove_background();
	void refresh_frame();
	void refresh_window();
	void compute_depth();

	void info();
};

//Constructor
eye_stereo_match::eye_stereo_match(){

	GUID left_id=CLEyeGetCameraUUID(1);
	GUID right_id=CLEyeGetCameraUUID(0);

	capture_left = CLEyeCreateCamera(left_id, CLEYE_MONO_RAW , CLEYE_VGA, (float)60);
	capture_right = CLEyeCreateCamera(right_id, CLEYE_MONO_RAW , CLEYE_VGA, (float)60);

	CLEyeCameraGetFrameDimensions(capture_left, width, height);

	pCapBufferLeft=NULL;
	pCapBufferRight=NULL;

	pCapImageLeft = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U   , 1);
	pCapImageRight = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U   , 1);

	mat_left=new Mat(height,width,CV_8UC1);
	mat_right=new Mat(height,width,CV_8UC1);
	rect_mat_left=new Mat(height,width,CV_8UC1);
	rect_mat_right=new Mat(height,width,CV_8UC1);
	depth_map=new Mat(height,width,CV_8UC1);



	thres_mask=new Mat(height,width,CV_8UC1);




	CLEyeSetCameraParameter(capture_left,CLEYE_AUTO_GAIN,true);
	CLEyeSetCameraParameter(capture_left,CLEYE_AUTO_EXPOSURE,true);
	CLEyeSetCameraParameter(capture_left,CLEYE_AUTO_WHITEBALANCE,true);

	CLEyeSetCameraParameter(capture_right,CLEYE_AUTO_GAIN,true);
	CLEyeSetCameraParameter(capture_right,CLEYE_AUTO_EXPOSURE,true);
	CLEyeSetCameraParameter(capture_right,CLEYE_AUTO_WHITEBALANCE,true);

	if(CLEyeCameraStart(capture_left)){
		cout<<"Left Camera initiated recording\n";
	}else{
		cout<<"Left Camera failed\n";
	}
	if(CLEyeCameraStart(capture_right)){
		cout<<"Right Camera initiated recording\n";
	}else{
		cout<<"Right Camera failed\n";
	}

	//namedWindow("original_camera_left",CV_WINDOW_AUTOSIZE);
	//namedWindow("original_camera_right",CV_WINDOW_AUTOSIZE);
	//namedWindow("camera_left",CV_WINDOW_AUTOSIZE);
	//namedWindow("camera_right",CV_WINDOW_AUTOSIZE);
	//namedWindow("depth",CV_WINDOW_AUTOSIZE);
	//namedWindow("depth2",CV_WINDOW_AUTOSIZE);
	//namedWindow("depth_histogram",CV_WINDOW_AUTOSIZE);
	//namedWindow("thres_mask",CV_WINDOW_AUTOSIZE);
	//cvMoveWindow("depth",0,0);
	//cvMoveWindow("depth_histogram",640,0);

	this->load_param();

	numberOfDisparities=48;

	sgbm.preFilterCap = 63; //previously 31
	sgbm.SADWindowSize = 5;
	int cn = 1;
	sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.minDisparity = 0;
	sgbm.numberOfDisparities = numberOfDisparities;
	sgbm.uniquenessRatio = 15;
	sgbm.speckleWindowSize = 100;//previously 50
	sgbm.speckleRange = 32;
	sgbm.disp12MaxDiff = 2;
	sgbm.fullDP = true;


	/*
	bm.state->roi1 = roi1;
	bm.state->roi2 = roi2;
	bm.state->preFilterCap = 31;
	bm.state->SADWindowSize = 19;
	bm.state->minDisparity = 0;
	bm.state->numberOfDisparities = numberOfDisparities;
	bm.state->textureThreshold = 100;
	bm.state->uniquenessRatio = 15;
	bm.state->speckleWindowSize = 100;
	bm.state->speckleRange = 32;
	bm.state->disp12MaxDiff = 1;
	 */
	clearview_mask=new Rect(numberOfDisparities,0,width,height);
	*clearview_mask = roi1 & roi2 & (*clearview_mask);

	previous_depth_map=new Mat(clearview_mask->height,clearview_mask->width,CV_8UC1);
	*previous_depth_map = Mat::zeros(clearview_mask->height, clearview_mask->width, CV_8UC1);

	depth_map2=new Mat(clearview_mask->height,clearview_mask->width,CV_8UC1);

	BSMOG=new BackgroundSubtractorMOG2(1000,30,true);
}

//Destructor
eye_stereo_match::~eye_stereo_match(){

	destroyAllWindows();
	CLEyeCameraStop(capture_left);
	CLEyeCameraStop(capture_right);
}

void eye_stereo_match::refresh_frame(){

	CLEyeCameraGetFrame(capture_left,	pCapBufferLeft, 100);
	CLEyeCameraGetFrame(capture_right,	pCapBufferRight, 100);

	cvGetImageRawData(pCapImageLeft, &pCapBufferLeft);
	cvGetImageRawData(pCapImageRight, &pCapBufferRight);
	*mat_left=pCapImageLeft;
	*mat_right=pCapImageRight;
	//mat_left->convertTo(*rect_mat_left,CV_8UC1);
	//mat_right->convertTo(*rect_mat_right,CV_8UC1);
	remap(*mat_left, *rect_mat_left, rmap[0][0], rmap[0][1], CV_INTER_LINEAR);
	remap(*mat_right, *rect_mat_right, rmap[1][0], rmap[1][1], CV_INTER_LINEAR);

	//keep left rectified image colored for background removal
	//*rect_mat_left_mask=*rect_mat_left;

	//cvtColor( *rect_mat_left, *rect_mat_left, CV_RGB2GRAY );
	//cvtColor( *rect_mat_right, *rect_mat_right, CV_RGB2GRAY );
}

void eye_stereo_match::refresh_window(){
	imshow( "original_camera_left", *mat_left );
	imshow( "original_camera_right", *mat_right );


	rectangle(*rect_mat_left, *clearview_mask, Scalar(255,255,255), 1, 8);
	rectangle(*rect_mat_right, *clearview_mask, Scalar(255,255,255), 1, 8);

	imshow( "camera_left", *rect_mat_left );
	imshow( "camera_right", *rect_mat_right );
	//imshow( "depth", *depth_map );

	Mat* jet_depth_map2=new Mat(height,width,CV_8UC3);
	applyColorMap(*depth_map2, *jet_depth_map2, COLORMAP_JET );
	imshow( "depth2", *jet_depth_map2 );
	//imshow( "depth2", *depth_map2 );
}


void eye_stereo_match::load_param(){

	bool flag1=false;
	bool flag2=false;

	string intrinsics="intrinsics_eye.yml";
	string extrinsics="extrinsics_eye.yml";

	Mat cameraMatrix[2], distCoeffs[2];
	Mat R, T, E, F;
	Mat R1, R2, P1, P2, Q;

	Size imageSize=mat_left->size();

	FileStorage fs(intrinsics, CV_STORAGE_READ);
	if( fs.isOpened() )
	{
		fs["M1"] >> cameraMatrix[0];
		fs["D1"] >> distCoeffs[0] ;
		fs["M2"] >> cameraMatrix[1];
		fs["D2"] >> distCoeffs[1] ;
		fs.release();
		flag1=true;
	}
	else
		cout << "Error: can not load the intrinsic parameters\n";

	fs.open(extrinsics, CV_STORAGE_READ);
	if( fs.isOpened() )
	{
		fs["R"] >> R;
		fs["T"] >> T;
		fs["R1"] >> R1;
		fs["R2"] >> R2;
		fs["P1"] >> P1;
		fs["P2"] >> P2;
		fs["Q"] >> Q;
		fs.release();
		flag2=true;
	}
	else
		cout << "Error: can not load the extrinsics parameters\n";

	if(flag1&&flag2){

		stereoRectify( cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1], imageSize, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, imageSize, &roi1, &roi2 );

		//getOptimalNewCameraMatrix(cameraMatrix[0], distCoeffs[0], imageSize, 1, imageSize, &roi1);
		//getOptimalNewCameraMatrix(cameraMatrix[1], distCoeffs[1], imageSize, 1, imageSize, &roi2);
		initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
		initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

	}

}

void eye_stereo_match::info(){

	//cout<<capture_left->get(CV_CAP_PROP_POS_FRAMES)<<"\n";
}

Mat eye_stereo_match::imHist(Mat hist, float scaleX=1, float scaleY=1){
	double maxVal=0;

	//Mask out the zero values from histogram
	Mat mask=Mat::ones(hist.rows,hist.cols,CV_8UC1);
	mask.at<float>(0,0)=0;

	//Find the 110% of max value
	minMaxLoc(hist, 0, &maxVal, 0, 0,mask);
	maxVal=maxVal*1.1;

	int rows = 64; //default height size
	int cols = hist.rows; //get the width size from the histogram

	Mat histImg = Mat::zeros(rows*scaleX, cols*scaleY, CV_8UC3);

	//for each bin

	for(int i=0;i<cols-1;i++) {
		float histValue = hist.at<float>(i,0);
		float nextValue = hist.at<float>(i+1,0);
		Point pt1 = Point(i*scaleX, rows*scaleY);
		Point pt2 = Point(i*scaleX+scaleX, rows*scaleY);
		Point pt3 = Point(i*scaleX+scaleX, (rows-nextValue*rows/maxVal)*scaleY);
		Point pt4 = Point(i*scaleX, (rows-nextValue*rows/maxVal)*scaleY);

		int numPts = 5;
		Point pts[] = {pt1, pt2, pt3, pt4, pt1};

		fillConvexPoly(histImg, pts, numPts, Scalar(255,255,255));
	}
	return histImg;
}

void eye_stereo_match::compute_depth(){
	rect_mat_left->convertTo(*rect_mat_left,CV_8UC1);
	rect_mat_right->convertTo(*rect_mat_right,CV_8UC1);
	sgbm(*rect_mat_left,*rect_mat_right,*depth_map);
	//var(*rect_mat_left,*rect_mat_right,*depth_map);
	//bm(*rect_mat_left,*rect_mat_right,*depth_map);
	depth_map->convertTo(*depth_map, CV_8UC1, 255/(numberOfDisparities*16.));

	//Cut-out the extra black space on areas which cannot be computed
	//Rect mask(numberOfDisparities,0,width,height);
	Mat tmp(*depth_map, *clearview_mask);
	tmp.copyTo(*depth_map2);
	//Mat tmp2(*depth_map, roi1 & roi2 & mask);
	//tmp2.copyTo(*thres_mask);

	//Smooth out the depth map
	this->smooth_depth_map();

	//depth_map2->convertTo(*depth_map2, CV_8U);

	int hbins = 64;
	int histSize[] = {hbins};
	// saturation varies from 0 (black-gray-white) to
	// 255 (pure spectrum color)
	float sranges[] = { 0, 255 };
	const float* ranges[] = { sranges };
	MatND depth_hist;
	int channels[] = {0};
	calcHist( depth_map2, 1, channels, Mat(), depth_hist, 1, histSize, ranges,	true, false );
	//depth_map->convertTo(*depth_map, CV_8U);
	depth_hist = imHist(depth_hist,4,4);
	imshow( "depth_histogram", depth_hist );


}

void eye_stereo_match::smooth_depth_map(){
	//*previous_depth_map= abs((*depth_map) - (*previous_depth_map)) ;
	//bitwise_and(*previous_depth_map, *depth_map, *depth_map);
	//addWeighted(*depth_map, (double)0.5, *previous_depth_map, (double)0.5, 0, *depth_map);

	//Filter out the zero values of depth map
	Mat temp(depth_map2->size(),CV_8UC1);
	depth_map2->copyTo(temp);
	threshold(temp,temp,0,255,THRESH_BINARY_INV); //Zero values go 255 and everything else goes 0

	//Keep only the calculated values of the last map that correspong to present zero values
	bitwise_and(*previous_depth_map, temp, *previous_depth_map);

	Mat result(depth_map2->size(),CV_8UC1);

	//Add the previous calculated depth values only to zero present values
	add(*depth_map2, *previous_depth_map, result);

	Mat* jet_result=new Mat(result.size(),CV_8UC3);
	applyColorMap(result, *jet_result, COLORMAP_JET );
	imshow( "smoothed", *jet_result);

	result.copyTo(*previous_depth_map);



	//*depth_map= abs((*depth_map) - (*previous_depth_map));
	//*previous_depth_map=*depth_map;
}


void eye_stereo_match::remove_background(){
	//BSMOG.bShadowDetection = false;
	//BSMOG.nmixtures = 3;

	(*BSMOG)(*rect_mat_left,*thres_mask,0.01);

	thres_mask->convertTo(*thres_mask,CV_8UC1);
	threshold(*thres_mask, *thres_mask, 128, 255, THRESH_BINARY); //shadows are 127
	//depth_map2 442 rows 550 cols
	//!int low_thres=128;
	//Mat mask=Mat::zeros(depth_map2->rows,depth_map2->cols,CV_8UC1);

	//!threshold(*depth_map2, *thres_mask, 128, 255, THRESH_TOZERO);
	//Mat * mask=new Mat(depth_map2->rows,depth_map2->cols,CV_8UC1);

	//mask.convertTo(mask, CV_8UC1);

	//mask.at<int>(0,0)=128;
	//cout<<mask.at<int>(100,100)<<"\n";

	//Rect mask(numberOfDisparities,0,width,height);
	Mat tmp(*thres_mask, *clearview_mask);
	tmp.copyTo(*thres_mask);
	//delete(mask);


	imshow( "thres_mask", *thres_mask );

	double minVal;
	double maxVal;
	//*depth_map2 = (*depth_map2)|(*thres_mask);
	depth_map2->copyTo(tmp);
	bitwise_and(*depth_map2, *thres_mask, tmp);
	//tmp.copyTo(*depth_map2);
	minMaxLoc(tmp,0,&maxVal,0,0,Mat());

	tmp=~tmp;
	threshold(tmp, tmp, 254, 255, THRESH_TOZERO_INV);
	minMaxLoc(tmp,0,&minVal,0,0,Mat());
	minVal=255-minVal;

	//imshow( "thres_mask", tmp );
	//threshold(tmp, tmp, 0, 255, THRESH_TOZERO_INV);
	threshold(*depth_map2, *depth_map2, ((int)maxVal)+1, 255, THRESH_TOZERO_INV);
	threshold(*depth_map2, *depth_map2, ((int)minVal), 255, THRESH_TOZERO);
}

int main(){

	int key_pressed=255;
	eye_stereo_match *eye_stereo = new eye_stereo_match();

	int numCams = CLEyeGetCameraCount();
	if(numCams == 0)
	{
		printf("No PS3Eye cameras detected\n");
		return -1;
	}
	printf("Found %d cameras\n", numCams);
	while(1){

		eye_stereo->refresh_frame();

		eye_stereo->compute_depth();

		//eye_stereo->remove_background();
		eye_stereo->refresh_window();
		key_pressed = cvWaitKey(1) & 255;
		if ( key_pressed == 27 ) break;

	}

	delete(eye_stereo);
	return 0;
}
