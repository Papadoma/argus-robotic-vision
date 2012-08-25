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

using namespace std;
using namespace cv;

class eye_stereo_match{
private:
	VideoCapture* capture_left;
	VideoCapture* capture_right;
	Mat* mat_left;
	Mat* mat_right;
	Mat* rect_mat_left;
	Mat* rect_mat_right;
	Mat* depth_map;
	Mat* depth_map2;

	Mat* thres_mask;

	Rect roi1, roi2;
	Mat rmap[2][2];

	StereoBM bm;
	StereoSGBM sgbm;
	StereoVar var;

	int numberOfDisparities;
	int width;
	int height;

	void load_param();

public:
	eye_stereo_match();
	~eye_stereo_match();

	Mat imHist(Mat, float, float);


	void refresh_frame();
	void refresh_window();
	void compute_depth();

	void info();
};

//Constructor
eye_stereo_match::eye_stereo_match(){
	width = 640;
	height = 480;
	numberOfDisparities=48;

	//mat_left=new Mat(height,width,CV_16UC3);
	//mat_right=new Mat(height,width,CV_16UC3);
	rect_mat_left=new Mat(height,width,CV_8UC3);
	rect_mat_right=new Mat(height,width,CV_8UC3);
	depth_map=new Mat(height,width,CV_8UC1);
	depth_map2=new Mat(height,width,CV_8UC1);

	thres_mask=new Mat(height,width,CV_8UC1);

	cout<<"Creating capture instances\n";
	capture_left = new VideoCapture();
	capture_right = new VideoCapture();

	if (capture_left&&capture_right)
	{
		std::cout << "Created capture instances!\n";

	}

	cout<<"Setting video parameters\n";
	capture_left->set(CV_CAP_PROP_FRAME_WIDTH, width);
	capture_left->set(CV_CAP_PROP_FRAME_HEIGHT, height);
	capture_left->set(CV_CAP_PROP_FPS, 30);
	capture_right->set(CV_CAP_PROP_FRAME_WIDTH, width);
	capture_right->set(CV_CAP_PROP_FRAME_HEIGHT, height);
	capture_right->set(CV_CAP_PROP_FPS, 30);

	cout<<"Opening video files\n";
	capture_left->open("D:/Videos/left_couple_fixed.avi");
	capture_right->open("D:/Videos/left_couple_fixed.avi");
	if((capture_left->isOpened())&&(capture_right->isOpened())){
		cout<<"Video files opened!\n";
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


	//this->load_param();

	sgbm.preFilterCap = 31;
	sgbm.SADWindowSize = 9;
	int cn = 1;
	sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.minDisparity = 0;
	sgbm.numberOfDisparities = numberOfDisparities;
	sgbm.uniquenessRatio = 15;
	sgbm.speckleWindowSize = 50;
	sgbm.speckleRange = 32;
	sgbm.disp12MaxDiff = 1;
	sgbm.fullDP = false;



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

	cout<<"Initialization complete\n";
}

//Destructor
eye_stereo_match::~eye_stereo_match(){

	destroyAllWindows();

	delete(capture_left);
	delete(capture_right);

}

void eye_stereo_match::refresh_frame(){
	if((capture_left->isOpened())&&(capture_right->isOpened())){
		//capture_left->grab();
		//capture_right->grab();
		//capture_left->retrieve(*mat_left);
		//capture_right->retrieve(*mat_right);
		//cout<<mat_left->size.height<<"\n";
		capture_left->grab();
		capture_right->grab();
		capture_left->retrieve(*mat_left,3);
		capture_right->retrieve(*mat_right,3);
		//capture_left->read(*mat_left);
		//capture_right->read(*mat_right);

		//cvtColor(*mat_left, *mat_left, CV_RGB2GRAY);
		//cvtColor(*mat_right, *mat_right, CV_RGB2GRAY);

		//remap(*mat_left, *rect_mat_left, rmap[0][0], rmap[0][1], CV_INTER_LINEAR);
		//remap(*mat_right, *rect_mat_right, rmap[1][0], rmap[1][1], CV_INTER_LINEAR);

		//cvtColor(*rect_mat_left, *rect_mat_left, CV_GRAY2RGB);
		//cvtColor(*rect_mat_right, *rect_mat_right, CV_GRAY2RGB);

		//rectangle(*rect_mat_left, roi1, Scalar(0,255,0), 1, 8);
		//rectangle(*rect_mat_right, roi2, Scalar(0,255,0), 1, 8);

	}else{
		cout<<"Could not read frame\n";
	}
}

void eye_stereo_match::refresh_window(){
	imshow( "original_camera_left", *mat_left );
	imshow( "original_camera_right", *mat_right );
	//imshow( "camera_left", *rect_mat_left );
	//imshow( "camera_right", *rect_mat_right );
	//imshow( "depth", *depth_map );
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


int main(){

	int key_pressed=255;
	eye_stereo_match *eye_stereo = new eye_stereo_match();

	while(1){

		eye_stereo->refresh_frame();
		//eye_stereo->refresh_window();
		//eye_stereo->compute_depth();

		eye_stereo->refresh_window();
		key_pressed = cvWaitKey(1) & 255;
		if ( key_pressed == 27 ) break;

	}

	delete(eye_stereo);
	return 0;
}
