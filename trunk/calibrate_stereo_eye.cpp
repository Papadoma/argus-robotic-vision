#include "cv.h"
#include "highgui.h"
#include <stdio.h>

using namespace std;
using namespace cv;

class eye_stereo_calibrate{
private:
	VideoCapture* capture_left;
	VideoCapture* capture_right;
	Mat* mat_left;
	Mat* mat_right;
	Mat* chess_mat_left;
	Mat* chess_mat_right;

	int width;
	int height;

	int image_num;

public:
	eye_stereo_calibrate();
	~eye_stereo_calibrate();
	void save_snapshot();
	void refresh_frame();
	void refresh_window();
	void detect_chessboard();
	void info();
};

//Constructor
eye_stereo_calibrate::eye_stereo_calibrate(){
	width = 640;
	height = 480;

	image_num=1;

	mat_left=new Mat(height,width,CV_8UC1);
	mat_right=new Mat(height,width,CV_8UC1);
	chess_mat_left=new Mat(height,width,CV_8UC1);
	chess_mat_right=new Mat(height,width,CV_8UC1);

	capture_left = new VideoCapture(0);
	capture_right = new VideoCapture(1);

	capture_left->set(CV_CAP_PROP_FRAME_WIDTH, width);
	capture_left->set(CV_CAP_PROP_FRAME_HEIGHT, height);

	capture_right->set(CV_CAP_PROP_FRAME_WIDTH, width);
	capture_right->set(CV_CAP_PROP_FRAME_HEIGHT, height);

	namedWindow("camera_left",CV_WINDOW_AUTOSIZE);
	namedWindow("camera_right",CV_WINDOW_AUTOSIZE);
}

//Destructor
eye_stereo_calibrate::~eye_stereo_calibrate(){
	destroyWindow("camera_left");
	destroyWindow("camera_right");
	delete(capture_left);
	delete(capture_right);

}

void eye_stereo_calibrate::refresh_frame(){
	if((capture_left->isOpened())&&(capture_right->isOpened())){
		capture_left->grab();
		capture_right->grab();
		capture_left->retrieve(*mat_left);
		capture_right->retrieve(*mat_right);

		cvtColor(*mat_left, *mat_left, CV_RGB2GRAY);
		cvtColor(*mat_right, *mat_right, CV_RGB2GRAY);

		mat_left->copyTo(*chess_mat_left);
		mat_right->copyTo(*chess_mat_right);
	}

}

void eye_stereo_calibrate::refresh_window(){
	imshow( "camera_left", *chess_mat_left );
	imshow( "camera_right", *chess_mat_right );
}


void eye_stereo_calibrate::save_snapshot(){

	stringstream ss1;
	string str1;
	ss1 <<"cal_left"<< image_num<<".jpg";
	ss1 >> str1;
	bool flagl = imwrite(str1.c_str(), *mat_left);

	stringstream ss2;
	string str2;
	ss2 <<"cal_right"<< image_num<<".jpg";
	ss2 >> str2;
	bool flagr = imwrite(str2.c_str(), *mat_right);

	if(flagl&&flagr){
		cout<<"Stereo image "<<image_num<<" captured!\n";
		image_num++;
	}
}

void eye_stereo_calibrate::detect_chessboard(){
	Size patternsize(6,9); //interior number of corners (row,col)
	vector<Point2f> left_corners;
	vector<Point2f> right_corners;



	//CALIB_CB_FAST_CHECK saves a lot of time on images
	//that don't contain any chessboard corners
	bool flag_pattern_left = findChessboardCorners(*mat_left, patternsize, left_corners,
			CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
	/*+ CALIB_CB_FAST_CHECK*/);

	bool flag_pattern_right = findChessboardCorners(*mat_right, patternsize, right_corners,
			CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
	/*+ CALIB_CB_FAST_CHECK*/);

	//printf("left status: %d right status: %d",(int)flag_pattern_left,(int)flag_pattern_right);
	/*
	if(flag_pattern_left&&flag_pattern_right){
		cornerSubPix(*mat_left, left_corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		cornerSubPix(*mat_right, right_corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
	}
	 */
	drawChessboardCorners(*chess_mat_left, patternsize, Mat(left_corners), flag_pattern_left);
	drawChessboardCorners(*chess_mat_right, patternsize, Mat(right_corners), flag_pattern_right);

}

void eye_stereo_calibrate::info(){

	//cout<<capture_left->get(CV_CAP_PROP_POS_FRAMES)<<"\n";
}

int main(){

	int key_pressed=255;
	eye_stereo_calibrate *eye_stereo = new eye_stereo_calibrate();

	while(1){
		eye_stereo->refresh_frame();
		//eye_stereo->detect_chessboard();
		eye_stereo->refresh_window();
		eye_stereo->info();
		key_pressed = cvWaitKey(10) & 255;
		if ( key_pressed == 27 ) break;
		if ( key_pressed == 99 ) {eye_stereo->save_snapshot();}
	}

	delete(eye_stereo);
	return 0;
}
