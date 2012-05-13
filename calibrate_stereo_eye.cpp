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

class eye_stereo_calibrate{
private:
	VideoCapture* capture_left;
	VideoCapture* capture_right;
	Mat* mat_left;
	Mat* mat_right;
	Mat* calib_mat_left;
	Mat* calib_mat_right;
	Mat* chess_mat_left;
	Mat* chess_mat_right;

	Size square_pattern_size;
	Size circle_pattern_size;

	int saved_data;
	int numSquares;

	vector<vector<Point2f> > imagePoints[2];
	vector<vector<Point3f> > objectPoints;
	vector<Point2f> left_corners;
	vector<Point2f> right_corners;

	Mat cameraMatrix[2], distCoeffs[2];
	Mat R, T, E, F;

	Mat rmap[2][2];

	int width;
	int height;
	float squareSize;
	float circleSize;
	int image_num;

	bool flag_pattern_right;
	bool flag_pattern_left;

	bool calib_left_flag;
	bool calib_right_flag;


public:
	int camera_choice;
	int pattern_choice;

	bool calib_stereo;

	eye_stereo_calibrate();
	~eye_stereo_calibrate();
	void save_snapshot();
	void refresh_frame();
	void refresh_window();
	void detect_chessboard();
	void add_calibration_data();
	void calibrate();
	void save_data();
	void info();
	void init_undistort();
	void undistort();
};

//Constructor
eye_stereo_calibrate::eye_stereo_calibrate(){
	width = 640;
	height = 480;
	squareSize = 25;//mm
	circleSize = 17.5;//mm

	saved_data=0;
	image_num=1;
	numSquares=6*9;

	camera_choice=0;
	pattern_choice=0;

	square_pattern_size=Size(9,6); //size(width,height)
	circle_pattern_size=Size(4,11);

	calib_stereo=false;
	calib_right_flag=false;
	calib_left_flag=false;

	cameraMatrix[0] = Mat::eye(3, 3, CV_64F);
	cameraMatrix[1] = Mat::eye(3, 3, CV_64F);

	mat_left=new Mat(height,width,CV_8UC1);
	mat_right=new Mat(height,width,CV_8UC1);
	calib_mat_left=new Mat(height,width,CV_8UC1);
	calib_mat_right=new Mat(height,width,CV_8UC1);
	chess_mat_left=new Mat(height,width,CV_8UC1);
	chess_mat_right=new Mat(height,width,CV_8UC1);

	capture_left = new VideoCapture(1);
	capture_right = new VideoCapture(0);
	if(!capture_left->isOpened())cout<<"Could not initialize left camera/n";
	if(!capture_right->isOpened())cout<<"Could not initialize right camera/n";
	
	capture_left->set(CV_CAP_PROP_FRAME_WIDTH, width);
	capture_left->set(CV_CAP_PROP_FRAME_HEIGHT, height);

	capture_right->set(CV_CAP_PROP_FRAME_WIDTH, width);
	capture_right->set(CV_CAP_PROP_FRAME_HEIGHT, height);

	namedWindow("camera_left",CV_WINDOW_AUTOSIZE);
	namedWindow("camera_right",CV_WINDOW_AUTOSIZE);
	namedWindow("calib_camera_left",CV_WINDOW_AUTOSIZE);
	namedWindow("calib_camera_right",CV_WINDOW_AUTOSIZE);
}

//Destructor
eye_stereo_calibrate::~eye_stereo_calibrate(){
	destroyWindow("camera_left");
	destroyWindow("camera_right");
	destroyWindow("calib_camera_left");
	destroyWindow("calib_camera_right");
	delete(capture_left);
	delete(capture_right);

}

void eye_stereo_calibrate::refresh_frame(){
	if((capture_left->isOpened())&&(capture_right->isOpened())){
		capture_left->grab();
		capture_right->grab();
		capture_left->retrieve(*mat_left);
		capture_right->retrieve(*mat_right);

		mat_left->copyTo(*chess_mat_left);
		mat_right->copyTo(*chess_mat_right);

		cvtColor(*mat_left, *mat_left, CV_RGB2GRAY);
		cvtColor(*mat_right, *mat_right, CV_RGB2GRAY);


	}

}

void eye_stereo_calibrate::refresh_window(){
	imshow( "camera_left", *chess_mat_left );
	imshow( "camera_right", *chess_mat_right );
	imshow( "calib_camera_left", *calib_mat_left );
	imshow( "calib_camera_right", *calib_mat_right );
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

	//CALIB_CB_FAST_CHECK saves a lot of time on images
	//that don't contain any chessboard corners
	if(camera_choice==0||camera_choice==2){
		if(pattern_choice==0){
			flag_pattern_left = findChessboardCorners(*mat_left, square_pattern_size, left_corners,
					CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE /*+ CALIB_CB_FAST_CHECK*/);
		}else{
			flag_pattern_left = findCirclesGrid(*mat_left, circle_pattern_size, left_corners,
					CALIB_CB_ASYMMETRIC_GRID);
		}
	}
	if(camera_choice==1||camera_choice==2){
		if(pattern_choice==0){
			flag_pattern_right = findChessboardCorners(*mat_right, square_pattern_size, right_corners,
					CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE/*+ CALIB_CB_FAST_CHECK*/);

		}else{
			flag_pattern_right = findCirclesGrid(*mat_right, circle_pattern_size, right_corners,
					CALIB_CB_ASYMMETRIC_GRID);
		}
	}

	if(flag_pattern_left&&flag_pattern_right&&pattern_choice==0){
		cornerSubPix(*mat_left, left_corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		cornerSubPix(*mat_right, right_corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
	}

	if(camera_choice==0||camera_choice==2){
		if(pattern_choice==0){
			drawChessboardCorners(*chess_mat_left, square_pattern_size, Mat(left_corners), flag_pattern_left);
		}else{
			drawChessboardCorners(*chess_mat_left, circle_pattern_size, Mat(left_corners), flag_pattern_left);
		}
	}
	if(camera_choice==1||camera_choice==2){
		if(pattern_choice==0){
			drawChessboardCorners(*chess_mat_right, square_pattern_size, Mat(right_corners), flag_pattern_right);
		}else{
			drawChessboardCorners(*chess_mat_right, circle_pattern_size, Mat(right_corners), flag_pattern_right);
		}
	}



}

void eye_stereo_calibrate::info(){

	cout<<"\nCalibrate Playstation Eye Stereo rig"<<"\n";
	cout<<"------------------------------------"<<"\n";
	cout<<"c: 	capture stereo images"<<"\n";
	cout<<"[:	select square pattern (default)"<<"\n";
	cout<<"]:	select circle pattern"<<"\n";
	cout<<"0:	select left camera (default)"<<"\n";
	cout<<"1:	select right camera"<<"\n";
	cout<<"2:	select stereo camera"<<"\n";
	cout<<"a:	add calibration data"<<"\n";
	cout<<"s:	start calibration"<<"\n";
	cout<<"ESC: terminate program"<<"\n\n";
}

void eye_stereo_calibrate::add_calibration_data(){

	if((flag_pattern_left&&camera_choice==0)||(flag_pattern_right&&camera_choice==1)||(flag_pattern_right&&flag_pattern_left&&camera_choice==2)){
		if(camera_choice==0||camera_choice==2)imagePoints[0].push_back(left_corners);
		if(camera_choice==1||camera_choice==2)imagePoints[1].push_back(right_corners);

		vector<Point3f> obj;
		if(pattern_choice==0){
			for( int j = 0; j < square_pattern_size.height; j++ ) //6
				for( int k = 0; k < square_pattern_size.width; k++ ) //9
					obj.push_back(Point3f(float(j*squareSize), float(k*squareSize), 0));

		}else{
			for( int i = 0; i < circle_pattern_size.height; i++ )
				for( int j = 0; j < circle_pattern_size.width; j++ )
					obj.push_back(Point3f(float((2*j + i % 2)*circleSize), float(i*circleSize), 0));
		}
		objectPoints.push_back(obj);

		saved_data++;
		cout<<"Added calibration snapshot:"<<saved_data<<"\n";
	}else{
		cout<<"Pattern not visible on configuration: "<<camera_choice<<"\n";
	}
}

void eye_stereo_calibrate::calibrate(){
	Size imageSize=mat_left->size();
	vector<Mat> rvecs;
	vector<Mat> tvecs;
	vector<float> reprojErrs;
	double rms_left;
	double rms_right;
	double rms_stereo;

	switch (camera_choice){
	case 0:
		cout<<"Calibration for left camera started!\n";
		rms_left = calibrateCamera(objectPoints, imagePoints[0], imageSize, cameraMatrix[0],
				distCoeffs[0], rvecs, tvecs,
				CV_CALIB_FIX_K4| CV_CALIB_FIX_K5);
		cout<<"Calibration for left camera ended! RMS error:" <<rms_left<<"\n";
		imagePoints[0].clear();
		objectPoints.clear();
		this->info();
		saved_data=0;
		calib_left_flag=true;
		break;

	case 1:
		cout<<"Calibration for right camera started!\n";
		rms_right = calibrateCamera(objectPoints, imagePoints[1], imageSize, cameraMatrix[1],
				distCoeffs[1], rvecs, tvecs,
				CV_CALIB_FIX_K4| CV_CALIB_FIX_K5);
		cout<<"Calibration for right camera ended! RMS error:" <<rms_right<<"\n";
		imagePoints[1].clear();
		objectPoints.clear();
		this->info();
		saved_data=0;
		calib_right_flag=true;
		break;

	case 2:
		if(calib_right_flag&&calib_left_flag){
			cout<<"Calibration for stereo started!\n";
			rms_stereo = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
					cameraMatrix[0], distCoeffs[0],
					cameraMatrix[1], distCoeffs[1],
					imageSize, R, T, E, F,
					TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5),
					CV_CALIB_FIX_INTRINSIC);
			cout<<"Calibration for stereo ended! RMS error:" <<rms_stereo<<"\n";
		}else{
			cout<<"Calibration for left camera started!\n";
			rms_left = calibrateCamera(objectPoints, imagePoints[0], imageSize, cameraMatrix[0],
					distCoeffs[0], rvecs, tvecs,
					CV_CALIB_FIX_K4| CV_CALIB_FIX_K5);
			cout<<"Calibration for left camera ended! RMS error:" <<rms_left<<"\n";
			cout<<"Calibration for right camera started!\n";
			rms_right = calibrateCamera(objectPoints, imagePoints[1], imageSize, cameraMatrix[1],
					distCoeffs[1], rvecs, tvecs,
					CV_CALIB_FIX_K4| CV_CALIB_FIX_K5);
			cout<<"Calibration for right camera ended! RMS error:" <<rms_right<<"\n";
			cout<<"Calibration for stereo started!\n";
			rms_stereo = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
					cameraMatrix[0], distCoeffs[0],
					cameraMatrix[1], distCoeffs[1],
					imageSize, R, T, E, F,
					TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5),
					CV_CALIB_FIX_INTRINSIC);
			cout<<"Calibration for stereo ended! RMS error:" <<rms_stereo<<"\n";
		}
		imagePoints[0].clear();
		imagePoints[1].clear();
		objectPoints.clear();
		this->info();
		this->save_data();
		saved_data=0;
		calib_stereo=true;
		this->init_undistort();
		break;
	}


}

void eye_stereo_calibrate::save_data(){
	Size imageSize=mat_left->size();
	FileStorage fs("intrinsics_eye.yml", CV_STORAGE_WRITE);
	if( fs.isOpened() )
	{
		fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
				"M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
		fs.release();
	}
	else
		cout << "Error: can not save the intrinsic parameters\n";

	Mat R1, R2, P1, P2, Q;
	Rect validRoi[2];

	stereoRectify(cameraMatrix[0], distCoeffs[0],
			cameraMatrix[1], distCoeffs[1],
			imageSize, R, T, R1, R2, P1, P2, Q,
			CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);

	fs.open("extrinsics_eye.yml", CV_STORAGE_WRITE);
	if( fs.isOpened() )
	{
		fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
		fs.release();
	}
	else
		cout << "Error: can not save the extrinsic parameters\n";

	cout << "Saved parameters!\n";

}

void eye_stereo_calibrate::init_undistort(){
	Mat R1, R2, P1, P2, Q;
	Size imageSize=mat_left->size();
	Rect roi1, roi2;

	stereoRectify( cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1], imageSize, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, imageSize, &roi1, &roi2 );

	initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
	initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);
}

void eye_stereo_calibrate::undistort(){
	remap(*mat_left, *calib_mat_left, rmap[0][0], rmap[0][1], INTER_LINEAR);
	remap(*mat_right, *calib_mat_right, rmap[1][0], rmap[1][1], INTER_LINEAR);
}

int main(){

	int key_pressed=255;
	eye_stereo_calibrate *eye_stereo = new eye_stereo_calibrate();

	eye_stereo->info();
	while(1){
		eye_stereo->refresh_frame();
		eye_stereo->detect_chessboard();
		eye_stereo->refresh_window();

		key_pressed = cvWaitKey(10) & 255;
		if ( key_pressed == 27 ) break; //esc
		if ( key_pressed == 99 ) {eye_stereo->save_snapshot();} //c
		if ( key_pressed == 91 ) {eye_stereo->pattern_choice=0;cout<<"Square pattern selected!"<<"\n";} //[
		if ( key_pressed == 93 ) {eye_stereo->pattern_choice=1;cout<<"Circle pattern selected!"<<"\n";} //]
		if ( key_pressed == 48 ) {eye_stereo->camera_choice=0;cout<<"Left camera selected!"<<"\n";} //0
		if ( key_pressed == 49 ) {eye_stereo->camera_choice=1;cout<<"Right camera selected!"<<"\n";} //1
		if ( key_pressed == 50 ) {eye_stereo->camera_choice=2;cout<<"Stereo camera selected!"<<"\n";} //2
		if ( key_pressed == 97 ) {eye_stereo->add_calibration_data();} //a
		if ( key_pressed == 115 ) {eye_stereo->calibrate();} //s

		if(eye_stereo->calib_stereo){eye_stereo->undistort();}

	}

	delete(eye_stereo);
	return 0;
}
