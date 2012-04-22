#include "cv.h"
#include "highgui.h"
#include <stdio.h>
#include <pthread.h>

using namespace std;
using namespace cv;

class parallel_eye_stereo_match{
private:
	static VideoCapture* capture_left;
	static VideoCapture* capture_right;
	static Mat* mat_left;
	static Mat* mat_right;
	static Mat* rect_mat_left;
	static Mat* rect_mat_right;
	static Mat* depth_map;

	static Mat rmap[2][2];

	StereoSGBM sgbm;

	int width;
	int height;

	pthread_t threads[4];

	void load_param();
	static void *refresh_frame(void *);

public:

	parallel_eye_stereo_match();
	~parallel_eye_stereo_match();


	void refresh_window();
	void compute_depth();

	void info();
};

//Constructor
parallel_eye_stereo_match::parallel_eye_stereo_match(){
	width = 640;
	height = 480;


	mat_left=new Mat(height,width,CV_8UC1);
	mat_right=new Mat(height,width,CV_8UC1);
	rect_mat_left=new Mat(height,width,CV_8UC1);
	rect_mat_right=new Mat(height,width,CV_8UC1);
	depth_map=new Mat(height,width,CV_16UC1);

	capture_left = new VideoCapture(1);
	capture_right = new VideoCapture(0);

	capture_left->set(CV_CAP_PROP_FRAME_WIDTH, width);
	capture_left->set(CV_CAP_PROP_FRAME_HEIGHT, height);

	capture_right->set(CV_CAP_PROP_FRAME_WIDTH, width);
	capture_right->set(CV_CAP_PROP_FRAME_HEIGHT, height);

	namedWindow("original_camera_left",CV_WINDOW_AUTOSIZE);
	namedWindow("original_camera_right",CV_WINDOW_AUTOSIZE);
	namedWindow("camera_left",CV_WINDOW_AUTOSIZE);
	namedWindow("camera_right",CV_WINDOW_AUTOSIZE);
	namedWindow("depth",CV_WINDOW_AUTOSIZE);
	this->load_param();

	sgbm.preFilterCap = 63;
	sgbm.SADWindowSize = 3;
	int cn = 1;
	sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.minDisparity = 0;
	sgbm.numberOfDisparities = 64;
	sgbm.uniquenessRatio = 10;
	sgbm.speckleWindowSize = 100;
	sgbm.speckleRange = 32;
	sgbm.disp12MaxDiff = 1;
	sgbm.fullDP = true;

	for(int i=0;i<2;i++){
		pthread_create(&threads[i], NULL, &parallel_eye_stereo_match::refresh_frame, (void *)i);
	}
}

//Destructor
parallel_eye_stereo_match::~parallel_eye_stereo_match(){
	destroyWindow("original_camera_left");
	destroyWindow("original_camera_right");
	destroyWindow("camera_left");
	destroyWindow("camera_right");
	destroyWindow("depth");
	delete(capture_left);
	delete(capture_right);

}

void *parallel_eye_stereo_match::refresh_frame(void *k){
	int camera;
	camera=*(int*)k;
	while(1){
		if(camera==0&&capture_left->isOpened()){
			capture_left->grab();
			capture_left->retrieve(*mat_left);
			cvtColor(*mat_left, *mat_left, CV_RGB2GRAY);
			remap(*mat_left, *rect_mat_left, rmap[0][0], rmap[0][1], CV_INTER_LINEAR);

		}else if(capture_right->isOpened()&&camera==1){
			capture_right->grab();
			capture_right->retrieve(*mat_right);
			cvtColor(*mat_right, *mat_right, CV_RGB2GRAY);
			remap(*mat_right, *rect_mat_right, rmap[1][0], rmap[1][1], CV_INTER_LINEAR);
		}
	}
}
/*
void parallel_eye_stereo_match::init_stream(){
	for(int i=0;i<2;i++){
		 pthread_create(&threads[i], NULL, parallel_eye_stereo_match::refresh_frame, (void *)i);
	}
}
 */
void parallel_eye_stereo_match::refresh_window(){
	imshow( "original_camera_left", *mat_left );
	imshow( "original_camera_right", *mat_right );
	imshow( "camera_left", *rect_mat_left );
	imshow( "camera_right", *rect_mat_right );
	imshow( "depth", *depth_map );
}

void parallel_eye_stereo_match::load_param(){

	bool flag1=false;
	bool flag2=false;

	string intrinsics="intrinsics_eye.yml";
	string extrinsics="extrinsics_eye.yml";

	Mat cameraMatrix[2], distCoeffs[2];
	Mat R, T, E, F;
	Mat R1, R2, P1, P2, Q;
	Rect roi1, roi2;
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

		//stereoRectify( cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1], imageSize, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, imageSize, &roi1, &roi2 );

		initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
		initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);
	}

}

void parallel_eye_stereo_match::info(){

	//cout<<capture_left->get(CV_CAP_PROP_POS_FRAMES)<<"\n";
}

void parallel_eye_stereo_match::compute_depth(){

	sgbm(*rect_mat_left,*rect_mat_right,*depth_map);
	depth_map->convertTo(*depth_map, CV_8U, 255/(sgbm.numberOfDisparities*16.));
}

int main(){

	int key_pressed=255;
	parallel_eye_stereo_match eye_stereo;



	while(1){


		eye_stereo.refresh_window();
		eye_stereo.compute_depth();
		key_pressed = cvWaitKey(10) & 255;
		if ( key_pressed == 27 ) break;

	}

	//delete(eye_stereo);
	pthread_exit(NULL);
	return 0;
}
