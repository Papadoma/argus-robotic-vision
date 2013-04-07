/**
	Argus System
    tracking3D.cpp
    Purpose: Tracks selectable objects in 3D space

    @author Miltiadis-Alexios Papadopoulos
 */

#include "module_input.hpp"

bool marking_procedure = false;
cv::Rect marker;
cv::Rect clearview_mask;

struct tracking_objects{
	bool on_tracking;
	cv::Rect bounding_rect;
	cv::Point3f center;
	cv::Point center2d;

	cv::Mat img;
	cv::Mat depth;
	cv::Mat mask;

	cv::MatND hist;
	std::deque<cv::Point3f> buf;
};

class tracking{
private:
	module_eye* test;

	cv::Mat rect_mat_left,rect_mat_right;

	cv::Mat cameraMatrix[2], distCoeffs[2];
	cv::Mat R, T, E, F, Q;
	cv::Mat R1, R2, P1, P2;
	cv::Mat rmap[2][2];
	cv::Rect roi1, roi2;

	void load_param();
	cv::MatND get_hist(cv::Mat);
	cv::Point3f get_center(cv::Rect);

	cv::StereoSGBM sgbm;
	int numberOfDisparities;
public:
	std::vector<tracking_objects> object_list;
	int width,height;

	tracking();

	void refresh_frame(bool);
	void track_objects();
	void refresh_window();

};

/**
 * Actions to be made on mouse click. Initializes tracking
 * of new objects.
 */
static void onMouse( int event, int x, int y, int, void* ptr)
{
	switch (event){
	case CV_EVENT_LBUTTONDOWN:
	{
		marking_procedure = true;
		marker = cv::Rect();
		marker.x = x;
		marker.y = y;
	}
	break;
	case CV_EVENT_LBUTTONUP:
	{
		marking_procedure = false;
		marker.width = x - marker.x;
		marker.height = y - marker.y;
		marker = clearview_mask & marker;
		if(marker!=cv::Rect()){
			tracking_objects obj;
			obj.bounding_rect = marker;
			obj.center2d = marker.tl() - marker.br();
			((tracking*)ptr)->object_list.push_back(obj);
		}
	}
	break;
	}

}

/**
 * Constructor
 */
tracking::tracking(){
	cv::namedWindow("Camera");
	cv::setMouseCallback( "Camera", onMouse, (void *)this );

	test =new module_eye("left.mpg","right.mpg");

	height = test->getSize().height;
	width = test->getSize().width;

	load_param();
	clearview_mask = roi1 & roi2;

	numberOfDisparities=32;
	sgbm.preFilterCap = 63; //previously 31
	sgbm.SADWindowSize = 3;
	int cn = 3;
	sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.minDisparity = 0;
	sgbm.numberOfDisparities = numberOfDisparities;
	sgbm.uniquenessRatio = 15;
	sgbm.speckleWindowSize = 100;//previously 50
	sgbm.speckleRange = 32;
	sgbm.disp12MaxDiff = 2;
	sgbm.fullDP = true;
}

/*
 * Loads necessary parameters
 */
void tracking::load_param(){
	bool flag1 = false;
	bool flag2 = false;
	std::string intrinsics="intrinsics_eye.yml";
	std::string extrinsics="extrinsics_eye.yml";
	cv::Size imageSize(width,height);
	cv::FileStorage fs(intrinsics, CV_STORAGE_READ);
	if( fs.isOpened() )
	{
		fs["M1"] >> cameraMatrix[0];
		fs["D1"] >> distCoeffs[0] ;
		fs["M2"] >> cameraMatrix[1];
		fs["D2"] >> distCoeffs[1] ;
		fs.release();
		flag1 = true;
	}
	else
		std::cout << "Error: can not load the intrinsic parameters" << std::endl;
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
		flag2 = true;
	}
	else
		std::cout << "Error: can not load the extrinsics parameters" << std::endl;

	if(flag1&&flag2){
		cv::Mat Q_local;
		stereoRectify( cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1], imageSize, R, T, R1, R2, P1, P2, Q_local, cv::CALIB_ZERO_DISPARITY, -1, imageSize, &roi1, &roi2 );
		initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
		initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);
	}

}

/**
 * Refrashes frame and object data
 */
void tracking::refresh_frame(bool pause){
	cv::Mat left, right;
	bool flag;
	if(!pause)flag = test->getFrame(left, right);

	if(flag){
		remap(left, rect_mat_left, rmap[0][0], rmap[0][1], CV_INTER_LINEAR);
		remap(right, rect_mat_right, rmap[1][0], rmap[1][1], CV_INTER_LINEAR);
	}

	for(int i=0;i<(int)object_list.size();i++){
		object_list[i].img = rect_mat_left(object_list[i].bounding_rect).clone();

		if(object_list[i].hist.empty()){
			object_list[i].hist = get_hist(object_list[i].img);
		}

		object_list[i].center = get_center(object_list[i].bounding_rect);

		object_list[i].buf.push_back(object_list[i].center);
		if(object_list[i].buf.size()>100)object_list[i].buf.pop_front();
	}
}

cv::MatND tracking::get_hist(cv::Mat image){
	int hbins = 30, sbins = 32, vbins = 32;
	int histSize[] = {hbins, sbins, vbins};
	float hranges[] = { 0, 180 };
	float sranges[] = { 0, 256 };
	float vranges[] = { 0, 256 };
	const float* ranges[] = { hranges, sranges, vranges };

	int channels[] = {0, 1, 2};

	cv::Mat hsv;
	cv::MatND result;
	cv::cvtColor(image, hsv, CV_BGR2HSV);

	calcHist( &hsv, 1, channels, cv::Mat(), result, 3, histSize, ranges, true, false );

	return result;
}

cv::Point3f tracking::get_center(cv::Rect rect){
	cv::Point3f result;
	cv::Rect refined_rect = rect;
	refined_rect.x = refined_rect.x-numberOfDisparities;
	refined_rect.width = refined_rect.width+numberOfDisparities;

	cv::Mat object_left=(rect_mat_left)(refined_rect);
	cv::Mat object_right=(rect_mat_right)(refined_rect);

	cv::Mat depth;
	sgbm(object_left,object_right,depth);
	depth = (depth)(cv::Rect(numberOfDisparities, 0, rect.width, rect.height)).clone();
	depth.convertTo(depth, CV_32FC1, 1./16);
	cv::Mat mask;
	cv::threshold(depth,mask,0,255,cv::THRESH_BINARY);
	mask.convertTo(mask,CV_8UC1);
	cv::Mat point_cloud;
	cv::reprojectImageTo3D(depth,point_cloud,Q);
	cv::Scalar mean_values = mean(point_cloud,mask);
	result.x = mean_values.val[0];
	result.y = mean_values.val[1];
	result.z = mean_values.val[2];
	return result;
}

/**
 * Refreshes window for viewing purposes
 */
void tracking::refresh_window(){
	cv::rectangle(rect_mat_left, clearview_mask, cv::Scalar(255,255,255), 1, 8);
	cv::rectangle(rect_mat_right, clearview_mask, cv::Scalar(255,255,255), 1, 8);

	cv::Mat imgResult(height,2*width,CV_8UC3); // Your final image
	cv::Mat roiImgResult_Left = imgResult(cv::Rect(0,0,rect_mat_left.cols,rect_mat_left.rows));
	cv::Mat roiImgResult_Right = imgResult(cv::Rect(rect_mat_right.cols,0,rect_mat_right.cols,rect_mat_right.rows));
	rect_mat_left.copyTo(roiImgResult_Left);
	rect_mat_right.copyTo(roiImgResult_Right);

	for(int i=0;i<(int)object_list.size();i++){
		cv::rectangle(imgResult,object_list[i].bounding_rect,cv::Scalar(0,255,0),1);
		cv::Point3f buf_mean(0,0,0);
		for(int j=0; j<(int)object_list[i].buf.size();j++){
			buf_mean = buf_mean + object_list[i].buf[i];
		}
		buf_mean.x = buf_mean.x/object_list[i].buf.size();
		buf_mean.y = buf_mean.y/object_list[i].buf.size();
		buf_mean.z = buf_mean.z/object_list[i].buf.size();
		std::stringstream ss;
		ss<<buf_mean;
		cv::putText(imgResult,ss.str(),object_list[i].center2d,cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(255,0,255),1);

	}

	imshow( "Camera", imgResult );
}

/**
 * Tracks selected objects using the Camshift algorithm
 */
void tracking::track_objects(){
	float hranges[] = { 0, 180 };
	float sranges[] = { 0, 256 };
	float vranges[] = { 0, 256 };
	const float* ranges[] = { hranges, sranges, vranges };
	int channels[] = {0, 1, 2};
	cv::Mat hsv;
	cvtColor(rect_mat_left, hsv, CV_BGR2HSV);

	cv::TermCriteria criteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 );


	for(int i=0;i<(int)object_list.size();i++){
		cv::MatND  backproj_img;
		calcBackProject(&hsv, 1, channels, object_list[i].hist, backproj_img, ranges, 2, true );

		if(object_list[i].bounding_rect.area()){
			cv::RotatedRect detection = CamShift(backproj_img, object_list[i].bounding_rect, criteria);
			object_list[i].bounding_rect = detection.boundingRect();
			object_list[i].bounding_rect = object_list[i].bounding_rect & clearview_mask;
			object_list[i].center2d = 0.5*object_list[i].bounding_rect.br() +  0.5*object_list[i].bounding_rect.tl();
		}else{
			object_list.erase (object_list.begin()+i);
		}
	}
}

int main(){
	tracking inst;
	bool pause = false;
	while(1){

		inst.refresh_frame(pause);
		inst.track_objects();
		inst.refresh_window();

		int key_pressed = cvWaitKey(1) & 255;
		if ( key_pressed == 27 ) break;
		if ( key_pressed == 32 ) pause=!pause;
		if ( key_pressed == 'c' ) inst.object_list.clear();

	}
	return 0;
}
