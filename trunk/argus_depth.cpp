#include <boost/thread.hpp>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/ocl/ocl.hpp>

#define USE_GPU false
#define DEBUG_MODE true
#define HUMAN_DET_RATE 10
#define DEPTH_COLOR_SRC false
#define SEGMENTATION_CONTOURS_SIZE 0.05

#include "module_input.hpp"
#include "pose_estimation.hpp"

cv::Mat old_depth;

struct human_struct{
	float propability;				//Propability that a detection is indeed a human
	cv::Rect body_bounding_rect;	//Bounding rectagle of the human
	cv::Point human_center;			//2D center of bounding rectangle
	cv::Rect head_bounding_rect;	//Bounding rectangle of head (if detected)
	cv::Point head_center;			//2D center of bounding rectangle of head
};

struct user_struct :public human_struct{
	cv::Point3f position_3d;		//3D position of user in its mask mass center, as acquired by reprojectImageto3D
	cv::Mat disparity;				//disparity map
	cv::Mat disparity_viewable;		//disparity map, scaled to CV_8UC1
	cv::Mat point_cloud;			//3D points of user scene, as acquired by reprojectImageto3D

	cv::Mat user_mask;				//The mask which seperates user from background
	cv::Point mask_mass_center;		//Mass center of mask
};

class argus_depth{
private:
	//boost::mutex imshow_mutex;

	module_eye* input_module;
	pose_estimator* pose_tracker;

	std::vector<human_struct> human_group;	//vector of possible humans
	user_struct user;						//final detected human

	cv::MatND user_hist;

	cv::Scalar ulimits;
	cv::Scalar llimits;

	cv::Scalar ulimits2;
	cv::Scalar llimits2;

	cv::Mat cameraMatrix[2], distCoeffs[2];
	cv::Mat R, T, E, F, Q;
	cv::Mat R1, R2, P1, P2;

	cv::Mat mat_left;
	cv::Mat mat_right;
	cv::Mat rect_mat_left;
	cv::Mat rect_mat_right;
	cv::Mat BW_rect_mat_left;
	cv::Mat BW_rect_mat_right;

	cv::Rect roi1, roi2;
	cv::Mat rmap[2][2];

	cv::StereoSGBM sgbm;
	cv::CascadeClassifier cas_cla;
#if USE_GPU
	cv::ocl::HOGDescriptor hog;
	cv::ocl::oclMat  frame_ocl;
#else
	cv::HOGDescriptor hog;
#endif

	cv::Rect clearview_mask;

	bool tracking;			//Tracking flag, to be enabled after 1st allignment

	int numberOfDisparities;
	int width;
	int height;
	int frame_counter;
	int fps;

	void load_param();

	unsigned int fast_distance(cv::Point, cv::Point);


	void debug_detected_human();
	void debug_detected_user();
	void cloud_to_disparity(cv::Mat&, cv::Mat);

	void refresh_frame();
	void refresh_window();

	void detect_human();
	cv::Mat find_skin(cv::Mat);
	void compute_depth();
	void smooth_depth_map(cv::Mat&);
	void segment_user();

	void camera_view(cv::Mat& , cv::Mat& , cv::Mat& , cv::Mat& , cv::Mat& , cv::Mat&, cv::Mat&, cv::Mat&, cv::Mat&);
	void lookat(cv::Point3d from, cv::Point3d to, cv::Mat& destR);
	void eular2rot(double yaw,double pitch, double roll,cv::Mat& dest);
	int Xu_value, Xl_value,Yu_value,Yl_value,Zu_value,Zl_value;
	int canny_up, canny_dn;


public:
	double baseline;
	cv::Point3d viewpoint;
	cv::Point3d lookatpoint;

	argus_depth();
	~argus_depth();

	void start();
	void take_snapshot();
	bool detect_user_flag;

};


//Constructor
argus_depth::argus_depth()
:frame_counter(0)
,tracking(false)
,detect_user_flag(true)
{
	std::cout << "[Argus] Initializing detectors" << std::endl;
	if(!cas_cla.load("haarcascade_frontalface_alt.xml")){
		std::cout << "[Argus] Cascade not found" << std::endl;
		exit(1);
	}else{
		std::cout << "[Argus] Cascade loaded" << std::endl;
	}
#if USE_GPU
	hog = cv::ocl::HOGDescriptor(cv::Size(48, 96));
	hog.setSVMDetector(cv::ocl::HOGDescriptor::getPeopleDetector48x96());//getPeopleDetector48x96, getPeopleDetector64x128 or getDefaultPeopleDetector
#else
	hog.winSize = cv::Size(48, 96);
	hog.setSVMDetector(cv::HOGDescriptor::getDaimlerPeopleDetector());//getPeopleDetector48x96, getPeopleDetector64x128 or getDefaultPeopleDetector
#endif

	input_module = new module_eye("left.mpg","right.mpg");
	cv::Size framesize = input_module->getSize();
	height = framesize.height;
	width = framesize.width;



	user.body_bounding_rect = cv::Rect(width/2,height/2,10,10);
	user.propability = -1;

	baseline = 9.5;
	viewpoint = cv::Point3d(0.0,0.0,baseline*10);
	lookatpoint = cv::Point3d(0.0,0.0,-baseline*10.0);

#if USE_GPU
	std::cout << "Begin creating ocl context..." << std::endl;
	std::vector<cv::ocl::Info> ocl_info;
	int devnums=getDevice(ocl_info);
	std::cout << "End creating ocl context...\n" << std::endl;
	if(devnums<1){
		std::cout << "no OPENCL device found\n";
	}
	frame_ocl = cv::ocl::oclMat(height,width,CV_8UC1);
#endif

	this->load_param();

	numberOfDisparities=32;
	sgbm.preFilterCap = 63; //previously 31
	sgbm.SADWindowSize = 3;
	int cn = 1;
	sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.minDisparity = 0;
	sgbm.numberOfDisparities = numberOfDisparities;
	sgbm.uniquenessRatio = 15;
	sgbm.speckleWindowSize = 100;//previously 50
	sgbm.speckleRange = 32;
	sgbm.disp12MaxDiff = 2;
	sgbm.fullDP = false;

	pose_tracker = new pose_estimator(width,height,numberOfDisparities);

	//clearview_mask= cv::Rect(numberOfDisparities,0,width,height);
	clearview_mask = roi1 & roi2;

	Xu_value=5,Xl_value=5,Yu_value=20,Yl_value=20,Zu_value=30,Zl_value=60;
	canny_up = 20, canny_dn =60;

	cv::namedWindow("XYZ floodfill",CV_WINDOW_NORMAL );

	cv::createTrackbar("canny up", "XYZ floodfill", &canny_up, 200);
	cv::createTrackbar("canny down", "XYZ floodfill", &canny_dn, 200);

	cv::createTrackbar("Xupper", "XYZ floodfill", &Xu_value, 500);
	cv::createTrackbar("Xlower", "XYZ floodfill", &Xl_value, 500);
	cv::createTrackbar("Yupper", "XYZ floodfill", &Yu_value, 500);
	cv::createTrackbar("Ylower", "XYZ floodfill", &Yl_value, 500);
	cv::createTrackbar("Zupper", "XYZ floodfill", &Zu_value, 500);
	cv::createTrackbar("Zlower", "XYZ floodfill", &Zl_value, 500);

	user.user_mask = cv::Mat::zeros(height,width,CV_8UC1);
	user.disparity_viewable = cv::Mat::zeros(height,width,CV_8UC1);
}

//Destructor
argus_depth::~argus_depth(){
	delete(input_module);
	delete(pose_tracker);
	cv::destroyAllWindows();

}

inline unsigned int argus_depth::fast_distance(cv::Point P1, cv::Point P2){
	unsigned int a,b;
	unsigned x = pow((P1.x - P2.x),2) + pow((P1.y - P2.y),2);
	b     = x;
	a = x = 0x3f;
	x     = b/x;
	a = x = (x+a)>>1;
	x     = b/x;
	a = x = (x+a)>>1;
	x     = b/x;
	x     = (x+a)>>1;
	return(x);
}

inline void argus_depth::debug_detected_user(){
	cv::Mat user_frame = cv::Mat::zeros(rect_mat_left.rows, 2*rect_mat_left.cols, CV_8UC3);
	cv::Mat ROI1 = user_frame(user.body_bounding_rect);
	cv::Mat ROI2 = user_frame(user.body_bounding_rect + cv::Point(rect_mat_left.cols,0));

	(rect_mat_right(user.body_bounding_rect)).copyTo(ROI1);
	applyColorMap(user.disparity_viewable(user.body_bounding_rect), ROI2, cv::COLORMAP_JET);
	imshow("detected user", user_frame);
	imshow("final mask", user.user_mask);
}

inline void argus_depth::refresh_frame(){
	if(input_module->getFrame(mat_left,mat_right)){

		remap(mat_left, rect_mat_left, rmap[0][0], rmap[0][1], CV_INTER_LINEAR);
		remap(mat_right, rect_mat_right, rmap[1][0], rmap[1][1], CV_INTER_LINEAR);
	}else{
		input_module = new module_eye("left.mpg","right.mpg");
	}
	cvtColor(rect_mat_left,BW_rect_mat_left,CV_BGR2GRAY);
	cvtColor(rect_mat_right,BW_rect_mat_right,CV_BGR2GRAY);

	frame_counter++;

#if USE_GPU
	cv::Mat local;
	cv::cvtColor(rect_mat_left,local,CV_BGR2BGRA);
	frame_ocl.upload(local);
#endif
}

inline void argus_depth::refresh_window(){
	cv::rectangle(rect_mat_left, clearview_mask, cv::Scalar(255,255,255), 1, 8);
	cv::rectangle(rect_mat_right, clearview_mask, cv::Scalar(255,255,255), 1, 8);

	cv::Mat imgResult(height,2*width,CV_8UC3); // Your final image
	cv::Mat roiImgResult_Left = imgResult(cv::Rect(0,0,rect_mat_left.cols,rect_mat_left.rows));
	cv::Mat roiImgResult_Right = imgResult(cv::Rect(rect_mat_right.cols,0,rect_mat_right.cols,rect_mat_right.rows));
	rect_mat_left.copyTo(roiImgResult_Left);
	rect_mat_right.copyTo(roiImgResult_Right);

	std::stringstream ss;//create a stringstream
	ss << fps;//add number to the stream
	cv::putText(imgResult, ss.str(), cv::Point (10,20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,255), 2, 8, false );

	imshow( "Camera", imgResult );


#if DEBUG_MODE	//Show the detected humans on screen
	debug_detected_human();
	debug_detected_user();
#endif
}

inline void argus_depth::debug_detected_human(){
	cv::Mat human_frame = rect_mat_left.clone();
	for( int i = 0; i < (int)human_group.size(); i++ )
	{
		cv::Scalar color = cv::Scalar((6*i*255/human_group.size())%255,8*(255-i*255/human_group.size())%255,i*10/human_group.size()%255);
		rectangle(human_frame, human_group[i].body_bounding_rect.tl(), human_group[i].body_bounding_rect.br(), color, 2);
		if(human_group[i].head_bounding_rect != cv::Rect())rectangle(human_frame, human_group[i].head_bounding_rect.tl(), human_group[i].head_bounding_rect.br(), color, 2);
		std::stringstream ss;
		ss<<human_group[i].propability;
		cv::putText(human_frame, ss.str(),human_group[i].human_center,0,0.5,cv::Scalar(255,255,255));
	}
	imshow("detected humans", human_frame);
}

/*
 * Loads necessary parameters
 */
void argus_depth::load_param(){
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
		cv::Mat Q_local=Q.clone();
		stereoRectify( cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1], imageSize, R, T, R1, R2, P1, P2, Q_local, cv::CALIB_ZERO_DISPARITY, -1, imageSize, &roi1, &roi2 );
		//getOptimalNewCameraMatrix(cameraMatrix[0], distCoeffs[0], imageSize, 1, imageSize, &roi1);
		//getOptimalNewCameraMatrix(cameraMatrix[1], distCoeffs[1], imageSize, 1, imageSize, &roi2);
		initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
		initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);
	}
}

/*
 * Compute the persons depth.
 */
inline void argus_depth::compute_depth(){
	//Will have to widen the user bounding rectangle a bit, so we wont loose any disparity
	cv::Rect refined_human_anchor = user.body_bounding_rect;
	refined_human_anchor.x = refined_human_anchor.x-numberOfDisparities;
	refined_human_anchor.width = refined_human_anchor.width+numberOfDisparities;
	refined_human_anchor = refined_human_anchor & clearview_mask;


#if DEPTH_COLOR_SRC
	cv::Mat person_left=(rect_mat_left)(refined_human_anchor).clone();
	cv::Mat person_right=(rect_mat_right)(refined_human_anchor).clone();
#else
	cv::Mat person_left=(BW_rect_mat_left)(refined_human_anchor).clone();
	cv::Mat person_right=(BW_rect_mat_right)(refined_human_anchor).clone();
#endif


	cv::GaussianBlur(person_left,person_left,cv::Size(0,0),1,0);
	cv::GaussianBlur(person_right,person_right,cv::Size(0,0),1,0);
	cv::Mat local_depth;
	sgbm(person_left,person_right,local_depth);							//Compute disparity
	local_depth.convertTo(local_depth, CV_32FC1, 1./16);				//Scale down to normal disparity
	//smooth_depth_map(local_depth);

	user.disparity = cv::Mat::zeros(rect_mat_left.size(), CV_32FC1);	//Prepare for copying
	local_depth.copyTo(user.disparity(refined_human_anchor));			//Copy disparity
	reprojectImageTo3D(user.disparity, user.point_cloud, Q, false, -1);	//Get the point cloud in WCS
	user.disparity.convertTo(user.disparity_viewable, CV_8UC3, 255./(numberOfDisparities));	//Get the viewable version of disparity
}

/*
 * Smooth out holes using inpaint technique.
 */
inline void argus_depth::smooth_depth_map(cv::Mat& depth_map){
	//	cv::Mat holes;
	//	threshold(depth_map,holes,1,255,cv::THRESH_BINARY_INV); //keep only the holes
	//	inpaint(depth_map, holes, depth_map, 2.0 , cv::INPAINT_NS );
	//	cv::Mat local = depth_map.clone();
	//	bilateralFilter(local, depth_map, 3, 30, 30);
	//cv::medianBlur(depth_map,depth_map,3);
	cv::Mat holes;
	threshold(depth_map,holes,1,255,cv::THRESH_BINARY_INV); //keep only the holes
	imshow("holes",holes);
}

/*
 * Returns binary image mask on skin range.
 */
inline cv::Mat argus_depth::find_skin(cv::Mat input){
	cv::Mat result;
	cvtColor(input, result, CV_BGR2YCrCb);
	inRange(result, cv::Scalar(60, 130, 80), cv::Scalar(255, 170, 125), result);
	return result;
}

/*
 * Detects possible humans based on 2D color information.
 */
inline void argus_depth::detect_human(){
	human_group.clear();

	std::vector<cv::Rect> found, found_filtered;
	std::vector<cv::Rect> faces;

	cas_cla.detectMultiScale(rect_mat_left,faces, 1.5, 3);
#if USE_GPU
	hog.detectMultiScale(frame_ocl, found, 0.6, cv::Size(8,8), cv::Size(0,0), 1.10, 1);
#else
	hog.detectMultiScale(BW_rect_mat_left, found, 0, cv::Size(8,8), cv::Size(32,32), 1.08, 1);
#endif

	//filter double human detections
	int i, j;
	for(int i = 0; i < (int)found.size(); i++ )
	{
		cv::Rect r = found[i];
		for( j = 0; j < (int)found.size(); j++ )
			if( j != i && (r & found[j]) == r)
				break;
		if( j == (int)found.size() )
			found_filtered.push_back(r);
	}

	/*
	 * Assign faces to detected bodies. Choose the best
	 * face by measuring the distance of the face from
	 * the body center.
	 */
	for( i = 0; i < (int)found_filtered.size(); i++ )
	{
		human_struct human_candidate;
		human_candidate.propability = 0;
		human_candidate.body_bounding_rect =(clearview_mask) & found_filtered[i];
		human_candidate.human_center = cv::Point(human_candidate.body_bounding_rect.x + human_candidate.body_bounding_rect.width/2, human_candidate.body_bounding_rect.y + human_candidate.body_bounding_rect.height/2);
		for(j = 0; j<(int)faces.size();j++)
		{
			cv::Point head_candidate = cv::Point(faces[j].x + faces[j].width/2, faces[j].y + faces[j].height/2);
			if( human_candidate.body_bounding_rect.contains(head_candidate) )
			{
				if(human_candidate.head_bounding_rect == cv::Rect())		//head not yet assigned
				{
					human_candidate.head_bounding_rect = faces[j];
					human_candidate.head_center = head_candidate;
				}else if(fast_distance(head_candidate,human_candidate.human_center) > fast_distance(human_candidate.head_center ,human_candidate.human_center)){
					human_candidate.head_bounding_rect = faces[j];
					human_candidate.head_center = cv::Point(human_candidate.head_bounding_rect.x + human_candidate.head_bounding_rect.width/2, human_candidate.head_bounding_rect.y + human_candidate.head_bounding_rect.height/2);
				}
				faces.erase(faces.begin()+j);
			}
		}

		if(human_candidate.head_bounding_rect != cv::Rect()){	//If head is detected
			human_candidate.body_bounding_rect = human_candidate.body_bounding_rect | human_candidate.head_bounding_rect;
			float dist = fast_distance(human_candidate.head_center ,human_candidate.human_center);
			int skin = cv::countNonZero(find_skin(rect_mat_left(human_candidate.head_bounding_rect)));
			human_candidate.propability = 0.5 * 2*dist/(human_candidate.body_bounding_rect.height-human_candidate.head_bounding_rect.height) + 0.5*(float)skin/human_candidate.head_bounding_rect.area();
		}
		human_group.push_back(human_candidate);
	}




	human_struct possible_user;
	possible_user.propability = 0;
	if(human_group.size()>0){
		//find the biggest human
		float max = human_group.at(0).body_bounding_rect.area() * (1 + human_group.at(0).propability);
		int pos = 0;
		for( i = 1; i < (int)human_group.size(); i++ ){
			if (human_group.at(i).body_bounding_rect.area() * (1 + human_group.at(i).propability)>max){
				max = human_group.at(i).body_bounding_rect.area() * (1 + human_group.at(i).propability);
				pos = i;
			}
		}
		possible_user = human_group.at(pos);
	}

	if(possible_user.propability > 0.9 * user.propability){
		tracking = false;
		user.body_bounding_rect = possible_user.body_bounding_rect;
		user.head_center = possible_user.head_center;
		user.head_bounding_rect = possible_user.head_bounding_rect;
		user.human_center = possible_user.human_center;
		user.propability = possible_user.propability;
		//user_hist = cv::Mat::zeros(user_hist.size(), CV_8UC1);
	}

}

/**
 * Segments the user from the overall bounding rectangle. Calculates his mask and mass center
 */
inline void argus_depth::segment_user(){
	ulimits = cv::Scalar(cv::getTrackbarPos("Xupper", "XYZ floodfill"),cv::getTrackbarPos("Yupper", "XYZ floodfill"),cv::getTrackbarPos("Zupper", "XYZ floodfill"));
	llimits = cv::Scalar(cv::getTrackbarPos("Xlower", "XYZ floodfill"),cv::getTrackbarPos("Ylower", "XYZ floodfill"),cv::getTrackbarPos("Zlower", "XYZ floodfill"));

	if(user.mask_mass_center ==  cv::Point() || !user.mask_mass_center.inside(user.body_bounding_rect)){
		user.mask_mass_center = cv::Point(user.body_bounding_rect.x + user.body_bounding_rect.width/2, user.body_bounding_rect.y + user.body_bounding_rect.height/2);
	}

	cv::Mat floodfill_mask = cv::Mat::zeros(height + 2, width + 2, CV_8U);

	if(user.point_cloud.data)floodFill(user.point_cloud, floodfill_mask,user.mask_mass_center , 255, 0, llimits, ulimits, 4 + (255 << 8) + cv::FLOODFILL_MASK_ONLY );

	cv::Rect crop(1,1,floodfill_mask.cols-2,floodfill_mask.rows-2);
	floodfill_mask = floodfill_mask(crop).clone();


	//cv::circle(floodfill_mask, user.mask_mass_center, 5, cv::Scalar(127), CV_FILLED);
	//imshow("human_mask", floodfill_mask);


	//Find edges present in the floodfill's region
	cv::Mat frame_edge_detection;
	Canny(BW_rect_mat_left, frame_edge_detection, canny_up, canny_dn );

	//Sometimes, floodfill fails to return a decent mask. In that case, use the previously detected mask
	if(cv::countNonZero(floodfill_mask) < 0.5*cv::countNonZero(user.user_mask)){
		bitwise_and(frame_edge_detection, user.user_mask, frame_edge_detection);
	}else{
		bitwise_and(frame_edge_detection, floodfill_mask, frame_edge_detection);
	}
	//imshow("edges", frame_edge_detection);


	//Close the edges
	morphologyEx(frame_edge_detection, frame_edge_detection, cv::MORPH_CLOSE, cv::Mat(), cv::Point(), 3 );
	//imshow("edges closed", frame_edge_detection);

	//Eliminate small edge fragments
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours( frame_edge_detection.clone(), contours, hierarchy, CV_RETR_EXTERNAL , CV_CHAIN_APPROX_SIMPLE );
	cv::Mat frame_edge_filled = cv::Mat::zeros( frame_edge_detection.size(), CV_8UC1 );
	for( int i = 0; i< (int)contours.size(); i++ )
	{
		if(contourArea(contours[i]) < user.body_bounding_rect.area()*SEGMENTATION_CONTOURS_SIZE)drawContours( frame_edge_detection, contours, i, cv::Scalar(0), CV_FILLED, 8, hierarchy, 0, cv::Point() );
	}
	//imshow( "Contours", frame_edge_detection );

	//Get only depth information on the user
	cv::Mat user_depth;
	if(user.disparity_viewable.data)cv::bitwise_and(user.disparity_viewable, frame_edge_detection, user_depth);
	//imshow( "user depth", user_depth );

	//Get scene color information in a different colorspace without luminosity
	cv::Mat hsv_user;
	cv::cvtColor(rect_mat_left, hsv_user, CV_BGR2HSV);

	//Mix depth and color information
	cv::Mat hsvd(hsv_user.size() ,CV_8UC3);
	cv::Mat in[]={user.disparity_viewable, hsv_user};
	int from_to[] = { 0,0, 1,1, 2,2 };
	mixChannels( in, 2, &hsvd, 1, from_to, 3 );

	//Prepare for histogram calculation
	int  hbins = 30, sbins = 32, dbins = 32;
	int histSize[] = {hbins, sbins, dbins};
	float hranges[] = { 0, 180 };
	float sranges[] = { 0, 255 };
	float dranges[] = { 0, 255 };
	const float* ranges[] = { hranges, sranges, dranges };
	int channels[] = {0, 1, 2};

	//Calculate back projection (if there is a previously calculated histogram)
	cv::Mat backproj_img;
	if(cv::countNonZero(user_hist)){
		calcBackProject(&hsvd, 1, channels, user_hist, backproj_img, ranges, 2, true );
		//cv::threshold(backproj_img,backproj_img,50,255,cv::THRESH_BINARY);
		//imshow("back projection",backproj_img);
	}

	//Calculate histogram
	calcHist( &hsvd, 1, channels, frame_edge_detection, user_hist, 3, histSize, ranges, true, false );

	if(cv::countNonZero(backproj_img) && user.body_bounding_rect.area()){
		//Use camshift algorithm to track the user and adjust his bounding rectangle
		cv::TermCriteria criteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 );
		cv::Rect local_rect = user.body_bounding_rect;

		//Add previous edge mask information
		morphologyEx(backproj_img, backproj_img, cv::MORPH_CLOSE, cv::Mat(), cv::Point(), 3 );
		cv::bitwise_or(backproj_img, frame_edge_detection, backproj_img);
		cv::RotatedRect detection = CamShift(backproj_img, local_rect, criteria);

		//Shape the detected rectangle a bit
		cv::Rect new_user_rect=detection.boundingRect();
		new_user_rect.x -= new_user_rect.width*0.3/2;
		//new_user_rect.y -= new_user_rect.height*0.05/2;
		new_user_rect.width = new_user_rect.width*1.3;
		new_user_rect.height = new_user_rect.height*1.05;
		new_user_rect &= clearview_mask;
		user.body_bounding_rect = new_user_rect;


		//Introduce an 80% confidence to create the final user mask
		cv::threshold(backproj_img,backproj_img,255*0.8,255,cv::THRESH_BINARY);

		//Eliminate small mask fragments
		contours.clear();
		hierarchy.clear();
		findContours( backproj_img.clone(), contours, hierarchy, CV_RETR_LIST , CV_CHAIN_APPROX_SIMPLE );
		cv::Mat frame_edge_filled = cv::Mat::zeros( frame_edge_detection.size(), CV_8UC1 );
		for( int i = 0; i< (int)contours.size(); i++ )
		{
			if(contourArea(contours[i]) < user.body_bounding_rect.area()*SEGMENTATION_CONTOURS_SIZE)drawContours( backproj_img, contours, i, cv::Scalar(0), CV_FILLED, 8, hierarchy, 0, cv::Point() );
		}
		//Find new mass center, used in floodfill
		cv::Moments mask_moments = moments(backproj_img, true);
		user.mask_mass_center = cv::Point(mask_moments.m10/mask_moments.m00, mask_moments.m01/mask_moments.m00);

		//finally, store the new user mask
		backproj_img.copyTo(user.user_mask);
		cv::bitwise_and(backproj_img,user.disparity_viewable,user.disparity_viewable);
	}
}

void argus_depth::take_snapshot(){
	std::cout<<"Snapshot taken!" << std::endl;
	cv::Mat depth_cropped;
	cv::bitwise_and(user.disparity_viewable, user.user_mask,depth_cropped);
	imwrite("depth.png", depth_cropped);
}

void argus_depth::cloud_to_disparity(cv::Mat& disparity, cv::Mat xyz){
	disparity = cv::Mat(xyz.rows,xyz.cols,CV_32FC1);
	float z,y;
	for(int i=0;i<xyz.rows;i++){
		for(int j=0;j<xyz.cols;j++){
			z=xyz.ptr<float>(i)[3*j+2];
			y=xyz.ptr<float>(i)[3*j+1];
			//			if(y>-95&&z>-350){
			//				disparity.at<float>(i,j)=Q.at<double>(2,3)/z*Q.at<double>(3,2);
			//			}else{
			//				disparity.at<float>(i,j)=0;
			//			}
			disparity.at<float>(i,j)=Q.at<double>(2,3)/z*Q.at<double>(3,2);
		}
	}

	normalize(disparity, disparity, 0.0, 255.0, cv::NORM_MINMAX);
	disparity.convertTo(disparity,CV_8UC1);

	//cvtColor(xyz,xyz,CV_RGB2GRAY);

	imshow("hello",disparity);
}

/*
void argus_depth::lookat(cv::Point3d from, cv::Point3d to, cv::Mat& destR)
{
	double x=(to.x-from.x);
	double y=(to.y-from.y);
	double z=(to.z-from.z);

	double pitch =asin(x/sqrt(x*x+z*z))/CV_PI*180.0;
	double yaw   =asin(-y/sqrt(y*y+z*z))/CV_PI*180.0;

	eular2rot(yaw, pitch, 0,destR);
}

void argus_depth::eular2rot(double yaw,double pitch, double roll, cv::Mat& dest)
{
	double theta = yaw/180.0*CV_PI;
	double pusai = pitch/180.0*CV_PI;
	double phi = roll/180.0*CV_PI;

	double datax[3][3] = {{1.0,0.0,0.0},
			{0.0,cos(theta),-sin(theta)},
			{0.0,sin(theta),cos(theta)}};
	double datay[3][3] = {{cos(pusai),0.0,sin(pusai)},
			{0.0,1.0,0.0},
			{-sin(pusai),0.0,cos(pusai)}};
	double dataz[3][3] = {{cos(phi),-sin(phi),0.0},
			{sin(phi),cos(phi),0.0},
			{0.0,0.0,1.0}};
	cv::Mat Rx(3,3,CV_64F,datax);
	cv::Mat Ry(3,3,CV_64F,datay);
	cv::Mat Rz(3,3,CV_64F,dataz);
	cv::Mat rr=Rz*Rx*Ry;
	rr.copyTo(dest);
}

void argus_depth::camera_view(cv::Mat& image, cv::Mat& destimage, cv::Mat& disp, cv::Mat& destdisp, cv::Mat& xyz, cv::Mat& R, cv::Mat& t, cv::Mat& K, cv::Mat& dist){

	if(destimage.empty())destimage = cv::Mat::zeros(cv::Size(image.size()),image.type());
	if(destdisp.empty())destdisp = cv::Mat::zeros(cv::Size(image.size()),disp.type());
	std::vector<cv::Point2f> pt;
	if(dist.empty()) dist = cv::Mat::zeros(cv::Size(5,1),CV_32F);
	cv::projectPoints(xyz,R,t,K,dist,pt);

	destimage.setTo(0);
	destdisp.setTo(0);


	for(int j=1;j<image.rows-1;j++)
	{
		int count=j*image.cols;
		//uchar* img=image.ptr<uchar>(j);

		for(int i=0;i<image.cols;i++,count++)
		{
			int x=(int)(pt[count].x+0.5);
			int y=(int)(pt[count].y+0.5);

			if(pt[count].x>=1 && pt[count].x<image.cols-1 && pt[count].y>=1 && pt[count].y<image.rows-1)
			{
				short v=destdisp.at<uchar>(y,x);
				if(v<disp.at<uchar>(j,i))
				{
					//					destimage.at<uchar>(y,3*x+0)=img[3*i+0];
					//					destimage.at<uchar>(y,3*x+1)=img[3*i+1];
					//					destimage.at<uchar>(y,3*x+2)=img[3*i+2];
					destdisp.at<uchar>(y,x)=disp.at<uchar>(j,i);
				}
			}
		}
	}

}
 */

void argus_depth::start(){


	refresh_frame();

	double t;
	//boost::thread t3;
	if((frame_counter%HUMAN_DET_RATE == 0)&&(detect_user_flag)){

		detect_human();

		//t3 = boost::thread(boost::bind(&argus_depth::detect_human, this) );
	}
	//boost::thread t1(boost::bind(&argus_depth::compute_depth, this) );
	//boost::thread t2(boost::bind(&argus_depth::segment_user, this) );
//	t1.join();
//	t2.join();
//	t3.join();

	compute_depth();


	segment_user();


	if(tracking){
		//pose_tracker->find_pose(user.disparity_viewable,false);
	}
//	if(!tracking){
//		//pose_tracker = new pose_estimator(width,height,numberOfDisparities);
//
//		tracking = true;
//	}

//	if(frame_counter>100){
//		cv::Mat input_frame = cv::imread("woman_dancing.png",0);
//
//		pose_tracker->find_pose(input_frame,true);
//		cv::waitKey(0);
//	}
	if(!detect_user_flag && !tracking){
		//pose_tracker->find_pose(user.disparity_viewable,true);
		//tracking = true;
	}


	refresh_window();

	t = (double)cv::getTickCount() - t;
	fps = 1/(t/cv::getTickFrequency());//for fps
	//eye_stereo->fps = t*1000./cv::getTickFrequency();//for ms




}

int main(){
	int key_pressed=255;

	argus_depth eye_stereo;

	bool loop=false;
	while(1){
		do{
			key_pressed = cvWaitKey(1) & 255;
			if ( key_pressed == 32 )loop=!loop;
			if ( key_pressed == 27 ) break;
		}while (loop);
		if ( key_pressed == 27 ) break;							//ESC
		if ( key_pressed == 13 ) eye_stereo.take_snapshot();	//ENTER
		if ( key_pressed == 't' ) eye_stereo.detect_user_flag=!eye_stereo.detect_user_flag;
		if ( key_pressed == 'f' ) eye_stereo.detect_user_flag=!eye_stereo.detect_user_flag;


		//		if ( key_pressed == 119 ) eye_stereo.viewpoint.y+=10;	//W
		//		if ( key_pressed == 97 ) eye_stereo.viewpoint.x+=10;		//A
		//		if ( key_pressed == 115 ) eye_stereo.viewpoint.y-=10;		//S
		//		if ( key_pressed == 100 ) eye_stereo.viewpoint.x-=10;		//D

		eye_stereo.start();



		//key_pressed = cvWaitKey(1) & 255;

	}



	return 0;
}
