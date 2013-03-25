#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/ocl/ocl.hpp>
#define USE_GPU true
#define DEBUG_MODE true

#include "module_input.hpp"

struct human{
	float movement_perc;
	float propability;

	cv::Rect bounding_rect;
	cv::Point human_center;

	cv::Rect head_rect;
	cv::Point head_center;

};

class argus_depth{
private:
	module_eye* input_module;



	cv::Mat cameraMatrix[2], distCoeffs[2];
	cv::Mat R, T, E, F, Q;
	cv::Mat R1, R2, P1, P2;

	cv::Mat mat_left;
	cv::Mat mat_right;
	cv::Mat rect_mat_left;
	cv::Mat rect_mat_right;
	cv::Mat BW_rect_mat_left;
	cv::Mat BW_rect_mat_right;

	cv::Mat depth_map;
	cv::Mat depth_map2;

	cv::Mat thres_mask;
	cv::Mat test;

	cv::Mat prev_frame, frame_diff;

	//ocl::oclMat* left_ocl;
	//ocl::oclMat* right_ocl;
	//ocl::oclMat* depth_ocl;

	cv::Rect human_anchor;
	cv::Rect roi1, roi2;
	cv::Mat rmap[2][2];

	cv::StereoSGBM sgbm;
	cv::CascadeClassifier cas_cla;
#if USE_GPU
	cv::ocl::HOGDescriptor hog;
	cv::ocl::oclMat  frame_ocl;
#else
	cv::HOGDescriptor hog;
	cv::CascadeClassifier cas_cla;
#endif




	cv::Rect clearview_mask;

	int numberOfDisparities;
	int width;
	int height;
	int frame_counter;

	void load_param();
	void smooth_depth_map();

	void lookat(cv::Point3d from, cv::Point3d to, cv::Mat& destR);
	void eular2rot(double yaw,double pitch, double roll,cv::Mat& dest);
	void cloud_to_disparity(cv::Mat, cv::Mat);

	unsigned int fast_distance(cv::Point, cv::Point);
	cv::Mat find_skin(cv::Mat);

	void debug_detected_human(std::vector<human>);
public:
	double baseline;
	cv::Point3d viewpoint;
	cv::Point3d lookatpoint;

	argus_depth();
	~argus_depth();
	void clustering();
	int fps;

	//void remove_background();
	void refresh_frame();
	void refresh_window();
	void compute_depth();
	void detect_human();
	void compute_depth_gpu();
	void take_snapshot();
	void camera_view(cv::Mat& , cv::Mat& , cv::Mat& , cv::Mat& , cv::Mat& , cv::Mat&, cv::Mat&, cv::Mat&, cv::Mat&);
};



//Constructor
argus_depth::argus_depth()
:frame_counter(0)
{
	cas_cla.load("haarcascade_frontalface_alt.xml");
#if USE_GPU
	hog = cv::ocl::HOGDescriptor(cv::Size(48, 96));
	hog.setSVMDetector(cv::ocl::HOGDescriptor::getPeopleDetector48x96());//getPeopleDetector48x96, getPeopleDetector64x128 or getDefaultPeopleDetector
#else
	hog = cv::HOGDescriptor(cv::Size(48, 96));
	hog.setSVMDetector(cv::HOGDescriptor::getPeopleDetector48x96());//getPeopleDetector48x96, getPeopleDetector64x128 or getDefaultPeopleDetector
#endif




	input_module = new module_eye("left1.mpg","right1.mpg");
	cv::Size framesize = input_module->getSize();
	height = framesize.height;
	width = framesize.width;

	prev_frame = cv::Mat::zeros(framesize, CV_8UC1);
	human_anchor = cv::Rect(width/2,height/2,10,10);

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
	sgbm.SADWindowSize = 1;
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

	clearview_mask= cv::Rect(numberOfDisparities,0,width,height);
	clearview_mask = roi1 & roi2 & clearview_mask;
}

//Destructor
argus_depth::~argus_depth(){
	cv::destroyAllWindows();
	delete(input_module);
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

inline void argus_depth::debug_detected_human(std::vector<human> list){
	for( int i = 0; i < (int)list.size(); i++ )
	{
		cv::Scalar color = cv::Scalar((6*i*255/list.size())%255,8*(255-i*255/list.size())%255,i*10/list.size()%255);
		rectangle(rect_mat_left, list[i].bounding_rect.tl(), list[i].bounding_rect.br(), color, 2);
		if(list[i].head_rect != cv::Rect())rectangle(rect_mat_left, list[i].head_rect.tl(), list[i].head_rect.br(), color, 2);
		std::stringstream ss;
		ss<<list[i].propability;
		cv::putText(rect_mat_left, ss.str(),list[i].human_center,0,0.5,cv::Scalar(255,255,255));
	}

}

inline void argus_depth::refresh_frame(){
	input_module->getFrame(mat_left,mat_right);

	remap(mat_left, rect_mat_left, rmap[0][0], rmap[0][1], CV_INTER_LINEAR);
	remap(mat_right, rect_mat_right, rmap[1][0], rmap[1][1], CV_INTER_LINEAR);

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

	cv::Mat imgResult(height,2*width,CV_8UC3); // Your final imageasd
	cv::Mat roiImgResult_Left = imgResult(cv::Rect(0,0,rect_mat_left.cols,rect_mat_left.rows));
	cv::Mat roiImgResult_Right = imgResult(cv::Rect(rect_mat_right.cols,0,rect_mat_right.cols,rect_mat_right.rows));
	cv::Mat roiImg1 = (rect_mat_left)(cv::Rect(0,0,rect_mat_left.cols,rect_mat_left.rows));
	cv::Mat roiImg2 = (rect_mat_right)(cv::Rect(0,0,rect_mat_right.cols,rect_mat_right.rows));
	roiImg1.copyTo(roiImgResult_Left);
	roiImg2.copyTo(roiImgResult_Right);

	std::stringstream ss;//create a stringstream
	ss << fps;//add number to the stream
	cv::putText(imgResult, ss.str(), cv::Point (10,20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,255), 2, 8, false );

	imshow( "Camera", imgResult );

	//	cv::Mat jet_depth_map2(height,width,CV_8UC3);
	//	cv::applyColorMap(depth_map2, jet_depth_map2, cv::COLORMAP_JET );
	//	cv::imshow( "depth2", jet_depth_map2 );
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

inline void argus_depth::compute_depth(){
	//	cv::Rect refined_human_anchor=human_anchor;
	//	refined_human_anchor.x=human_anchor.x-numberOfDisparities;
	//	refined_human_anchor.width=human_anchor.width+numberOfDisparities;
	//
	//	person_left=(BW_rect_mat_left)(refined_human_anchor);
	//	person_right=(BW_rect_mat_right)(refined_human_anchor);
	//
	//	sgbm(person_left,person_right,depth_map);
	//	depth_map.convertTo(depth_map, CV_8UC1, 255/(numberOfDisparities*16.));
	//
	//	depth_map = (depth_map)(cv::Rect(numberOfDisparities,0,human_anchor.width,human_anchor.height)).clone();
	//	//this->smooth_depth_map();
	//
	//	depth_map.copyTo(depth_map2);
}

inline void argus_depth::smooth_depth_map(){
	cv::Mat holes;
	threshold(depth_map,holes,1,255,cv::THRESH_BINARY_INV); //keep only the holes
	inpaint(depth_map, holes, depth_map, 2.0 , cv::INPAINT_NS );
}

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
	std::vector<human> human_group;	//vector of possible humans

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
		human human_candidate;
		human_candidate.propability = 0;
		human_candidate.bounding_rect =(clearview_mask) & found_filtered[i];
		human_candidate.human_center = cv::Point(human_candidate.bounding_rect.x + human_candidate.bounding_rect.width/2, human_candidate.bounding_rect.y + human_candidate.bounding_rect.height/2);
		for(j = 0; j<(int)faces.size();j++)
		{
			cv::Point head_candidate = cv::Point(faces[j].x + faces[j].width/2, faces[j].y + faces[j].height/2);
			if( human_candidate.bounding_rect.contains(head_candidate) )
			{
				if(human_candidate.head_rect == cv::Rect())		//head not yet assigned
				{
					human_candidate.head_rect = faces[j];
					human_candidate.head_center = head_candidate;
				}else if(fast_distance(head_candidate,human_candidate.human_center) > fast_distance(human_candidate.head_center ,human_candidate.human_center)){
					human_candidate.head_rect = faces[j];
					human_candidate.head_center = cv::Point(human_candidate.head_rect.x + human_candidate.head_rect.width/2, human_candidate.head_rect.y + human_candidate.head_rect.height/2);
				}
				faces.erase(faces.begin()+j);
			}
		}

		if(human_candidate.head_rect != cv::Rect()){
			human_candidate.bounding_rect = human_candidate.bounding_rect | human_candidate.head_rect;
			float dist = fast_distance(human_candidate.head_center ,human_candidate.human_center);
			int skin = cv::countNonZero(find_skin(rect_mat_left(human_candidate.head_rect)));
			human_candidate.propability = 0.5 * 2*dist/(human_candidate.bounding_rect.height-human_candidate.head_rect.height) + 0.5*(float)skin/human_candidate.head_rect.area();
		}
		human_group.push_back(human_candidate);
	}

#if DEBUG_MODE
	debug_detected_human(human_group);
#endif

	human user;
	if(human_group.size()>1){
		//find the biggest human
		float max = human_group.at(0).bounding_rect.area() * (1 + human_group.at(0).propability);
		int pos = 0;
		for( i = 1; i < (int)human_group.size(); i++ ){
			if (human_group.at(i).bounding_rect.area() * (1 + human_group.at(i).propability)>max){
				max = human_group.at(i).bounding_rect.area() * (1 + human_group.at(i).propability);
				pos = i;
			}
		}
		user = human_group.at(pos);
	}

	if(human_group.size()>0){
		rectangle(rect_mat_right, user.bounding_rect.tl(), user.bounding_rect.br(), cv::Scalar(255,0,255), 3);
		//if(final.area()>human_anchor.area())human_anchor=final;
		if(user.bounding_rect.area()>0.7*human_anchor.area())human_anchor=user.bounding_rect;

	}
	human_anchor=(clearview_mask) & human_anchor;

	//person_left = (*BW_rect_mat_left)(human_anchor).clone();
	//person_right = (*BW_rect_mat_right)(human_anchor).clone();

}

void argus_depth::take_snapshot(){
	std::cout<<"Snapshot taken!" << std::endl;
	imwrite("depth.png", depth_map2);
	//imwrite("person.png", person_left);
}

void argus_depth::clustering(){

	//Mat dataset((depth_map2->rows)*(depth_map2->cols),3,CV_8UC1);
	//Mat intensity_data=(*BW_rect_mat_left)(human_anchor);
	cv::Mat data=depth_map2.reshape(1,1);
	cv::Mat dataset(1,data.cols,CV_8UC1);
	data.row(0).copyTo(dataset.row(0));
	//equalizeHist(intensity_data,intensity_data);
	//Mat dataset2=dataset.reshape(1,depth_map2->rows);


	dataset.convertTo(dataset,CV_32F);
	dataset=dataset.t();

	cv::Mat labels,centers;

	cv::TermCriteria criteria;
	criteria.type=CV_TERMCRIT_ITER;
	criteria.maxCount=5;

	kmeans(dataset, 3, labels, criteria, 2, cv::KMEANS_PP_CENTERS);

	//Find out which label shows noise
	int noise_label, labelA, labelB, popA=0, popB=0;
	float mean_teamA=0, mean_teamB=0;
	for(int i=0;i<dataset.rows;i++){
		if(dataset.at<int>(i,0)==(float)0){
			noise_label=(int)labels.at<int>(i,0);
			labelA=(noise_label+1)%3;
			labelB=(noise_label+2)%3;
			break;
		};
	}

	//Calculate the sum of depth for the other teams
	for(int i=0;i<dataset.rows;i++){
		if(labels.at<int>(i,0)==(float)labelA){
			mean_teamA+=dataset.at<int>(i,0);
			popA++;
		}else if(labels.at<int>(i,0)==(float)labelB){
			mean_teamB+=dataset.at<int>(i,0);
			popB++;
		};
	}
	mean_teamA=popA*(mean_teamA/popA);
	mean_teamB=popB*(mean_teamB/popB);

	//Decide which team should be background and which foreground
	if(mean_teamA>mean_teamB){
		for(int i=0;i<dataset.rows;i++){
			if(labels.at<int>(i,0)==(float)labelA){
				labels.at<int>(i,0)=(float)2;
			}else if(labels.at<int>(i,0)==(float)labelB){
				labels.at<int>(i,0)=(float)1;
			}else{
				labels.at<int>(i,0)=(float)0;
			};
		}
	}else{
		for(int i=0;i<dataset.rows;i++){
			if(labels.at<int>(i,0)==(float)labelA){
				labels.at<int>(i,0)=(float)1;
			}else if(labels.at<int>(i,0)==(float)labelB){
				labels.at<int>(i,0)=(float)2;
			}else{
				labels.at<int>(i,0)=(float)0;
			};
		}
	}

	labels=labels.t();
	labels.convertTo(labels,CV_8UC1);
	labels=labels/2;
	labels=labels*255;
	labels=labels.reshape(1,depth_map2.rows);

	//bilateralFilter(labels, labels2, 9, 30, 30, BORDER_DEFAULT );
	//labels2.copyTo(labels);
	//medianBlur(labels, labels, 5);

	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::Mat biggest_blob;
	labels.copyTo(biggest_blob);

	findContours( biggest_blob, contours, hierarchy,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );

	//Find biggest contour
	std::vector<cv::Point> max_contour = contours[0];
	int max_contour_pos = 0;
	for(int i=0;i<(int)contours.size();i++){

		if(contours[i].size()>max_contour.size()){
			max_contour=contours[i];
			max_contour_pos=i;
		}
	}


	drawContours( biggest_blob, contours, max_contour_pos, cv::Scalar(255), CV_FILLED, 8 ); //Fill biggest blob
	threshold(biggest_blob,biggest_blob,254,255,cv::THRESH_BINARY); //This clears any contour leftovers caused by findContours
	bitwise_and(biggest_blob,labels,labels); 					//Keep the internal detail of the biggest blob
	//	imshow("mask",biggest_blob);

	bitwise_and(depth_map2,labels,depth_map2);	//Filter the depth map. Keep only foreground and biggest blob
	//medianBlur(*depth_map2, *depth_map2, 5);
	//	imshow("Foreground bigest blob",*depth_map2);

	cv::Mat holes1,holes2;
	threshold(depth_map2,holes1,1,255,cv::THRESH_BINARY_INV); //find the holes of the original depth map
	threshold(depth_map,holes2,1,255,cv::THRESH_BINARY_INV); //find the holes of the fragmented depth map
	//imshow("holes1",holes1);
	//imshow("holes2",holes2);
	bitwise_and(holes2,biggest_blob,holes2); //keep only the holes inside the blob
	//imshow("both holes",holes2);
	bitwise_and(holes1,holes2,holes1); //combine the 2 holes, so you know which holes are real, fake ones have depth

	cv::Mat cover_depth;
	cv::Mat kernel(9,9,CV_8U,cv::Scalar(1));
	medianBlur(depth_map,cover_depth, 5);
	morphologyEx(cover_depth, cover_depth, cv::MORPH_CLOSE , kernel);	//Create a depth map free of holes but not sharp
	//imshow("cover_depth",cover_depth);
	bitwise_and(holes1,cover_depth,holes1);	//keep the info for the holes

	imshow("before",depth_map2);
	add(holes1,depth_map2,depth_map2); //cover the holes
	imshow("after",depth_map2);

	//imshow("filtered",*depth_map2);

	cv::Mat jet_depth_map2(height,width,CV_8UC3);
	applyColorMap(depth_map2, jet_depth_map2, cv::COLORMAP_JET );
	//imshow( "test", jet_depth_map2 );

	//imshow("test1",depth_map2);
	cv::Mat point_cloud;
	reprojectImageTo3D(depth_map, point_cloud, Q, false, -1 );

	cv::Mat xyz= point_cloud.reshape(3,point_cloud.size().area());
	cv::Mat R_vec;
	Rodrigues(R1, R_vec);

	cv::Mat destimage, destdisp, image, disp;

	image = (BW_rect_mat_left)(human_anchor).clone();
	disp = depth_map.clone();

	//image.convertTo(image,CV_8UC1);
	//disp.convertTo(disp,CV_8UC1);

	double focal_length = 598.57;

	//(3) camera setting
	cv::Mat K = cv::Mat::eye(3,3,CV_64F);
	K.at<double>(0,0) = focal_length;
	K.at<double>(1,1) = focal_length;
	K.at<double>(0,2) = (image.cols-1.0)/2.0;
	K.at<double>(1,2) = (image.rows-1.0)/2.0;

	cv::Mat dist = cv::Mat::zeros(5,1,CV_64F);
	cv::Mat R = cv::Mat::eye(3,3,CV_64F);
	cv::Mat t = cv::Mat::zeros(3,1,CV_64F);



	lookat(viewpoint, lookatpoint , R);

	t.at<double>(0,0)=viewpoint.x;
	t.at<double>(1,0)=viewpoint.y;
	t.at<double>(2,0)=viewpoint.z;
	t=R*t;



	camera_view(image,destimage,disp,destdisp,xyz,R,t,K,dist);
	destdisp.convertTo(destdisp,CV_8U,0.5);
	imshow("out",destdisp);

	cv::Mat new_disp;
	cloud_to_disparity(new_disp, point_cloud);
}

void argus_depth::cloud_to_disparity(cv::Mat disparity, cv::Mat xyz){
	disparity = cv::Mat(xyz.rows,xyz.cols,CV_32FC1);
	float z,y;
	for(int i=0;i<xyz.rows;i++){
		for(int j=0;j<xyz.cols;j++){
			z=xyz.ptr<float>(i)[3*j+2];
			y=xyz.ptr<float>(i)[3*j+1];
			if(y>-95&&z>-350){
				disparity.at<float>(i,j)=Q.at<double>(2,3)/z*Q.at<double>(3,2);
			}else{
				disparity.at<float>(i,j)=0;
			}

		}
	}

	normalize(disparity, disparity, 0.0, 255.0, cv::NORM_MINMAX);
	disparity.convertTo(disparity,CV_8UC1);

	//cvtColor(xyz,xyz,CV_RGB2GRAY);

	imshow("hello",disparity);
}

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

int main(){
	int key_pressed=255;

	argus_depth *eye_stereo = new argus_depth;
	bool loop=false;
	while(1){
		do{
			key_pressed = cvWaitKey(1) & 255;
			if ( key_pressed == 32 )loop=!loop;
			if ( key_pressed == 27 ) break;
		}while (loop);
		if ( key_pressed == 27 ) break;
		if ( key_pressed == 13 ) eye_stereo->take_snapshot();

		if ( key_pressed == 119 ) eye_stereo->viewpoint.y+=10;	//W
		if ( key_pressed == 97 ) eye_stereo->viewpoint.x+=10;		//A
		if ( key_pressed == 115 ) eye_stereo->viewpoint.y-=10;		//S
		if ( key_pressed == 100 ) eye_stereo->viewpoint.x-=10;		//D

		double t = (double)cv::getTickCount();
		eye_stereo->refresh_frame();
		eye_stereo->detect_human();

		//eye_stereo->compute_depth();
		t = (double)cv::getTickCount() - t;
		eye_stereo->fps = 1/(t/cv::getTickFrequency());//for fps
		//eye_stereo->fps = t*1000./cv::getTickFrequency();//for ms
		eye_stereo->refresh_window();
		//eye_stereo->clustering();
		//key_pressed = cvWaitKey(1) & 255;

	}

	delete(eye_stereo);

	return 0;
}
