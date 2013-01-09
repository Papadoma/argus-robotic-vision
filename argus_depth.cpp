#include <stdio.h>
#include <opencv.hpp>
//#include "opencv2/ocl/ocl.hpp"
//#include <skeltrack.h>


#include "module_input.hpp"
//#include "module_file.hpp"

using namespace std;
using namespace cv;
//using namespace ocl;

class argus_depth{
private:
	module_eye input_module;
	//module_file input_module;

	Mat cameraMatrix[2], distCoeffs[2];
	Mat R, T, E, F, Q;
	Mat R1, R2, P1, P2;

	Mat mat_left;
	Mat mat_right;
	Mat rect_mat_left;
	Mat rect_mat_right;
	Mat BW_rect_mat_left;
	Mat BW_rect_mat_right;

	Mat depth_map_lowres;
	Mat depth_map_highres;

	Mat depth_map2;

	Mat person_left;
	Mat person_right;

	Mat thres_mask;
Mat test;


	//ocl::oclMat* left_ocl;
	//ocl::oclMat* right_ocl;
	//ocl::oclMat* depth_ocl;

	Rect human_anchor;
	Rect roi1, roi2;
	Mat rmap[2][2];

	StereoBM bm;
	StereoSGBM sgbm;
	StereoVar var;

	//ocl::HOGDescriptor hog;
	HOGDescriptor hog;


	Rect* clearview_mask;

	int numberOfDisparities;
	int width;
	int height;
	int frame_counter;

	void load_param();
	void smooth_depth_map();

	void lookat(Point3d from, Point3d to, Mat& destR);
	void eular2rot(double yaw,double pitch, double roll,Mat& dest);
	void cloud_to_disparity(Mat, Mat);
public:
	double baseline;
	Point3d viewpoint;
	Point3d lookatpoint;

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
	void camera_view(Mat& , Mat& , Mat& , Mat& , Mat& , Mat&, Mat&, Mat&, Mat&);
};

//Constructor
argus_depth::argus_depth()
{

	frame_counter=0;
	//hog.setSVMDetector(ocl::HOGDescriptor::getDefaultPeopleDetector());//getPeopleDetector64x128 or getDefaultPeopleDetector
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());//getPeopleDetector64x128 or getDefaultPeopleDetector

	Size framesize = input_module.getSize();
	height=framesize.height;
	width=framesize.width;
	human_anchor=Rect(width/2,height/2,10,10);

	baseline = 9.5;
	viewpoint=Point3d(0.0,0.0,baseline*10);
	lookatpoint=Point3d(0.0,0.0,-baseline*10.0);

	//left_ocl=new ocl::oclMat(height,width,CV_8UC1);
	//right_ocl=new ocl::oclMat(height,width,CV_8UC1);
	//depth_ocl=new ocl::oclMat(height,width,CV_8UC1);

	//	printf("Begin creating ocl context...\n");
	//	//std::vector<ocl::Info> oclinfo;
	//	//int devnums = ocl::getDevice(oclinfo);
	//	vector<Info> ocl_info;
	//	int devnums=getDevice(ocl_info);
	//	printf("End creating ocl context...\n");
	//
	//	if(devnums<1){
	//		std::cout << "no OPENCL device found\n";
	//	}

	this->load_param();

	numberOfDisparities=32;

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

	clearview_mask=new Rect(numberOfDisparities,0,width,height);
	*clearview_mask = roi1 & roi2 & (*clearview_mask);

	//depth_map2=new Mat(clearview_mask->height,clearview_mask->width,CV_8UC1);



}

//Destructor
argus_depth::~argus_depth(){
	destroyAllWindows();
	//	input_module.~module_eye();
}

void argus_depth::refresh_frame(){
	input_module.getFrame(mat_left,mat_right);

	remap(mat_left, rect_mat_left, rmap[0][0], rmap[0][1], CV_INTER_LINEAR);
	remap(mat_right, rect_mat_right, rmap[1][0], rmap[1][1], CV_INTER_LINEAR);

	cvtColor(rect_mat_left,BW_rect_mat_left,CV_RGB2GRAY);
	cvtColor(rect_mat_right,BW_rect_mat_right,CV_RGB2GRAY);

	frame_counter++;
	//	equalizeHist(*BW_rect_mat_left,*BW_rect_mat_left);
	//	equalizeHist(*BW_rect_mat_right,*BW_rect_mat_right);

	//left_ocl->upload(*BW_rect_mat_left);
	//right_ocl->upload(*BW_rect_mat_right);


	//oclMat left_ocl(height,width,CV_8UC3);

	//	left_ocl.upload(*rect_mat_left);

	//oclMat* disp_ocl;
}

void argus_depth::refresh_window(){
	//		imshow( "original_camera_left", *mat_left );
	//	imshow( "original_camera_right", *mat_right );

	rectangle(rect_mat_left, *clearview_mask, Scalar(255,255,255), 1, 8);
	rectangle(rect_mat_right, *clearview_mask, Scalar(255,255,255), 1, 8);

	Mat imgResult(height,2*width,CV_8UC3); // Your final imageasd
	Mat roiImgResult_Left = imgResult(Rect(0,0,rect_mat_left.cols,rect_mat_left.rows));
	Mat roiImgResult_Right = imgResult(Rect(rect_mat_right.cols,0,rect_mat_right.cols,rect_mat_right.rows));
	Mat roiImg1 = (rect_mat_left)(Rect(0,0,rect_mat_left.cols,rect_mat_left.rows));
	Mat roiImg2 = (rect_mat_right)(Rect(0,0,rect_mat_right.cols,rect_mat_right.rows));
	roiImg1.copyTo(roiImgResult_Left);
	roiImg2.copyTo(roiImgResult_Right);

	stringstream ss;//create a stringstream
	ss << fps;//add number to the stream
	putText(imgResult, ss.str(), Point (10,20), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,0,255), 2, 8, false );

	imshow( "Camera", imgResult );
	//imshow( "depth", *depth_map_lowres );

	Mat jet_depth_map2(height,width,CV_8UC3);
	applyColorMap(depth_map2, jet_depth_map2, COLORMAP_JET );
	imshow( "depth2", jet_depth_map2 );

	//imshow("person",person_left);
	//imshow( "depth2", *depth_map2 );
}


void argus_depth::load_param(){

	bool flag1=false;
	bool flag2=false;

	string intrinsics="intrinsics_eye.yml";
	string extrinsics="extrinsics_eye.yml";

	Size imageSize(width,height);

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
		Mat Q_local=Q.clone();
		stereoRectify( cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1], imageSize, R, T, R1, R2, P1, P2, Q_local, CALIB_ZERO_DISPARITY, -1, imageSize, &roi1, &roi2 );

		//getOptimalNewCameraMatrix(cameraMatrix[0], distCoeffs[0], imageSize, 1, imageSize, &roi1);
		//getOptimalNewCameraMatrix(cameraMatrix[1], distCoeffs[1], imageSize, 1, imageSize, &roi2);
		initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
		initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

	}

}

void argus_depth::compute_depth(){

	Rect refined_human_anchor=human_anchor;
	refined_human_anchor.x=human_anchor.x-numberOfDisparities;
	refined_human_anchor.width=human_anchor.width+numberOfDisparities;

	person_left=(BW_rect_mat_left)(refined_human_anchor);
	person_right=(BW_rect_mat_right)(refined_human_anchor);
imshow("debugleft",person_left);
imshow("debugright",person_right);
	//sgbm.SADWindowSize = 15;
	//sgbm(*person_left,*person_right,*depth_map_lowres);

	sgbm.SADWindowSize = 3;
	sgbm(person_left,person_right,depth_map_highres);

	sgbm.SADWindowSize = 11;
	//sgbm(*person_left,*person_right,*depth_map_lowres);
	depth_map_lowres=depth_map_highres.clone();

	depth_map_lowres.convertTo(depth_map_lowres, CV_8UC1, 255/(numberOfDisparities*16.));
	depth_map_highres.convertTo(depth_map_highres, CV_8UC1, 255/(numberOfDisparities*16.));

	//bilateralFilter(*depth_map_highres, *depth_map_lowres, 9, 30, 30, BORDER_DEFAULT );
	//blur(*depth_map_highres, *depth_map_lowres, Size(5,5));
	//medianBlur(*depth_map_highres,*depth_map_lowres, 11);

	//medianBlur(*depth_map_lowres,*depth_map_lowres, 9);
	//GaussianBlur(*depth_map_lowres, *depth_map_lowres, Size(5,5), 50);

	//Smooth out the depth map


	depth_map_highres=(depth_map_highres)(Rect(numberOfDisparities,0,human_anchor.width,human_anchor.height)).clone();
	depth_map_lowres=(depth_map_lowres)(Rect(numberOfDisparities,0,human_anchor.width,human_anchor.height)).clone();

	//this->smooth_depth_map();

	depth_map_highres.copyTo(depth_map2);

	//bilateralFilter(*depth_map_lowres, *depth_map2, 5, 30, 30, BORDER_DEFAULT );

}

void argus_depth::smooth_depth_map(){
	Mat holes;
	threshold(depth_map_highres,holes,1,255,THRESH_BINARY_INV); //keep only the holes

	bitwise_and(holes,depth_map_lowres,holes);
	bitwise_or(holes,depth_map_highres,depth_map2);

	//	Mat kernel(5,5,CV_8U,cv::Scalar(1));
	//	morphologyEx(*depth_map_highres, *depth_map_lowres, MORPH_CLOSE , kernel);

}

void argus_depth::detect_human(){

	//		hog.setSVMDetector(ocl::HOGDescriptor::getDefaultPeopleDetector());
	//		ocl::oclMat img_ocl;
	//		img_ocl.upload(*BW_rect_mat_left);

	vector<Rect> found, found_filtered;

	hog.detectMultiScale(BW_rect_mat_left, found, 0, Size(8,8), Size(0,0), 1.05, 0);   //Window stride. It must be a multiple of block stride.
	//hog.detectMultiScale(img_ocl, found, 0, Size(8,8), Size(0,0), 1.05, 0);

	size_t i, j;

	for( i = 0; i < found.size(); i++ )
	{
		Rect r = found[i];
		for( j = 0; j < found.size(); j++ )
			if( j != i && (r & found[j]) == r)
				break;
		if( j == found.size() )
			found_filtered.push_back(r);
	}

	int found_area[found_filtered.size()];
	for( i = 0; i < found_filtered.size(); i++ )
	{
		//found_filtered[i].x += cvRound(r.width*0.1);
		//found_filtered[i].width = cvRound(r.width*0.8);
		found_filtered[i].y += cvRound(found_filtered[i].height*0.05);
		//found_filtered[i].height = cvRound(found_filtered[i].height*1.05);
		found_area[i] = found_filtered[i].area();
	}


	vector<Rect> found_person;
	if(found_filtered.size()>1){
		int max = found_area[0];
		int pos = 0;
		for( i = 1; i < found_filtered.size(); i++ ){
			if (found_area[i]>max){
				max = found_area[i];
				pos = i;
			}
		}


		Point p1(found_filtered[pos].x+found_filtered[pos].width/2,found_filtered[pos].y+found_filtered[pos].height/2);
		for( i = 0; i < found_filtered.size(); i++ ){
			Point p2(found_filtered[i].x+found_filtered[i].width/2,found_filtered[i].y+found_filtered[i].height/2);
			int distance = sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y));
			if(distance<found_filtered[pos].width/2 && distance<found_filtered[pos].height/2){
				found_person.push_back(found_filtered[i]);
			}
		}

	}else if(found_filtered.size()==1){
		found_person.push_back(found_filtered[0]);
	}

	for( i = 0; i < found_filtered.size(); i++ )
	{
		// the HOG detector returns slightly larger rectangles than the real objects.
		// so we slightly shrink the rectangles to get a nicer output.
		Rect r = found_filtered[i];

		rectangle(rect_mat_left, r.tl(), r.br(), cv::Scalar(0,255,0), 1);
		rectangle(rect_mat_right, r.tl(), r.br(), cv::Scalar(0,255,0), 1);
	}
	//	if(!found_filtered.empty()){
	//		stringstream ss2;//create a stringstream
	//		ss2 << found_area[pos];//add number to the stream
	//		putText(*rect_mat_left, ss2.str(), found_filtered[pos].br(), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,0,255), 1, 8, false );
	//	}
	//	for( i = 0; i < found_person.size(); i++ ){
	//		rectangle(*rect_mat_left, found_person[i].tl(), found_person[i].br(), cv::Scalar(255,0,0), 1);
	//	}
	vector<Point> points_person;
	for( i = 0; i < found_person.size(); i++ ){
		rectangle(rect_mat_left, found_person[i].tl(), found_person[i].br(), cv::Scalar(0,255,255), 2);
		Point p1=found_person[i].tl();
		Point p2(found_person[i].x+found_person[i].width,found_person[i].y);
		Point p3=found_person[i].br();
		Point p4(found_person[i].x,found_person[i].y+found_person[i].height);
		points_person.push_back(p1);
		points_person.push_back(p2);
		points_person.push_back(p3);
		points_person.push_back(p4);
	}
	if(points_person.size()>1){
		Rect final = boundingRect(points_person);
		rectangle(rect_mat_left, final.tl(), final.br(), cv::Scalar(0,0,255), 3);
		//if(final.area()>human_anchor.area())human_anchor=final;
		if(final.area()>0.7*human_anchor.area())human_anchor=final;

	}
	human_anchor=(*clearview_mask) & human_anchor;
	rectangle(rect_mat_left, human_anchor.tl(), human_anchor.br(), cv::Scalar(255,0,255), 1);

	//	person_left = (*BW_rect_mat_left)(human_anchor).clone();
	//	person_right = (*BW_rect_mat_right)(human_anchor).clone();

}

void argus_depth::take_snapshot(){
	cout<<"Snapshot taken!\n";
	imwrite("depth.png", depth_map2);
	imwrite("person.png", person_left);
}

void argus_depth::clustering(){

	//Mat dataset((depth_map2->rows)*(depth_map2->cols),3,CV_8UC1);
	//Mat intensity_data=(*BW_rect_mat_left)(human_anchor);
	Mat data=depth_map2.reshape(1,1);
	Mat dataset(1,data.cols,CV_8UC1);
	data.row(0).copyTo(dataset.row(0));
	//equalizeHist(intensity_data,intensity_data);
	//Mat dataset2=dataset.reshape(1,depth_map2->rows);


	dataset.convertTo(dataset,CV_32F);
	dataset=dataset.t();

	Mat labels,centers;

	TermCriteria criteria;
	criteria.type=CV_TERMCRIT_ITER;
	criteria.maxCount=5;

	kmeans(dataset, 3, labels, criteria, 2, KMEANS_PP_CENTERS    );

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

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	Mat biggest_blob;
	labels.copyTo(biggest_blob);

	findContours( biggest_blob, contours, hierarchy,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );

	//Find biggest contour
	vector<Point> max_contour = contours[0];
	int max_contour_pos = 0;
	for(int i=0;i<(int)contours.size();i++){

		if(contours[i].size()>max_contour.size()){
			max_contour=contours[i];
			max_contour_pos=i;
		}
	}


	drawContours( biggest_blob, contours, max_contour_pos, Scalar(255), CV_FILLED, 8 ); //Fill biggest blob
	threshold(biggest_blob,biggest_blob,254,255,THRESH_BINARY); //This clears any contour leftovers caused by findContours
	bitwise_and(biggest_blob,labels,labels); 					//Keep the internal detail of the biggest blob
	//	imshow("mask",biggest_blob);

	bitwise_and(depth_map2,labels,depth_map2);	//Filter the depth map. Keep only foreground and biggest blob
	//medianBlur(*depth_map2, *depth_map2, 5);
	//	imshow("Foreground bigest blob",*depth_map2);

	Mat holes1,holes2;
	threshold(depth_map2,holes1,1,255,THRESH_BINARY_INV); //find the holes of the original depth map
	threshold(depth_map_highres,holes2,1,255,THRESH_BINARY_INV); //find the holes of the fragmented depth map
	//imshow("holes1",holes1);
	//imshow("holes2",holes2);
	bitwise_and(holes2,biggest_blob,holes2); //keep only the holes inside the blob
	//imshow("both holes",holes2);
	bitwise_and(holes1,holes2,holes1); //combine the 2 holes, so you know which holes are real, fake ones have depth

	Mat cover_depth;
	Mat kernel(9,9,CV_8U,cv::Scalar(1));
	medianBlur(depth_map_highres,cover_depth, 5);
	morphologyEx(cover_depth, cover_depth, MORPH_CLOSE , kernel);	//Create a depth map free of holes but not sharp
	//imshow("cover_depth",cover_depth);
	bitwise_and(holes1,cover_depth,holes1);	//keep the info for the holes

	//imshow("before",*depth_map2);
	add(holes1,depth_map2,depth_map2); //cover the holes
	//imshow("after",*depth_map2);

	//imshow("filtered",*depth_map2);

	Mat jet_depth_map2(height,width,CV_8UC3);
	applyColorMap(depth_map2, jet_depth_map2, COLORMAP_JET );
	//imshow( "test", jet_depth_map2 );

	//imshow("test1",depth_map2);
	Mat point_cloud;
	reprojectImageTo3D(depth_map_highres, point_cloud, Q, false, -1 );

	Mat xyz= point_cloud.reshape(3,point_cloud.size().area());
	Mat R_vec;
	Rodrigues(R1, R_vec);

	Mat destimage, destdisp, image, disp;

	image = (BW_rect_mat_left)(human_anchor).clone();
	disp=depth_map_highres.clone();

	//image.convertTo(image,CV_8UC1);
	//disp.convertTo(disp,CV_8UC1);

	double focal_length = 598.57;

	//(3) camera setting
	Mat K=Mat::eye(3,3,CV_64F);
	K.at<double>(0,0)=focal_length;
	K.at<double>(1,1)=focal_length;
	K.at<double>(0,2)=(image.cols-1.0)/2.0;
	K.at<double>(1,2)=(image.rows-1.0)/2.0;

	Mat dist=Mat::zeros(5,1,CV_64F);
	Mat R=Mat::eye(3,3,CV_64F);
	Mat t=Mat::zeros(3,1,CV_64F);



	lookat(viewpoint, lookatpoint , R);

	t.at<double>(0,0)=viewpoint.x;
	t.at<double>(1,0)=viewpoint.y;
	t.at<double>(2,0)=viewpoint.z;
	t=R*t;



	camera_view(image,destimage,disp,destdisp,xyz,R,t,K,dist);
	destdisp.convertTo(destdisp,CV_8U,0.5);
	imshow("out",destdisp);

	Mat new_disp;
	cloud_to_disparity(new_disp, point_cloud);
}
void argus_depth::cloud_to_disparity(Mat disparity, Mat xyz){
	disparity=Mat(xyz.rows,xyz.cols,CV_32FC1);
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

	//cout<<z<<" ";
	normalize(disparity, disparity, 0.0, 255.0, NORM_MINMAX);
	disparity.convertTo(disparity,CV_8UC1);

	//cvtColor(xyz,xyz,CV_RGB2GRAY);

	imshow("hello",disparity);
}

void argus_depth::lookat(Point3d from, Point3d to, Mat& destR)
{
	double x=(to.x-from.x);
	double y=(to.y-from.y);
	double z=(to.z-from.z);

	double pitch =asin(x/sqrt(x*x+z*z))/CV_PI*180.0;
	double yaw   =asin(-y/sqrt(y*y+z*z))/CV_PI*180.0;

	eular2rot(yaw, pitch, 0,destR);
}

void argus_depth::eular2rot(double yaw,double pitch, double roll,Mat& dest)
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
	Mat Rx(3,3,CV_64F,datax);
	Mat Ry(3,3,CV_64F,datay);
	Mat Rz(3,3,CV_64F,dataz);
	Mat rr=Rz*Rx*Ry;
	rr.copyTo(dest);
}

void argus_depth::camera_view(Mat& image, Mat& destimage, Mat& disp, Mat& destdisp, Mat& xyz, Mat& R, Mat& t, Mat& K, Mat& dist){

	if(destimage.empty())destimage=Mat::zeros(Size(image.size()),image.type());
	if(destdisp.empty())destdisp=Mat::zeros(Size(image.size()),disp.type());
	vector<Point2f> pt;
	if(dist.empty()) dist = Mat::zeros(Size(5,1),CV_32F);
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

	//	vector<ocl::Info> info;
	//	CV_Assert(ocl::getDevice(info));
	//	int devnums =ocl::getDevice(info);


	//	if(devnums<1){
	//		std::cout << "no OPENCL device found\n";
	//	}

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

		double t = (double)getTickCount();
		eye_stereo->refresh_frame();
		eye_stereo->detect_human();

		eye_stereo->compute_depth();
		t = (double)getTickCount() - t;
		eye_stereo->fps = 1/(t/cv::getTickFrequency());//for fps
		//eye_stereo->fps = t*1000./cv::getTickFrequency();//for ms
		eye_stereo->refresh_window();
		//eye_stereo->clustering();
		//key_pressed = cvWaitKey(1) & 255;

	}

	delete(eye_stereo);
	return 0;
}
