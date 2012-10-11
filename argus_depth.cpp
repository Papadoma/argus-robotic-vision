#include <stdio.h>
#include <opencv.hpp>
#include "opencv2/ocl/ocl.hpp"
#include <skeltrack.h>


#include "module_eye.hpp"
#include "module_file.hpp"

using namespace std;
using namespace cv;
//using namespace ocl;

class argus_depth{
private:
	//module_eye input_module;
	module_file input_module;

	Mat* mat_left;
	Mat* mat_right;
	Mat* rect_mat_left;
	Mat* rect_mat_right;
	Mat* BW_rect_mat_left;
	Mat* BW_rect_mat_right;

	Mat* depth_map;
	Mat* previous_depth_map;
	Mat* depth_map2;

	Mat* person_left;
	Mat* person_right;

	Mat* thres_mask;

	Mat* prev_rect_mat_left;

	ocl::oclMat* left_ocl;
	ocl::oclMat* right_ocl;
	ocl::oclMat* depth_ocl;

	Rect human_anchor;
	Rect roi1, roi2;
	Mat rmap[2][2];

	StereoBM bm;
	StereoSGBM sgbm;
	StereoVar var;

	//ocl::HOGDescriptor hog;
	HOGDescriptor hog;

	BackgroundSubtractorMOG2* BSMOG;

	Rect* clearview_mask;

	int numberOfDisparities;
	int width;
	int height;
	int frame_counter;

	void load_param();
	void smooth_depth_map();


public:
	argus_depth();
	~argus_depth();
	void clustering();
	Mat imHist(Mat, float, float);
	int fps;

	void remove_background();
	void refresh_frame();
	void refresh_window();
	void compute_depth();
	void detect_human();
	void compute_depth_gpu();
	void info();
};

//Constructor
argus_depth::argus_depth(){
	frame_counter=0;
	//hog.setSVMDetector(ocl::HOGDescriptor::getDefaultPeopleDetector());//getPeopleDetector64x128 or getDefaultPeopleDetector
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());//getPeopleDetector64x128 or getDefaultPeopleDetector

	Size framesize = input_module.getSize();
	height=framesize.height;
	width=framesize.width;

	mat_left=new Mat(height,width,CV_8UC3);
	mat_right=new Mat(height,width,CV_8UC3);
	BW_rect_mat_left=new Mat(height,width,CV_8UC1);
	BW_rect_mat_right=new Mat(height,width,CV_8UC1);
	person_left=new Mat(height,width,CV_8UC1);
	person_right=new Mat(height,width,CV_8UC1);
	rect_mat_left=new Mat(height,width,CV_8UC3);
	rect_mat_right=new Mat(height,width,CV_8UC3);

	depth_map=new Mat(height,width,CV_8UC1);

	thres_mask=new Mat(height,width,CV_8UC1);

	prev_rect_mat_left=new Mat(height,width,CV_8UC1);

	left_ocl=new ocl::oclMat(height,width,CV_8UC1);
	right_ocl=new ocl::oclMat(height,width,CV_8UC1);
	depth_ocl=new ocl::oclMat(height,width,CV_8UC1);

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

	previous_depth_map=new Mat(clearview_mask->height,clearview_mask->width,CV_8UC1);
	*previous_depth_map = Mat::zeros(clearview_mask->height, clearview_mask->width, CV_8UC1);

	depth_map2=new Mat(clearview_mask->height,clearview_mask->width,CV_8UC1);

	BSMOG=new BackgroundSubtractorMOG2(500,100,true);

}

//Destructor
argus_depth::~argus_depth(){

	delete(mat_left);
	delete(mat_right);
	delete(rect_mat_left);
	delete(rect_mat_right);
	delete(BW_rect_mat_left);
	delete(BW_rect_mat_right);
	delete(depth_map);
	delete(previous_depth_map);
	delete(depth_map2);

	delete(thres_mask);
	delete(BSMOG);

	delete(prev_rect_mat_left);

	destroyAllWindows();
	//	input_module.~module_eye();
}

void argus_depth::refresh_frame(){
	input_module.getFrame(mat_left,mat_right);

	remap(*mat_left, *rect_mat_left, rmap[0][0], rmap[0][1], CV_INTER_LINEAR);
	remap(*mat_right, *rect_mat_right, rmap[1][0], rmap[1][1], CV_INTER_LINEAR);

	cvtColor(*rect_mat_left,*BW_rect_mat_left,CV_RGB2GRAY);
	cvtColor(*rect_mat_right,*BW_rect_mat_right,CV_RGB2GRAY);

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

	rectangle(*rect_mat_left, *clearview_mask, Scalar(255,255,255), 1, 8);
	rectangle(*rect_mat_right, *clearview_mask, Scalar(255,255,255), 1, 8);

	Mat imgResult(height,2*width,CV_8UC3); // Your final imageasd
	Mat roiImgResult_Left = imgResult(Rect(0,0,rect_mat_left->cols,rect_mat_left->rows));
	Mat roiImgResult_Right = imgResult(Rect(rect_mat_right->cols,0,rect_mat_right->cols,rect_mat_right->rows));
	Mat roiImg1 = (*rect_mat_left)(Rect(0,0,rect_mat_left->cols,rect_mat_left->rows));
	Mat roiImg2 = (*rect_mat_right)(Rect(0,0,rect_mat_right->cols,rect_mat_right->rows));
	roiImg1.copyTo(roiImgResult_Left);
	roiImg2.copyTo(roiImgResult_Right);

	imshow( "Camera", imgResult );
	//imshow( "depth", *depth_map );

	Mat jet_depth_map2(height,width,CV_8UC3);
	applyColorMap(*depth_map2, jet_depth_map2, COLORMAP_JET );
	imshow( "depth2", jet_depth_map2 );

	//imshow("person",person_left);
	//imshow( "depth2", *depth_map2 );
}


void argus_depth::load_param(){

	bool flag1=false;
	bool flag2=false;

	string intrinsics="intrinsics_eye.yml";
	string extrinsics="extrinsics_eye.yml";

	Mat cameraMatrix[2], distCoeffs[2];
	Mat R, T, E, F;
	Mat R1, R2, P1, P2, Q;

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

		stereoRectify( cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1], imageSize, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, imageSize, &roi1, &roi2 );

		//getOptimalNewCameraMatrix(cameraMatrix[0], distCoeffs[0], imageSize, 1, imageSize, &roi1);
		//getOptimalNewCameraMatrix(cameraMatrix[1], distCoeffs[1], imageSize, 1, imageSize, &roi2);
		initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
		initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

	}

}

void argus_depth::info(){

	//cout<<capture_left->get(CV_CAP_PROP_POS_FRAMES)<<"\n";
}

Mat argus_depth::imHist(Mat hist, float scaleX=1, float scaleY=1){
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

void argus_depth::compute_depth(){

	Rect refined_human_anchor=human_anchor;
	refined_human_anchor.x=human_anchor.x-numberOfDisparities;
	refined_human_anchor.width=human_anchor.width+numberOfDisparities;

	*person_left=(*BW_rect_mat_left)(refined_human_anchor);
	*person_right=(*BW_rect_mat_right)(refined_human_anchor);
	sgbm(*person_left,*person_right,*depth_map);
	//bm(*BW_rect_mat_left,*BW_rect_mat_right,*depth_map);
	//var(*BW_rect_mat_left,*BW_rect_mat_right,*depth_map);
	depth_map->convertTo(*depth_map, CV_8UC1, 255/(numberOfDisparities*16.));

	*depth_map=(*depth_map)(Rect(numberOfDisparities,0,human_anchor.width,human_anchor.height));
	depth_map->copyTo(*depth_map2);
	//bilateralFilter(*depth_map, *depth_map2, 5, 30, 30, BORDER_DEFAULT );


	//Mat tmp2(*depth_map, roi1 & roi2 & mask);
	//tmp2.copyTo(*thres_mask);

	//Smooth out the depth map
	//this->smooth_depth_map();

	//depth_map2->convertTo(*depth_map2, CV_8U);

	//	int hbins = 64;
	//	int histSize[] = {hbins};
	//	// saturation varies from 0 (black-gray-white) to
	//	// 255 (pure spectrum color)
	//	float sranges[] = { 0, 255 };
	//	const float* ranges[] = { sranges };
	//	MatND depth_hist;
	//	int channels[] = {0};
	//	calcHist( depth_map2, 1, channels, Mat(), depth_hist, 1, histSize, ranges,	true, false );
	//	//depth_map->convertTo(*depth_map, CV_8U);
	//	depth_hist = imHist(depth_hist,4,4);
	//	imshow( "depth_histogram", depth_hist );
}

void argus_depth::compute_depth_gpu(){

	//	ocl::bilateralFilter(*left_ocl,*left_ocl,3,30,30,BORDER_REPLICATE    );
	//	ocl::bilateralFilter(*right_ocl,*right_ocl,3,30,30,BORDER_REPLICATE    );
	//	gpu_bm(*left_ocl, *right_ocl, *depth_ocl);
	//	//ocl::equalizeHist(*depth_ocl,*depth_ocl);
	//	depth_ocl->download(*depth_map);
	//	depth_map->convertTo(*depth_map, CV_8UC1, 255/(numberOfDisparities));
	//	//(*depth_map,*depth_map);
	//
	//	Mat tmp(*depth_map, *clearview_mask);
	//	tmp.copyTo(*depth_map2);
}

void argus_depth::smooth_depth_map(){
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

	//depth_map2->copyTo(result);
	Mat result2(result.size(),CV_8UC1);
	bilateralFilter(result, result2, 9, 30, 30, BORDER_DEFAULT );

	Mat jet_result(result.size(),CV_8UC3);
	applyColorMap(result, jet_result, COLORMAP_JET );
	imshow( "smoothed", jet_result);

	applyColorMap(result2, jet_result, COLORMAP_JET );
	imshow( "extrasmoothed", jet_result);


	result.copyTo(*previous_depth_map);

	//TODO Remove result matrix, pass depth_map2 directly

	//*depth_map= abs((*depth_map) - (*previous_depth_map));
	//*previous_depth_map=*depth_map;
}


void argus_depth::remove_background(){
	(*BSMOG)(*BW_rect_mat_left,*thres_mask,0.00005);

	threshold(*thres_mask, *thres_mask, 128, 255, THRESH_BINARY); //shadows are 127
	imshow( "MOG based mask", *thres_mask );

	Mat temp(height,width,CV_8UC1);
	BW_rect_mat_left->copyTo(temp);
	temp=abs(temp-(*prev_rect_mat_left));
	//bitwise_xor(*prev_rect_mat_left, temp,temp);
	threshold(temp, temp, 20, 255, THRESH_BINARY);
	//medianBlur(temp,temp, 5);
	//	imshow("Frame difference based mask", temp);
	BW_rect_mat_left->copyTo(*prev_rect_mat_left);

	bitwise_or(*thres_mask, temp, *thres_mask);
	//imshow( "Fused", *thres_mask );

	Mat skin_image(height,width,CV_8UC1);
	Mat rect_mat_left_YCrCb(height,width,CV_8UC3);
	cvtColor(*rect_mat_left, rect_mat_left_YCrCb, CV_BGR2YCrCb);
	Scalar yccMin(0, 131, 80);
	Scalar yccMax(255, 185, 135);
	inRange(rect_mat_left_YCrCb, yccMin, yccMax, skin_image);
	imshow("Skin",skin_image);

	medianBlur(*thres_mask, *thres_mask, 5);
	imshow( "Fused and smoothed", *thres_mask );

	Mat img_8uc3(height,width,CV_8UC3);
	thres_mask->copyTo(img_8uc3);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	dilate(*thres_mask,*thres_mask,Mat(),Point(),25);
	erode(*thres_mask,*thres_mask,Mat(),Point(),12);

	//	Mat tmp(*thres_mask, *clearview_mask);
	//	tmp.copyTo(*thres_mask);

	findContours( *thres_mask, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );

	cvtColor( *thres_mask, img_8uc3, CV_GRAY2BGR );
	drawContours( img_8uc3, contours, -1, CV_RGB(250,0,0), 1, 8, hierarchy,1 );
	//	for(int idx = 0 ; idx >= 0; idx = hierarchy[idx][0] )
	//	{
	//		//Scalar color( rand()&255, rand()&255, rand()&255 );
	//		//drawContours( img_8uc3, contours, idx, CV_RGB(250,0,0), CV_FILLED, 8, hierarchy );
	//	}

	//	int n=0;
	//printf( "Total Contours Detected: %d\n", Nc );
	//	CvScalar red = CV_RGB(250,0,0);
	//	CvScalar blue = CV_RGB(0,0,250);

	//	for( CvSeq* c=first_contour; c!=NULL; c=c->h_next ){
	//		cvCvtColor( thres_mask, &img_8uc3, CV_GRAY2BGR );
	//		cvDrawContours(&img_8uc3,c,red,blue,1,2,8);
	////		printf( "Contour #%dn", n );
	////		printf( " %d elements:\n", c->total );
	////		for( int i=0; itotal; ++i ){
	////			CvPoint* p = CV_GET_SEQ_ELEM( CvPoint, c, i );
	////			printf(" (%d,%d)\n", p->x, p->y );
	////		}
	////		cvWaitKey();
	////		n++;
	//	}
	//bitwise_or(img_8uc3, *thres_mask, img_8uc3);
	imshow( "Contours 2", img_8uc3 );

	dilate(*thres_mask,*thres_mask,Mat(),Point(),30);
	erode(*thres_mask,*thres_mask,Mat(),Point(),12);
	//imshow( "Fused, smoothed and inflated", *thres_mask );

	//	//	Mat tmp2(*rect_mat_left, *clearview_mask);
	//	Mat tmp2(*rect_mat_left);
	//	//tmp2.copyTo(*rect_mat_left);
	//	bitwise_and(tmp2, *thres_mask, *thres_mask);
	//	imshow( "output_thres", *thres_mask );



	//	double minVal;
	//	double maxVal;
	//	//*depth_map2 = (*depth_map2)|(*thres_mask);
	//	depth_map2->copyTo(tmp);
	//	bitwise_and(*depth_map2, *thres_mask, tmp);
	//	//tmp.copyTo(*depth_map2);
	//	minMaxLoc(tmp,0,&maxVal,0,0,Mat());
	//
	//	tmp=~tmp;
	//	threshold(tmp, tmp, 254, 255, THRESH_TOZERO_INV);
	//	minMaxLoc(tmp,0,&minVal,0,0,Mat());
	//	minVal=255-minVal;
	//
	//	//imshow( "thres_mask", tmp );
	//	//threshold(tmp, tmp, 0, 255, THRESH_TOZERO_INV);
	//	threshold(*depth_map2, *depth_map2, ((int)maxVal)+1, 255, THRESH_TOZERO_INV);
	//	threshold(*depth_map2, *depth_map2, ((int)minVal), 255, THRESH_TOZERO);
}

void argus_depth::detect_human(){

	//	hog.setSVMDetector(ocl::HOGDescriptor::getDefaultPeopleDetector());
	//	ocl::oclMat img_ocl;
	//	img_ocl.upload(*BW_rect_mat_left);


	vector<Rect> found, found_filtered;

	double t = (double)getTickCount();
	hog.detectMultiScale(*BW_rect_mat_left, found, 0, Size(8,8), Size(0,0), 1.05, 0);   //Window stride. It must be a multiple of block stride.
	t = (double)getTickCount() - t;

	stringstream ss;//create a stringstream
	ss << t*1000./cv::getTickFrequency();//add number to the stream
	putText(*rect_mat_left, ss.str(), Point (10,20), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,0,255), 2, 8, false );

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

		rectangle(*rect_mat_left, r.tl(), r.br(), cv::Scalar(0,255,0), 1);
		rectangle(*rect_mat_right, r.tl(), r.br(), cv::Scalar(0,255,0), 1);
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
		rectangle(*rect_mat_left, found_person[i].tl(), found_person[i].br(), cv::Scalar(0,255,255), 2);
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
		rectangle(*rect_mat_left, final.tl(), final.br(), cv::Scalar(0,0,255), 3);
		//if(final.area()>human_anchor.area())human_anchor=final;
		if(final.area()>0.7*human_anchor.area())human_anchor=final;

	}
	human_anchor=(*clearview_mask) & human_anchor;
	rectangle(*rect_mat_left, human_anchor.tl(), human_anchor.br(), cv::Scalar(255,0,255), 1);

	//	person_left = (*BW_rect_mat_left)(human_anchor).clone();
	//	person_right = (*BW_rect_mat_right)(human_anchor).clone();

}

void argus_depth::clustering(){
	//Mat dataset((depth_map2->rows)*(depth_map2->cols),3,CV_8UC1);
	//Mat intensity_data=(*BW_rect_mat_left)(human_anchor);
	Mat data=depth_map2->reshape(1,1);
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
	int noise_label, labelA, labelB;
	float sum_teamA=0, sum_teamB=0;
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
			sum_teamA+=dataset.at<int>(i,0);
		}else if(labels.at<int>(i,0)==(float)labelB){
			sum_teamB+=dataset.at<int>(i,0);
		};
	}

	//Decide which team should be background and which foreground
	if(sum_teamA>sum_teamB){
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
labels=labels.reshape(1,depth_map2->rows);

	bitwise_and(*depth_map2,labels,*depth_map2);
imshow("filtered",*depth_map2);
	//cout<<"data "<<data.cols<<" "<<data.rows<<" lables "<<labels.cols<<" "<<labels.rows<<"\n";

	//imshow("test",labels.reshape(1,depth_map2->rows));

	Mat jet_depth_map2(height,width,CV_8UC3);
	applyColorMap(labels, jet_depth_map2, COLORMAP_JET );
	imshow( "test", jet_depth_map2 );


	//imshow("left person",intensity_data);
	//imshow("test2",dataset2);
	//	imshow("test2",dataset);
}

int main(){
	int key_pressed=255;

	vector<ocl::Info> info;
	CV_Assert(ocl::getDevice(info));
	int devnums =ocl::getDevice(info);


	if(devnums<1){
		std::cout << "no OPENCL device found\n";
	}

	argus_depth *eye_stereo = new argus_depth();
	bool loop=false;
	while(1){
		double t = (double)getTickCount();
		eye_stereo->refresh_frame();
		//eye_stereo->compute_depth_gpu();
		eye_stereo->detect_human();
		eye_stereo->compute_depth();

		//eye_stereo->remove_background();
		t = (double)getTickCount() - t;
		eye_stereo->fps= 1/(t/cv::getTickFrequency());
		eye_stereo->refresh_window();
		eye_stereo->clustering();
		//key_pressed = cvWaitKey(1) & 255;
		do{
			key_pressed = cvWaitKey(1) & 255;
			if ( key_pressed == 32 )loop=!loop;
		}while (loop);
		if ( key_pressed == 27 ) break;

	}

	delete(eye_stereo);
	return 0;
}
