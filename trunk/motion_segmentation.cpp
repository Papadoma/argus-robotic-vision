#include <stdio.h>
#include <opencv.hpp>
#include "module_input.hpp"

const double MHI_DURATION = 1;
const double MAX_TIME_DELTA = 0.3;

class motion_segmentation{
private:
	module_eye* input_instance;

	cv::Mat frame_grayscale;
	cv::Mat frame_left, frame_right;
	cv::Mat prev_frame, prev_mask;
	cv::Mat MHI;
	cv::Mat motion_seg;
public:
	motion_segmentation();
	~motion_segmentation();
	void get_frame();
	void refresh_output();
	void scene_segmentation();
};

motion_segmentation::motion_segmentation(){
	input_instance = new module_eye("left.mpg","right.mpg");
	cv::Size framesize = input_instance->getSize();
	prev_frame = cv::Mat::zeros(framesize,CV_8UC1);
	prev_mask = cv::Mat::ones(framesize,CV_8UC1);
	prev_mask=prev_mask*127;
	motion_seg = cv::Mat::zeros(framesize,CV_8UC1);
	MHI = cv::Mat::zeros(framesize,CV_32FC1);
}

motion_segmentation::~motion_segmentation(){
	delete(input_instance);
}
void motion_segmentation::refresh_output(){
	cv::imshow("feed",frame_left);
}

void motion_segmentation::get_frame(){
	input_instance->getFrame(frame_left,frame_right);
	cv::cvtColor(frame_left, frame_grayscale,cv::COLOR_RGB2GRAY);
}

void motion_segmentation::scene_segmentation(){
	double timestamp = (double)cv::getTickCount()/cv::getTickFrequency();

	cv::Mat frame_diff, temp1, temp2, temp3;
	absdiff(frame_grayscale, prev_frame, frame_diff);



	threshold( frame_diff, frame_diff, 20, 1, CV_THRESH_BINARY );
	frame_diff.copyTo(temp3);
	temp3=temp3*255;
	imshow("frame difference",temp3);
	cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT , cv::Size(5,5), cv::Point( 2, 2 ) );
	morphologyEx( frame_diff, frame_diff, cv::MORPH_CLOSE  , element);


	//	updateMotionHistory(frame_diff, MHI, timestamp, MHI_DURATION);
	frame_diff.copyTo(temp2);
	temp2=temp2*255;
	imshow("frame difference morph",temp2);

	cv::cvtColor(frame_diff,frame_diff,CV_GRAY2RGB);
	cv::Mat test;
	cv::bitwise_and(frame_left,frame_diff,test);
	imshow("test",test);


	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Point> joined_contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours( temp2, contours, hierarchy,CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE );
	//temp2=cv::Mat::zeros(temp2.size(),CV_8UC3);
	temp2=cv::Mat::ones(temp2.size(),CV_8UC1);
	temp2=temp2*127;
	//cv::cvtColor(temp2,temp2,CV_GRAY2RGB);
	for( int idx=0; idx < (int)hierarchy.size(); idx ++)
	{
		if(contourArea(contours[idx], false)>5){
			joined_contours.insert(joined_contours.end(),contours[idx].begin(),contours[idx].end());
			cv::Scalar color( rand()&255, rand()&255, rand()&255 );
			//drawContours( temp2, contours, idx, color, CV_FILLED, 8, hierarchy );
			drawContours( temp2, contours, idx, cv::Scalar(255), CV_FILLED, 8, hierarchy );
			drawContours( temp2, contours, idx, cv::Scalar(0), 2, 8, hierarchy );
		}
	}

//	cv::Mat frame_extract;
//	frame_left.copyTo(frame_extract);
//
//	cv::Moments moments;
//	std::vector<cv::Point> blobs_center;
//	for( int i = 0; i < (int)contours.size(); i ++)
//	{
//		if(contourArea(contours[i], false)>5){
//			moments = cv::moments(contours[i]);
//			cv::circle(frame_extract,cv::Point(moments.m10/moments.m00,moments.m01/moments.m00),2,cv::Scalar(255,255,255));
//			blobs_center.push_back(cv::Point(moments.m10/moments.m00,moments.m01/moments.m00));
//			//std::cout<<cv::Point(moments.m10/moments.m00,moments.m01/moments.m00)<<std::endl;
//		}
//	}
//	for( int i = 0; i < (int)blobs_center.size(); i ++)
//	{
//
//		floodFill(frame_extract, blobs_center[i], cv::Scalar(255,0,0),0,cv::Scalar(5,5,5),cv::Scalar(5,5,5));
//	}
	cv::bitwise_and(temp2,prev_mask,temp2);
	imshow("contours",temp2);
	//cv::rectangle(temp2,bound_rect,cv::Scalar(0,255,255),1);
	//std::vector<std::vector<cv::Point> > hull (1);
	//if(!joined_contours.empty()) approxPolyDP(joined_contours, hull[0], 100, true);//convexHull( joined_contours, hull[0], false );
	//drawContours( temp2, hull, -1, cv::Scalar(0,0,255), 1, 8 );

	//	cv::Rect bound_rect;
	//	if(!joined_contours.empty()) bound_rect = boundingRect(joined_contours);
	//	imshow("contours",temp2);
	//	cv::Mat grab_mask,bgdModel,fgdModel;
	//	grabCut(frame_left, grab_mask, bound_rect,  bgdModel,  fgdModel, 1, cv::GC_INIT_WITH_RECT  );
	//	grab_mask=grab_mask*255/3;
	//	imshow("result",grab_mask);

	//	cv::Moments moments;
	//	std::vector<cv::Point> blobs_center;
	//	for( int i = 0; i < (int)contours.size(); i ++)
	//	{
	//		moments = cv::moments(contours[i]);
	//		//cv::circle(temp2,cv::Point(moments.m10/moments.m00,moments.m01/moments.m00),2,cv::Scalar(255,255,255));
	//		blobs_center.push_back(cv::Point(moments.m10/moments.m00,moments.m01/moments.m00));
	//	}
	//	imshow("contours",temp2);
	//
	//		cv::Mat markers(frame_left.size(),CV_32S);
	//		for( int i = 0; i < (int)blobs_center.size(); i ++){
	//			markers.at<double>(blobs_center[i])=i;
	//
	//		}
	//		watershed(frame_left,  markers);
	//		markers.convertTo(markers,CV_8UC1);
	//		//markers=markers*255;
	//		std::cout << markers << std::endl;
	//		imshow("watershed",markers);


	//	//MHI.convertTo(temp2,CV_8UC1,255/(*max));
	//	imshow("MHI",MHI);
	//	//MHI.convertTo(MHI,CV_8UC1,255./MHI_DURATION,(MHI_DURATION - timestamp)*255./MHI_DURATION);
	//	cv::Mat segmask;
	//	std::vector <cv::Rect> boundingRects;
	//	cv::segmentMotion(MHI, segmask, boundingRects, timestamp, MAX_TIME_DELTA);
	//	std::cout<< MHI << std::endl;
	//	int rect_size = boundingRects.size();
	//	for(int i = 0; i<rect_size;i++){
	//		cv::rectangle(frame_left,boundingRects[i],cv::Scalar((6*i*255/rect_size)%255,8*(255-i*255/rect_size)%255,i*255*10/rect_size%255));
	//	}


	//imshow("Segmentation mask",segmask);

	temp2.copyTo(prev_mask);
	frame_grayscale.copyTo(prev_frame);
}



int main(){
	motion_segmentation test;
	bool loop=false;
	while(1){
		int key_pressed = cvWaitKey(1) & 255;
		do{
			int key_pressed = cvWaitKey(30) & 255;
			if ( key_pressed == 32 )loop=!loop;
			if ( key_pressed == 27 ) break;
		}while (loop);
		if ( key_pressed == 27 ) break;

		test.get_frame();
		test.scene_segmentation();

		test.refresh_output();
		//cvWaitKey(50);
	}
}
