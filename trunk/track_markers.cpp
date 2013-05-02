#ifndef MARKER_TRACKER_HPP
#define MARKER_TRACKER_HPP

#include "module_single_cam.hpp"
#define CONTOURS_SIZE 40

class marker_tracker{
private:
	cv::Mat color_frame;
	cv::MatND right_marker_hist,left_marker_hist;
	cv::Mat left_backproj, right_backproj;
	bool right_marker_visible,left_marker_visible;
	cv::Rect left_marker_br, right_marker_br;

public:
	marker_tracker();
	void set_frame(cv::Mat input_frame){input_frame.copyTo(color_frame);};
	void find_markers();
	void track_markers();
	void debug_view();
};

marker_tracker::marker_tracker()
:right_marker_visible(false),
 left_marker_visible(false)
{

	cv::Mat left_marker_img = cv::imread("white_marker.png");
	cv::Mat right_marker_img = cv::imread("green_marker.png");;
	if(!left_marker_img.data || !right_marker_img.data)exit(1);

	cvtColor( left_marker_img, left_marker_img, CV_BGR2HSV );
	cvtColor( right_marker_img, right_marker_img, CV_BGR2HSV );

	int hbins = 30, sbins = 32;
	int histSize[] = {hbins, sbins};
	float hue_range[] = { 0, 179 };
	float sat_range[] = { 0, 255 };
	int channels[] = {0, 1};
	const float* ranges[] = { hue_range, sat_range };

	calcHist( &left_marker_img, 1, channels, cv::Mat(), left_marker_hist, 2, histSize, ranges, true, false );
	calcHist( &right_marker_img, 1, channels, cv::Mat(), right_marker_hist, 2, histSize, ranges, true, false );

	//normalize( left_marker_hist, left_marker_hist, 0, 255, cv::NORM_MINMAX, -1, cv::Mat() );
	//normalize( right_marker_hist, right_marker_hist, 0, 255, cv::NORM_MINMAX, -1, cv::Mat() );


}

void marker_tracker::find_markers(){
	cv::Mat hsv;
	cv::cvtColor(color_frame, hsv, CV_BGR2HSV);

	cv::Mat left_mask,right_mask;
	float hue_range[] = { 0, 180 };
	float sat_range[] = { 0, 255 };
	int channels[] = {0, 1};
	const float* ranges[] = { hue_range, sat_range };
	calcBackProject( &hsv, 1, channels, right_marker_hist, right_backproj, ranges, 1, true );
	calcBackProject( &hsv, 1, channels, left_marker_hist, left_backproj, ranges, 1, true );

	cv::threshold(left_backproj,left_mask,50,255,cv::THRESH_BINARY);
	cv::threshold(right_backproj,right_mask,50,255,cv::THRESH_BINARY);
	cv::morphologyEx(left_mask,left_mask,cv::MORPH_OPEN,cv::Mat(),cv::Point(),1);
	cv::morphologyEx(right_mask,right_mask,cv::MORPH_OPEN,cv::Mat(),cv::Point(),1);
	cv::morphologyEx(left_mask,left_mask,cv::MORPH_CLOSE,cv::Mat(),cv::Point(),2);
	cv::morphologyEx(right_mask,right_mask,cv::MORPH_CLOSE,cv::Mat(),cv::Point(),2);

	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours( right_mask, contours, hierarchy, CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE );
	right_mask = cv::Mat::zeros( right_mask.size(), CV_8UC1 );
	right_marker_visible = false;
	right_marker_br = cv::Rect();
	for( int i = 0; i< (int)contours.size(); i++ )
	{
		if(contourArea(contours[i]) > CONTOURS_SIZE){
			drawContours( right_mask, contours, i, cv::Scalar(255), CV_FILLED, 8, hierarchy, 0, cv::Point() );
			right_marker_br = boundingRect(contours[i]);
			right_marker_visible = true;
			break;
		}
	}
	cv::imshow("right_backprojection",right_mask);
}

void marker_tracker::track_markers(){
	if(right_marker_br!=cv::Rect()){
		cv::TermCriteria criteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 );
		cv::RotatedRect detection = CamShift(right_backproj, right_marker_br, criteria);
		right_marker_br |= detection.boundingRect();
	}
}

void marker_tracker::debug_view(){
	cv::Mat local_frame = color_frame.clone();

	if(right_marker_visible){
		cv::putText(local_frame,"right marker visible!",cv::Point(5,20), 0,0.5,cv::Scalar(0,255,0),1);
		cv::rectangle(local_frame, right_marker_br,cv::Scalar(0,255,0),2);
	}else{
		cv::putText(local_frame, "right marker lost!",cv::Point(5,20), 0,0.5,cv::Scalar(0,0,255),1);
	}
	cv::imshow("input", local_frame);
}

int main(){
	module_cam input;
	marker_tracker tracker;
	cv::Mat input_frame;
	while(1){
		int key_pressed = cvWaitKey(1) & 255;
		if ( key_pressed == 27 ) break;			//ESC


		input.getFrame(input_frame);
		tracker.set_frame(input_frame);
		tracker.find_markers();
		tracker.track_markers();
		tracker.debug_view();

	}
	return 0;
}

#endif
