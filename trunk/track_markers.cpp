#ifndef MARKER_TRACKER_HPP
#define MARKER_TRACKER_HPP

//#include "module_single_cam.hpp"
#include "module_input.hpp"
#define CONTOURS_SIZE 10
#define MARKER_DENSITY 0.3

class marker_tracker{
private:
	cv::Mat color_frame;
	cv::MatND marker_hist;

	cv::Mat marker_mask;
	cv::Mat backproj;
	bool marker_found;
	bool marker_visible;
	cv::Rect marker_bounding_rect;
	float marker_density;

	cv::KalmanFilter* KF;

	cv::Point measured_center, filtered_center;

	void filter_image();
	void find_marker();
	void track_marker();
	void debug_view();

public:
	marker_tracker(cv::Mat);
	cv::Point get_marker_center(cv::Mat);
};

marker_tracker::marker_tracker(cv::Mat marker_img)
:marker_found(false),
 marker_visible(false),
 marker_density(0)
{
	if(!marker_img.data)exit(1);

	cv::Mat marker_local_img;
	cvtColor( marker_img, marker_local_img, CV_BGR2HSV );

	int hbins = 180, sbins = 256;
	int histSize[] = {hbins, sbins};
	float hue_range[] = { 0, 179 };
	float sat_range[] = { 0, 255 };
	int channels[] = {0,1};
	const float* ranges[] = { hue_range, sat_range };

	calcHist( &marker_local_img, 1, channels, cv::Mat(), marker_hist, 2, histSize, ranges, true, false );
	//normalize( marker_hist, marker_hist, 0, 255, cv::NORM_MINMAX, -1, cv::Mat() );

	//Kalman filter part
	KF = new cv::KalmanFilter(4, 2, 0);
	KF->transitionMatrix = *(cv::Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
	//	measurement = cv::Mat_<float>(2,1);
	//	measurement.setTo(cv::Scalar(0));

	// init...
	KF->statePre.at<float>(0) = -1;
	KF->statePre.at<float>(1) = -1;
	KF->statePre.at<float>(2) = 0;
	KF->statePre.at<float>(3) = 0;
	setIdentity(KF->measurementMatrix);
	setIdentity(KF->processNoiseCov, cv::Scalar::all(0.0001));
	setIdentity(KF->measurementNoiseCov, cv::Scalar::all(0.01));
	setIdentity(KF->errorCovPost, cv::Scalar::all(0.05));
}



/**
 * Computes back projection and constructs mask
 * which matches given histogram
 */
void marker_tracker::filter_image(){
	cv::Mat hsv;
	cv::cvtColor(color_frame, hsv, CV_BGR2HSV);

	float hue_range[] = { 0, 179 };
	float sat_range[] = { 0, 255 };
	int channels[] = {0,1};
	const float* ranges[] = { hue_range, sat_range };
	calcBackProject( &hsv, 1, channels, marker_hist, backproj, ranges, 1, true );
	//normalize( backproj, backproj, 0, 255, cv::NORM_MINMAX, -1, cv::Mat() );

	imshow("right_backproj",backproj);

	cv::threshold(backproj,marker_mask,10,255,cv::THRESH_BINARY);
	cv::morphologyEx(marker_mask,marker_mask,cv::MORPH_OPEN,cv::Mat(),cv::Point(),1);
	cv::morphologyEx(marker_mask,marker_mask,cv::MORPH_CLOSE,cv::Mat(),cv::Point(),2);
}

void marker_tracker::find_marker(){
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours( marker_mask.clone(), contours, hierarchy, CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE );

	//find max blob
	double max_area = 0;
	int max_pos = 0;
	for( int i = 0; i< (int)contours.size(); i++ )
	{
		double cont_area = contourArea(contours[i]);
		if(cont_area > max_area){
			max_area = cont_area;
			max_pos = i;
		}
	}

	//update marker bounding rectangle and existance
	if( max_area > CONTOURS_SIZE){
		drawContours( marker_mask, contours, max_pos, cv::Scalar(255), CV_FILLED, 8, hierarchy, 0, cv::Point() );
		marker_bounding_rect = boundingRect(contours[max_pos]);
		marker_found = true;
		//marker_visible = true;
	}

}

void marker_tracker::track_marker(){
	cv::TermCriteria criteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 );

	if(marker_bounding_rect.area()>1){
		cv::RotatedRect detection = CamShift(backproj, marker_bounding_rect, criteria);
		if(detection.boundingRect().area())marker_bounding_rect = detection.boundingRect() & cv::Rect(0,0,color_frame.cols,color_frame.rows);
	}
	marker_density = (float)cv::countNonZero(backproj(marker_bounding_rect))/marker_bounding_rect.area();

	if(marker_density>MARKER_DENSITY){
		marker_visible = true;
	}else{
		marker_visible = false;
		find_marker();
	}





	// First predict, to update the internal statePre variable
	cv::Mat prediction = KF->predict();
	filtered_center = cv::Point(prediction.at<float>(0),prediction.at<float>(1));


	if(filtered_center.x<0)filtered_center=measured_center;
	if(filtered_center.y<0)filtered_center=measured_center;
	if(filtered_center.x>color_frame.cols)filtered_center=measured_center;
	if(filtered_center.y>color_frame.rows)filtered_center=measured_center;

	if(marker_visible || marker_bounding_rect.contains(filtered_center)){
		measured_center = cv::Point(marker_bounding_rect.x+marker_bounding_rect.width/2,marker_bounding_rect.y+marker_bounding_rect.height/2);


		// Get mouse point
		cv::Mat_<float> measurement(2,1);
		//measurement = cv::Mat_<float>(2,1);
		measurement(0) = measured_center.x;
		measurement(1) = measured_center.y;

		// The "correct" phase that is going to use the predicted value and our measurement
		cv::Mat estimated = KF->correct(measurement);
		filtered_center = cv::Point(estimated.at<float>(0),estimated.at<float>(1));
	}

	if(filtered_center.x<0)filtered_center.x=0;
	if(filtered_center.y<0)filtered_center.y=0;
	if(filtered_center.x>color_frame.cols)filtered_center.x=color_frame.cols;
	if(filtered_center.y>color_frame.rows)filtered_center.y=color_frame.rows;

}

void marker_tracker::debug_view(){
	cv::Mat local_frame = color_frame.clone();

	if(marker_visible){
		cv::putText(local_frame,"marker visible!",cv::Point(5,20), 0,0.5,cv::Scalar(0,255,0),2);
	}
	else if(marker_found){
		cv::putText(local_frame,"marker lost!",cv::Point(5,20), 0,0.5,cv::Scalar(50,100,120),2);
	}else{
		cv::putText(local_frame, "marker not found!",cv::Point(5,20), 0,0.5,cv::Scalar(0,0,255),2);
	}
	cv::rectangle(local_frame, marker_bounding_rect,cv::Scalar(0,255,0),2);
	cv::circle(local_frame,measured_center,2,cv::Scalar(0,255,0),2);
	cv::circle(local_frame,filtered_center,3,cv::Scalar(0,0,255),-2);
	cv::imshow("input", local_frame);

	cv::imshow("marker_mask",marker_mask);
}

cv::Point marker_tracker::get_marker_center(cv::Mat input_frame){
	input_frame.copyTo(color_frame);

	filter_image();
	if(!marker_found)find_marker();
	if(marker_found){
		track_marker();

//		if(marker_visible){
//			debug_view();
//			return measured_center;
//		}else{
//			debug_view();
//			return filtered_center;
//		}
		debug_view();
		return filtered_center;
	}else{
		debug_view();
		return cv::Point(-1,-1);
	}
}

int main(){
	module_eye input;

	cv::Mat input_frame,l;

	cv::Mat marker_img = cv::imread("green_marker.jpg");
	marker_tracker tracker(marker_img);
	while(1){
		int key_pressed = cvWaitKey(1) & 255;
		if ( key_pressed == 27 ) break;			//ESC

		input.getFrame(input_frame,l);
		cv::Point mark = tracker.get_marker_center(input_frame);
		cv::circle(input_frame,mark,2,cv::Scalar(0,255,0),2);
		cv::imshow("output",input_frame);
	}
	return 0;
}

#endif
