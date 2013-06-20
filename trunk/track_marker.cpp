#include "track_marker.hpp"

marker_tracker::marker_tracker(std::string filename)
:marker_found(false),
 marker_visible(false),
 previous_state(false),
 marker_density(0)
{

	cv::FileStorage fs(filename, cv::FileStorage::READ);
	if (!fs.isOpened()) {std::cout << "[Marker tracker] unable to open marker histogram!" << std::endl; exit(1);}
	fs["marker_histogram"] >> marker_hist;
	fs.release();

	//Kalman filter part
	KF = new cv::KalmanFilter(4, 2, 0);
	KF->transitionMatrix = *(cv::Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
	//KF = new cv::KalmanFilter(6, 2, 0);
	//KF->transitionMatrix = *(cv::Mat_<float>(6, 6) << 1,0,1,0,0.5,0,   0,1,0,1,0,0.5,  0,0,1,0,1,0,  0,0,0,1,0,1, 0,0,0,0,1,0, 0,0,0,0,0,1);
	//	measurement = cv::Mat_<float>(2,1);
	//	measurement.setTo(cv::Scalar(0));

	// init...
	KF->statePre.at<float>(0) = -1;
	KF->statePre.at<float>(1) = -1;
	KF->statePre.at<float>(2) = 0;
	KF->statePre.at<float>(3) = 0;
	//KF->statePre.at<float>(4) = 0;		//////////////////
	//KF->statePre.at<float>(5) = 0;		//////////////////
	setIdentity(KF->measurementMatrix);
	setIdentity(KF->processNoiseCov, cv::Scalar::all(0.0001));
	setIdentity(KF->measurementNoiseCov, cv::Scalar::all(0.01));
	setIdentity(KF->errorCovPost, cv::Scalar::all(0.05));
}

/**
 * Computes back projection and constructs mask
 * which matches given histogram
 */
inline void marker_tracker::filter_image(){
	cv::Mat hsv;
	cv::cvtColor(color_frame, hsv, CV_BGR2HSV);

	float hue_range[] = { 0, 179 };
	float sat_range[] = { 0, 255 };
	int channels[] = {0,1};
	const float* ranges[] = { hue_range, sat_range };
	calcBackProject( &hsv, 1, channels, marker_hist, backproj, ranges, 1, true );
	//normalize( backproj, backproj, 0, 255, cv::NORM_MINMAX, -1, cv::Mat() );

	cv::Mat local_back_proj;
	normalize( backproj, local_back_proj, 0, 255, cv::NORM_MINMAX, -1, cv::Mat() );
#if DEBUG_COUT
	imshow("right_backproj",local_back_proj);
#endif

	cv::threshold(local_back_proj,marker_mask,10,255,cv::THRESH_BINARY);
	cv::morphologyEx(marker_mask,marker_mask,cv::MORPH_CLOSE,cv::Mat(),cv::Point(),1);
	cv::morphologyEx(marker_mask,marker_mask,cv::MORPH_OPEN,cv::Mat(),cv::Point(),2);
}

/**
 * Searches for the biggest blob
 */
inline void marker_tracker::find_marker(){
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

/**
 * Tracks marker using Camshift algorithm
 */
void marker_tracker::track_marker(){
	cv::TermCriteria criteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 );
	cv::RotatedRect detection;
	if(marker_bounding_rect.area()>1){
		cv::Rect local_rect = marker_bounding_rect;
		detection = CamShift(backproj, local_rect, criteria);
		cv::Rect new_marker_bounding_rect  = detection.boundingRect() & cv::Rect(0,0,color_frame.cols,color_frame.rows);
		if(new_marker_bounding_rect.area()){
			marker_bounding_rect = new_marker_bounding_rect;
		}
	}
	marker_density = (float)cv::countNonZero(backproj(marker_bounding_rect))/(detection.size.height*detection.size.width+1);
	if(marker_density>1)marker_density=1;

#if DEBUG_COUT
	std::cout<<marker_density<<std::endl;
#endif
	if(marker_density>MARKER_DENSITY){
		marker_visible = true;
		if(previous_state==false){
			KF->statePost.at<float>(0) = detection.center.x;
			KF->statePost.at<float>(1) = detection.center.y;
		}
	}else{
		marker_visible = false;
		find_marker();
	}
	previous_state = marker_visible;

	// First predict, to update the internal statePre variable
	cv::Mat prediction = KF->predict();
	filtered_center = cv::Point(prediction.at<float>(0),prediction.at<float>(1));

	if(marker_visible){
		//measured_center = cv::Point(marker_bounding_rect.x+marker_bounding_rect.width/2,marker_bounding_rect.y+marker_bounding_rect.height/2);
		measured_center = detection.center;

		// Get mouse point
		cv::Mat_<float> measurement(2,1);
		//measurement = cv::Mat_<float>(2,1);
		measurement(0) = measured_center.x;
		measurement(1) = measured_center.y;

		// The "correct" phase that is going to use the predicted value and our measurement
		cv::Mat estimated = KF->correct(measurement);
		filtered_center = cv::Point(estimated.at<float>(0),estimated.at<float>(1));
	}else if(given_center != cv::Point()){
		// Get mouse point
		cv::Mat_<float> measurement(2,1);
		//measurement = cv::Mat_<float>(2,1);
		measurement(0) = given_center.x;
		measurement(1) = given_center.y;
		cv::Mat estimated = KF->correct(measurement);
		filtered_center = cv::Point(estimated.at<float>(0),estimated.at<float>(1));
	}

	if(filtered_center.x<0 || filtered_center.y<0 || filtered_center.x>color_frame.cols || filtered_center.y>color_frame.rows)filtered_center = cv::Point(-1,-1);
}

/**
 * Refreshes histogramm
 */
void marker_tracker::recalc_hist(){
	cv::Mat local_color_frame;
	cv::cvtColor(color_frame(marker_bounding_rect),local_color_frame,CV_BGR2HSV);
	int histSize[] = {marker_hist.rows, marker_hist.cols};
	float hue_range[] = { 0, 179 };
	float sat_range[] = { 0, 255 };
	int channels[] = {0,1};
	const float* ranges[] = { hue_range, sat_range };

	calcHist( &local_color_frame, 1, channels, marker_mask(marker_bounding_rect), marker_hist, 2, histSize, ranges, true, true );
#if DEBUG_COUT
	cv::MatND local_marker_hist;
	normalize( marker_hist, local_marker_hist, 0, 255, cv::NORM_MINMAX, CV_8UC1, cv::Mat() );
	imshow("histogram",local_marker_hist);
#endif
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

/**
 * Returns calculated and filtered center of marker. If no marker is detected, returns (-1,-1)
 */
cv::Point marker_tracker::get_marker_center(cv::Mat input_frame){
	input_frame.copyTo(color_frame);

	filter_image();
	if(!marker_found)find_marker();
	if(marker_found){
		track_marker();

		if(marker_visible){
			recalc_hist();
		}
#if DEBUG_COUT
		debug_view();
#endif
		return filtered_center;
	}else{
#if DEBUG_COUT
		debug_view();
#endif
		//return filtered_center;
		return cv::Point(-1,-1);
	}
}

#undef DEBUG_COUT
