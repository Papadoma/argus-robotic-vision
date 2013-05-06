#ifndef MARKER_TRACKER_HPP
#define MARKER_TRACKER_HPP

//#include "module_single_cam.hpp"
#include "module_input.hpp"
#define CONTOURS_SIZE 10
#define MARKER_DENSITY 0.5
#define DEBUG_COUT_MARKER true

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

	void recalc_hist();
public:
	marker_tracker(std::string);
	cv::Point get_marker_center(cv::Mat);
	float get_density(){return marker_density;};
};

#endif
