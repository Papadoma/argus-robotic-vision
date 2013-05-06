#ifndef MARKER_TRACKER_HPP
#define MARKER_TRACKER_HPP
#define WIDTH 640
#define HEIGHT 480
#define hbins	180
#define sbins	256

#include "module_input.hpp"

cv::Rect marker;
bool marking_procedure = false;
bool marker_updated = false;
cv::Mat view_frame;

class histogram_scanner{
private:
	cv::MatND marker_hist;
	cv::Mat color_frame;
	int hist_count;
public:
	void get_hist(cv::Mat);
	void reset();
	void save();
	histogram_scanner();
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
		marker_updated = true;
		marking_procedure = false;
		marker.width = x - marker.x;
		marker.height = y - marker.y;
		marker = marker & cv::Rect(0,0,WIDTH,HEIGHT);
	}
	break;
	}
}


histogram_scanner::histogram_scanner()
:hist_count(0)
{
	cv::namedWindow("Camera");
	cv::setMouseCallback( "Camera", onMouse, (void *)this );
}

void histogram_scanner::get_hist(cv::Mat input_frame){
	input_frame.copyTo(view_frame);
	cv::cvtColor(input_frame,color_frame,CV_BGR2HSV);

	cv::rectangle(view_frame,marker,cv::Scalar(0,255,0),1);
	imshow("Camera",view_frame);

	if(marker.area() && marker_updated){
		cv::Mat img_part = color_frame(marker).clone();
		int histSize[] = {hbins, sbins};
		float hue_range[] = { 0, 179 };
		float sat_range[] = { 0, 255 };
		int channels[] = {0,1};
		const float* ranges[] = { hue_range, sat_range };

		calcHist( &img_part, 1, channels, cv::Mat(), marker_hist, 2, histSize, ranges, true, true );
		std::cout<<"Calculated histogram!"<<std::endl;

		//marker_updated = false;
		hist_count++;
	}
	if(marker_hist.data){
		cv::MatND local_marker_hist;
		normalize( marker_hist/hist_count, local_marker_hist, 0, 255, cv::NORM_MINMAX, CV_8UC1, cv::Mat() );
		imshow("histogram",local_marker_hist);
	}
}

void histogram_scanner::reset(){
	marker_hist.setTo(0);
	hist_count = 0;
}

void histogram_scanner::save(){
	if(hist_count>0){
		cv::FileStorage fs("marker_histogram.yml", cv::FileStorage::WRITE);
		if (!fs.isOpened()) {std::cout << "unable to open file storage!" << std::endl; return;}
		cv::MatND rounded = marker_hist/hist_count;
		fs << "marker_histogram" << rounded;
		fs.release();
		std::cout<<"Saved histogram!"<<std::endl;
	}else{
		std::cout<<"No hist to save!"<<std::endl;
	}
}

int main(){
	module_eye input;

	cv::Mat input_frame,l;

	cv::Mat marker_img = cv::imread("green_marker.jpg");
	histogram_scanner scanner;
	while(1){
		input.getFrame(input_frame,l);
		scanner.get_hist(input_frame);

		int key_pressed = cvWaitKey(1) & 255;
		if ( key_pressed == 27 ) break;			//ESC
		if ( key_pressed == 'r' ) scanner.reset();			//ESC
		if ( key_pressed == 's' ) scanner.save();			//ESC
	}
	return 0;
}

#endif
