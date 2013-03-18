#include "module_single_cam.hpp"



cv::MatND capture(cv::Mat input){

	cv::Mat source_template = input.clone();

	cv::Mat hsv, hue_sat;
	cvtColor( source_template, hsv, CV_BGR2HSV );

	int hbins = 30, sbins = 32;
	int histSize[] = {hbins, sbins};
	float hranges[] = { 0, 180 };
	float sranges[] = { 0, 256 };
	const float* ranges[] = { hranges, sranges };
	cv::MatND hist;
	// we compute the histogram from the 0-th and 1-st channels
	int channels[] = {0, 1};
	// Get the Histogram and normalize it
	calcHist( &hsv, 1, channels, cv::Mat(), // do not use mask
			hist, 2, histSize, ranges,
			true, // the histogram is uniform
			false );
	normalize( hist, hist, 0, 255, cv::NORM_MINMAX, -1, cv::Mat() );
	std::cout<< "Refreshed histogram" << std::endl;

	return hist;
}

int main(){
	module_eye test;
	cv::Mat left;
	cv::MatND hist;
	cv::namedWindow("left");

	cv::Mat temp;
	temp = cv::imread("wood.jpg");

	while(1){
		int key_pressed = cvWaitKey(1) & 255;
		if ( key_pressed == 27 ) break;
		if ( key_pressed == 'c' ) hist = capture(left).clone();
		if ( key_pressed == 'l' ) hist = capture(temp).clone();
		test.getFrame(left);

		cv::imshow("left",left);

		if(!hist.empty()){
			float hranges[] = { 0, 180 };
			float sranges[] = { 0, 256 };
			const float* ranges[] = { hranges, sranges };
			cv::Mat hsv;
			cvtColor( left, hsv, CV_BGR2HSV );
			cv::MatND backproj;
			const int ch[] = {0,1};
			cv::calcBackProject( &hsv, 1, ch, hist, backproj, ranges, 1, true );


			cv::threshold(backproj,backproj,10,255,CV_THRESH_BINARY);
			cv::erode(backproj,backproj,cv::Mat(),cv::Point(),1);
			imshow("test",backproj);
		}

	}
	return 0;
}
