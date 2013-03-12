#include <opencv2/opencv.hpp>



int main(){
	cv::Mat color = cv::imread("snap_color.png",CV_LOAD_IMAGE_COLOR);
	cv::Mat depth = cv::imread("snap_depth.png",CV_LOAD_IMAGE_GRAYSCALE );
	//cv::cvtColor(color,color,CV_BGR2GRAY);
	//cv::cvtColor(color,color,CV_GRAY2BGR);


	cv::namedWindow("color");
	cv::namedWindow("depth");
	imshow("color", color);
	imshow("depth", depth);

	cv::Mat color_region = color.clone();
	//cv::rectangle(color_region,cv::Point(200,2),cv::Point(400,410),cv::Scalar(255),1);
	//cv::Rect ROI = cv::Rect(cv::Point(200,2),cv::Point(400,410));

	cv::rectangle(color_region,cv::Point(200,30),cv::Point(460,450),cv::Scalar(255),1);
	cv::Rect ROI = cv::Rect(cv::Point(200,30),cv::Point(460,450));


	imshow("color", color_region);

	cv::Mat mask,bgdModel,fgdModel;

	//	cv::grabCut(color, mask, ROI, bgdModel, fgdModel, 2, cv::GC_INIT_WITH_RECT  );
	//	mask -=2;
	//	mask.convertTo(mask,CV_8UC1,255);
	//	cv::namedWindow("color_result");
	//	imshow("color_result", mask);

	//	cv::grabCut(depth, mask, ROI, bgdModel, fgdModel, 2, cv::GC_INIT_WITH_RECT  );
	//	mask -=2;
	//	mask.convertTo(mask,CV_8UC1,255);
	//	cv::namedWindow("depth_result");
	//	imshow("depth_result", mask);

	cv::Mat factor, color_fix;
	depth.convertTo(factor,CV_32FC1,1./255);
	cv::Mat depth_ROI = factor(ROI);

	double minVal, maxVal;
	minMaxIdx(depth_ROI,  &minVal,  &maxVal);
	factor = min(factor,maxVal);
	factor = max(factor,minVal);
	factor.convertTo(factor,CV_32FC3,1./maxVal);
	cvtColor(factor,factor,CV_GRAY2BGR);
	std::cout << maxVal<<" "<<minVal<<std::endl;


	color.convertTo(color_fix, CV_32FC3);
	color_fix = color_fix.mul(factor);
	//multiply(color_fix, factor, color_fix, 1, CV_32FC3 );
	color_fix.convertTo(color_fix, CV_8UC3);
	cv::namedWindow("color_fix");
	imshow("color_fix", color_fix);

	cv::grabCut(color_fix, mask, ROI, bgdModel, fgdModel, 2, cv::GC_INIT_WITH_RECT  );
	mask -=2;
	mask.convertTo(mask,CV_8UC1,255);
	cv::namedWindow("color&depth_result");
	imshow("color&depth_result", mask);

cv::Mat depth2;
cv::Mat filtered;
	depth.convertTo(depth2,CV_32FC1,1./255);
	cv::Mat kernel = cv::Mat::ones( 6, 1, CV_32F );
	kernel.at<float>(3)=-1;
	kernel.at<float>(4)=-1;
	kernel.at<float>(5)=-1;
	std::cout << kernel<<std::endl;
	filter2D(depth2, filtered, CV_32F, kernel);
	cv::namedWindow("filtered");
	imshow("filtered", filtered);

	cv::waitKey(0);
	return 1;
}

