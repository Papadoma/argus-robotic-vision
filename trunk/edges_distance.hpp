#ifndef EDGES_DISTANCE_HPP
#define EDGES_DISTANCE_HPP

#include <opencv2/opencv.hpp>


class edge_similarity{
private:
	double CosineSimilarity(cv::Mat, cv::Mat);
	double HausdorffDistance(cv::Mat,cv::Mat);
	double ChamferDistance(cv::Mat, cv::Mat);
public:
	double calculate_edge_distance(cv::Mat, cv::Mat, int);
};

inline double edge_similarity::CosineSimilarity(cv::Mat array1,cv::Mat array2)
{
	//std::cout<<"CosineSimilarity"<<std::endl;
	if(array1.size() == array2.size() && array1.data)
	{
		double vectorlength1=0;
		double vectorlength2=0;
		for(int i=0;i<array1.rows;i++)
		{
			for(int j=0;j<array1.cols;j++)
			{
				vectorlength1+=array1.at<uchar>(i,j)*array1.at<uchar>(i,j);
				vectorlength2+=array2.at<uchar>(i,j)*array2.at<uchar>(i,j);
			}
		}
		vectorlength1=sqrt(vectorlength1);
		vectorlength2=sqrt(vectorlength2);


		double vectorproduct=0;
		for(int i=0;i<array1.rows;i++)
		{
			for(int j=0;j<array1.cols;j++)
			{
				double item=array1.at<uchar>(i,j)*array2.at<uchar>(i,j);
				vectorproduct+=item;
			}
		}
		//return vectorproduct/(vectorlength1*vectorlength2);
		return 1 - fabs(vectorproduct/(vectorlength1*vectorlength2));
	}
	return -1;
}

inline double edge_similarity::HausdorffDistance(cv::Mat array1,cv::Mat array2)
{
	//std::cout<<"HausdorffDistance"<<std::endl;
	if(array1.size() == array2.size() && array1.data){
		cv::Mat mat_min;
//		imshow("array1",array1);
//		imshow("array2",array2);
		array1 = 255 - array1;
		//find minimum;
		distanceTransform(array1, mat_min, CV_DIST_L1, 3);
		//cv::Mat test;
		//cv::normalize(mat_min,test,0,255,cv::NORM_MINMAX,CV_8UC1);
		//imshow("distance_trans",test);

		//find maximum
		double max_dist;
		minMaxIdx(mat_min, 0, &max_dist,0,0,array2);

		return 1-max_dist/sqrt(array1.cols*array1.cols + array1.rows*array1.rows);
	}else{
		return -1;
	}
}

inline double edge_similarity::ChamferDistance(cv::Mat array1,cv::Mat array2)
{
	//std::cout<<"ChamferDistance"<<std::endl;
	if(array1.size() == array2.size() && array1.data){
		cv::Mat mat_min;
//		imshow("array1",array1);
//		imshow("array2",array2);
		array1 = 255 - array1;
		//find minimum;
		distanceTransform(array1, mat_min, CV_DIST_L1, 3);
		//cv::Mat test;
		//cv::normalize(mat_min,test,0,255,cv::NORM_MINMAX,CV_8UC1);
		//imshow("distance",test);

		//keep only values in mask
		cv::Mat local_array2;
		array2.convertTo(local_array2,CV_32FC1,FLT_MAX/255);
		cv::bitwise_and(mat_min, local_array2, local_array2);
		cv::pow(local_array2,2,local_array2);

		double dist = 1 - sqrt(cv::sum(local_array2).val[0]/cv::countNonZero(local_array2))/10;
		if(dist<0)dist=0;
		return dist;
	}else{
		return -1;
	}
}

inline double edge_similarity::calculate_edge_distance(cv::Mat input, cv::Mat estimation, int type){
	double result;
	cv::Mat sobel_input, sobel_estimation;

	Canny( input, sobel_input, 50, 150, 3 );
	Canny( estimation, sobel_estimation, 50, 150, 3 );

	if(type == 0)result = CosineSimilarity(sobel_input,sobel_estimation);
	else if(type == 1)result = HausdorffDistance(sobel_input,sobel_estimation);
	else result = ChamferDistance(sobel_input,sobel_estimation);

//	sobel_input.convertTo(sobel_input,CV_8UC1,255);
//	sobel_estimation.convertTo(sobel_estimation,CV_8UC1,255);
	//imshow("edges_input",sobel_input);
	//imshow("edges_estimation",sobel_estimation);
	return result;
}

#endif
