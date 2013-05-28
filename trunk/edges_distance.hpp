#ifndef EDGES_DISTANCE_HPP
#define EDGES_DISTANCE_HPP

#include <opencv2/opencv.hpp>


class edge_similarity{
private:
	double CosineSimilarity(cv::Mat, cv::Mat);
public:
	double calculate_edge_distance(cv::Mat, cv::Mat);
};

inline double edge_similarity::CosineSimilarity(cv::Mat array1,cv::Mat array2)
{
	if(array1.size() == array2.size() && array1.data)
	{
		double vectorlength1=0;
		double vectorlength2=0;
		for(int i=0;i<array1.rows;i++)
		{
			for(int j=0;j<array1.cols;j++)
			{
				vectorlength1+=array1.at<float>(i,j)*array1.at<float>(i,j);
				vectorlength2+=array2.at<float>(i,j)*array2.at<float>(i,j);
			}
		}
		vectorlength1=sqrt(vectorlength1);
		vectorlength2=sqrt(vectorlength2);


		double vectorproduct=0;
		for(int i=0;i<array1.rows;i++)
		{
			for(int j=0;j<array1.cols;j++)
			{
				double item=array1.at<float>(i,j)*array2.at<float>(i,j);
				vectorproduct+=item;
			}
		}
		return vectorproduct/(vectorlength1*vectorlength2);
	}
	return -1;
}

inline double edge_similarity::calculate_edge_distance(cv::Mat input, cv::Mat estimation){
	double result;
	cv::Mat sobel_input, sobel_estimation;
	Sobel(input, sobel_input, CV_32F, 1, 1);
	Sobel(estimation, sobel_estimation, CV_32F, 1, 1);
	result = CosineSimilarity(sobel_input,sobel_estimation);

	sobel_input.convertTo(sobel_input,CV_8UC1,255);
	sobel_estimation.convertTo(sobel_estimation,CV_8UC1,255);
	imshow("sobel_input",sobel_input);
	imshow("sobel_estimation",sobel_estimation);
	return result;
}

#endif
