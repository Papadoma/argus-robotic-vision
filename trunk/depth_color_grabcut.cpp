#include <opencv2/opencv.hpp>

//#include <pcl/io/pcd_io.h>
//#include <pcl/point_types.h>
//#include <pcl/filters/statistical_outlier_removal.h>
cv::Mat Q;

/**
 * Load camera parameters
 */
void load_param(){
	std::cout << "Loading settings"<< std::endl;
	std::string extrinsics="extrinsics_eye.yml";

	cv::FileStorage fs;

	fs.open(extrinsics, CV_STORAGE_READ);
	if( fs.isOpened() )
	{
		fs["Q"] >> Q;
		fs.release();
	}
	else
		std::cout << "Error: can not load the extrinsics parameters" << std::endl;

}

cv::Mat cloud_to_disparity(cv::Mat xyz){
	cv::Mat disparity = cv::Mat(xyz.rows,xyz.cols,CV_32FC1);
	float z,y;
	for(int i=0;i<xyz.rows;i++){
		for(int j=0;j<xyz.cols;j++){
			z=xyz.ptr<float>(i)[3*j+2];
			y=xyz.ptr<float>(i)[3*j+1];
			//if(y>-95&&z>-350){
			disparity.at<float>(i,j)=Q.at<double>(2,3)/z*Q.at<double>(3,2);
			//}else{
			//disparity.at<float>(i,j)=0;
			//}

		}
	}

	normalize(disparity, disparity, 0.0, 255.0, cv::NORM_MINMAX);
	disparity.convertTo(disparity,CV_8UC1);
	return disparity;
}

bool coplanarity(cv::Mat cloud){
	//check for coplanarity of points
	int rows = cloud.rows;
	int cols = cloud.cols;

	std::vector<cv::Mat> cloudXYZ;
	split( cloud, cloudXYZ );
	cloudXYZ[0] = cloudXYZ[0].reshape(1, rows*cols); //rows*cols rows
	cloudXYZ[1] = cloudXYZ[1].reshape(1, rows*cols);
	cloudXYZ[2] = cloudXYZ[2].reshape(1, rows*cols);

	std::cout << "hi"<<std::endl;
	//std::cout << cloudXYZ[0].cols<<std::endl;
	cv::Mat combined(rows*cols, 3, CV_32FC1);
	cloudXYZ[0].copyTo(combined.col(0));
	cloudXYZ[1].copyTo(combined.col(1));
	cloudXYZ[2].copyTo(combined.col(2));

	cv::Mat_<double> mean;
	cv::Mat test2;

	combined +=1;
	//	int cvIsInf(double value)

	cv::PCA pca(combined,mean,CV_PCA_DATA_AS_ROW);

	double p_to_plane_thresh = pca.eigenvalues.at<double>(2);
	int num_inliers = 0;
	cv::Vec3d nrm = pca.eigenvectors.row(2);
	nrm = nrm / norm(nrm);
	cv::Vec3d x0 = pca.mean;
	std::vector<cv::Point3d> test;
	for (int i=0; i<rows*cols; i++) {
		cv::Vec3d w = cv::Vec3d(combined.at<float>(i,0),combined.at<float>(i,1),combined.at<float>(i,2)) - x0;
		double D = fabs(nrm.dot(w));

		if(D < p_to_plane_thresh) num_inliers++;
	}

	std::cout << num_inliers << "/" << rows*cols << " are coplanar" << std::endl;
	if((double)num_inliers / (double)(rows*cols) > 0.9)
		return false;
}


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

	//	cv::grabCut(color_fix, mask, ROI, bgdModel, fgdModel, 2, cv::GC_INIT_WITH_RECT  );
	//	mask -=2;
	//	mask.convertTo(mask,CV_8UC1,255);
	//	cv::namedWindow("color&depth_result");
	//	imshow("color&depth_result", mask);

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

	load_param();
	cv::Mat disparity, image3d;
	depth.convertTo(disparity,CV_32FC1,32./255);
	//disparity = max(disparity,1);
	reprojectImageTo3D(disparity, image3d, Q, false, -1);

	coplanarity(image3d);

	disparity = cloud_to_disparity(image3d);
	//disparity.convertTo(disparity, CV_32FC1);
	cv::namedWindow("cloud_map");
	imshow("cloud_map", disparity);

	//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

	cv::waitKey(0);
	return 1;
}

