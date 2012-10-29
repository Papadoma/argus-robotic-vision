#include <stdio.h>
#include <opencv.hpp>
#include "opencv2/ocl/ocl.hpp"
//#include <skeltrack.h>

using namespace std;
using namespace cv;

class skeleton{
private:
	Mat depth;
	Mat image;
	Mat edges;
	Mat edges2;


public:

	void load();
	void show();
	void voronoi();
	int thres1;
	int thres2;

};

void skeleton::voronoi(){
	equalizeHist(image, image);
	//	Mat grad_x, grad_y;
	//	Mat abs_grad_x, abs_grad_y;
	//
	//	/// Gradient X
	//	Sobel( image, grad_x, CV_16S , 1, 0, 3 );
	//	/// Gradient Y
	//	Sobel( image, grad_y, CV_16S , 0, 1, 3 );
	//	convertScaleAbs( grad_x, abs_grad_x );
	//	convertScaleAbs( grad_y, abs_grad_y );
	//	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, edges );
	Mat mask;
	//Canny(image, edges2, thres1, thres2, 3, false );
	//Laplacian(image, edges2, 0, 3, 1, 0, BORDER_DEFAULT );
	Sobel(image, edges2, -1, 2, 2, 5, 1, 0, BORDER_DEFAULT );
	normalize(edges2, edges2, 0.0, 255.0, NORM_MINMAX);

	threshold(depth,mask,1,255,THRESH_BINARY);
	//imshow("test",mask);
	bitwise_and(edges2,mask,edges2);

	//erode(mask, mask, Mat(),Point(),5);
	//dilate(mask, mask, Mat(),Point(),5);

	imshow("mask",mask);

	Mat dist_tran, labels,show_dist;
	distanceTransform(mask, dist_tran, labels, CV_DIST_L2, 3, DIST_LABEL_CCOMP );

	//normalize(dist_tran, show_dist, 0.0, 255.0, NORM_MINMAX);

	normalize(dist_tran, show_dist, 0.0, 255.0, NORM_MINMAX);
	show_dist.convertTo(show_dist,CV_8UC1);
	imshow("distance",show_dist);


	Mat laplace;

	Laplacian(dist_tran, laplace, -1, 3, 1, 0, BORDER_DEFAULT );
	//abs(laplace);
	//laplace.convertTo(laplace,CV_8UC1);
	normalize(laplace, laplace, 0.0, 255, NORM_MINMAX);


	laplace.convertTo(laplace,CV_8UC1);
	imshow("laplace",laplace);
	threshold(laplace, laplace, 120, 255, THRESH_BINARY_INV);
	imshow("laplace2",laplace);

	//img = cv2.imread('sofsk.png',0)
	//size = np.size(img)
	//skel = np.zeros(img.shape,np.uint8)


	Mat cropped_image;
	bitwise_and(image,mask,cropped_image);
	vector<Vec3f> circles;
	/// Apply the Hough Transform to find the circles
	HoughCircles( cropped_image, circles, CV_HOUGH_GRADIENT, 1,100, 10, 50, 0, 0 );
cout<<"circles "<<circles.size()<<"\n";
	/// Draw the circles detected
	for( size_t i = 0; i < circles.size(); i++ )
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		circle( cropped_image, center, 3, Scalar(255,255,255), -1, 8, 0 );
		// circle outline
		circle( cropped_image, center, radius, Scalar(255,255,255), 3, 8, 0 );
	}

	/// Show your results
	imshow( "detected circles", cropped_image );


	Mat skel(mask.size(), CV_8UC1, cv::Scalar(0));
	Mat temp;
	Mat eroded;

	cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

	bool done;
	do
	{
		erode(mask, eroded, element);
		dilate(eroded, temp, element); // temp = open(img)
		subtract(mask, temp, temp);
		bitwise_or(skel, temp, skel);
		eroded.copyTo(mask);

		done = (cv::norm(mask) == 0);
	} while (!done);

	imshow("Skeleton", skel);


	//
	//
	//	//normalize(labels, labels, 0.0, 1.0, NORM_MINMAX);
	//	labels.convertTo(labels, CV_8UC1);
	//
	//	//	for(int i=0;i<labels.rows;i++){
	//	//		for(int j=0;j<labels.cols;j++){
	//	//			cout<<labels.at<int>(i,j)<<" ";
	//	//		}
	//	//		cout<<"\n";
	//	//	}
	//	double min, max;
	//	minMaxLoc(labels, &min, &max, 0, 0);
	//	labels=labels*255/max;
	//	//imshow("labels",labels);
	//	//	cvWaitKey(0);

}

void skeleton::load(){
	depth=imread("depth.png",0);
	image=imread("person.png",0);
}

void skeleton::show(){
	imshow("depth", depth);
	imshow("person", image);
	//imshow("edges", edges);
	imshow("edges2", edges2);
}

int main(){

	skeleton test;
	test.load();
	test.thres1=86;
	test.thres2=86;
	while(1){
		int key_pressed = cvWaitKey(1) & 255;
		if ( key_pressed == 27 ) break;
		if ( key_pressed == 46 ) test.thres1++;
		if ( key_pressed == 44 && test.thres1>=0) test.thres1--;
		if ( key_pressed == 93 ) test.thres2++;
		if ( key_pressed == 91 && test.thres2>=0) test.thres2--;
		test.voronoi();
		test.show();
	}
	cout<<test.thres1<<" "<<test.thres2;
}
