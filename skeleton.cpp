#include <stdio.h>
#include <opencv.hpp>
//#include "opencv2/ocl/ocl.hpp"
//#include <skeltrack.h>
//#include <glib-object.h>

using namespace std;
using namespace cv;

class skeleton{
private:
	Mat depth;
	Mat image;
	Mat edges;
	Mat edges2;

	//	SkeltrackSkeleton *skeleton;
	//	SkeltrackJointList list;

public:

	void load();
	void show();
	void voronoi();
	Mat thinning(Mat);
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

	imshow("thinning", thinning(mask));

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

Mat skeleton::thinning(Mat src){
	//src.convertTo(src,CV_8U);
	normalize(src, src, 0.0, 255.0, NORM_MINMAX);
	Mat result;
	//Mat marker=Mat::zeros(src.rows, src.cols, CV_8UC1);
	int count=0;
	for(int y=0;y<src.rows;y++){
		for(int x=0;x<src.cols;x++){
			int P1=src.at<int>(0,y,x);
//			if((P1==255) && (y>0) && (x>0)){
//				int AP1=0;
//				Mat element(1,8,CV_8UC1);
//				//at<int>(rows,cols)
//				element.at<int>(0,0,0)=src.at<int>(y-1,x,0);
//				element.at<int>(0,1,0)=src.at<int>(y-1,x+1,0);
//				element.at<int>(0,2,0)=src.at<int>(y,x+1,0);
//				element.at<int>(0,3,0)=src.at<int>(y+1,x+1,0);
//				element.at<int>(0,4,0)=src.at<int>(y+1,x,0);
//				element.at<int>(0,5,0)=src.at<int>(y+1,x-1,0);
//				element.at<int>(0,6,0)=src.at<int>(y,x-1,0);
//				element.at<int>(0,7,0)=src.at<int>(y-1,x-1,0);
//				for(int i=0;i<8;i++){
//					if( (element.at<int>(0,i,0)==0) && (element.at<int>(0,(i+1)%8,0)==255)){
//						AP1++;
//					}
//				}
//
//
//			}
			cout<<(int)P1<<"_";
			count++;
			//src.at<int>(y,x,1)=255;

			//cvWaitKey(100);
		}
		cout<<"\n";

	}
	cout<<"\n"<<count;
	imshow("test",src);

	return src;
}

int main(){
cout<<"start"<<"\n";
	skeleton test;
	test.load();
	test.thres1=86;
	test.thres2=86;
	test.voronoi();

	test.show();
	while(1){
		int key_pressed = cvWaitKey(1) & 255;
		if ( key_pressed == 27 ) break;
		if ( key_pressed == 46 ) test.thres1++;
		if ( key_pressed == 44 && test.thres1>=0) test.thres1--;
		if ( key_pressed == 93 ) test.thres2++;
		if ( key_pressed == 91 && test.thres2>=0) test.thres2--;
		//test.voronoi();
		//test.show();
		//test.track_skel();
	}
	cout<<test.thres1<<" "<<test.thres2;
}
