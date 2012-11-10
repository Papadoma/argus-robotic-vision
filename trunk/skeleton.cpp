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

	Mat thinning_sub(Mat , int , int , int , int *);

	void locate_upper_torso(Mat, Mat, Mat);
public:

	void load();
	void show();
	void voronoi();
	void thinning1(Mat);
	void thinning2(Mat);
	void thinning3(Mat);
	int thres1;
	int thres2;

};



void skeleton::voronoi(){
	equalizeHist(image, image);

	Mat mask;
	threshold(depth,mask,1,255,THRESH_BINARY);
	medianBlur(mask, mask, 5);

	//erode(mask, mask, Mat(),Point(),5);
	//dilate(mask, mask, Mat(),Point(),5);

	imshow("mask",mask);



	Mat skeleton;
	mask.copyTo(skeleton);
	//	mask.copyTo(skeleton2);
	//	mask.copyTo(skeleton3);

	double t = (double)getTickCount();
	thinning1(skeleton);
	t = (double)getTickCount() - t;
	//	cout<< t*1000./cv::getTickFrequency()<<" ";


	//	t = (double)getTickCount();
	//	thinning2(skeleton2);
	//	t = (double)getTickCount() - t;
	//	cout<< t*1000./cv::getTickFrequency()<<" ";
	//
	//	t = (double)getTickCount();
	//	thinning3(skeleton3);
	//	t = (double)getTickCount() - t;
	//	cout<< t*1000./cv::getTickFrequency()<<"\n";

	imshow("custom", skeleton);
	//	imshow("erode/dilate", skeleton2);
	//	imshow("laplace", skeleton3);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	RNG rng(12345);

	Mat skeleton2;
	skeleton.copyTo(skeleton2);
	findContours( skeleton2, contours, hierarchy,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_KCOS);
	cvtColor( skeleton, skeleton, CV_GRAY2BGR );

	vector<Point> contour_aprox;
	approxPolyDP(contours[0], contour_aprox, 5, false);

	//	for( int i = 0; i< contour_aprox.size(); i++ )
	//	{
	//		Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
	//		drawContours( skeleton1, contour_aprox, i, color, 1, 8, hierarchy, 0, Point() );
	//	}
	for(int i=0;i<contour_aprox.size();i++){
		circle(skeleton, contour_aprox[i], 3, Scalar(0,255,0), 1, 8, 0);
		line(skeleton, contour_aprox[i], contour_aprox[(i+1)%contour_aprox.size()], Scalar(0,0,255), 1, 8);
	}

	imshow("contours", skeleton);

	Mat mask_A, mask_B;
	locate_upper_torso(mask,mask_A, mask_B);
}

void skeleton::load(){
	depth=imread("depth2.png",0);
	image=imread("person2.png",0);
}

void skeleton::show(){
	imshow("depth2", depth);
	imshow("person2", image);
	//imshow("edges", edges);
	//imshow("edges2", edges2);
}

void skeleton::locate_upper_torso(Mat src,Mat mask_A, Mat mask_B){
	Mat local;
	src.copyTo(local);



//	rectangle(mask_B, Point(83,0), Point(340,149), Scalar(255), CV_FILLED);		//head
//	rectangle(mask_B, Point(0,149), Point(428,532), Scalar(255), CV_FILLED);	//torso
//
//	rectangle(mask_A, Point(145,52), Point(283,212), Scalar(255), CV_FILLED);		//head
//	rectangle(mask_A, Point(66,212), Point(362,532), Scalar(255), CV_FILLED);	//torso

	mask_A=Mat::zeros(177,154,CV_8UC1);
	mask_B=Mat::zeros(177,154,CV_8UC1);
	rectangle(mask_B, Point(25,0), Point(129,50), Scalar(255), CV_FILLED);		//head
	rectangle(mask_B, Point(0,50), Point(154,177), Scalar(255), CV_FILLED);	//torso

	rectangle(mask_A, Point(52,27), Point(102,77), Scalar(255), CV_FILLED);		//head
	rectangle(mask_A, Point(27,77), Point(127,177), Scalar(255), CV_FILLED);	//torso
	mask_B-=mask_A;
	int areaA=countNonZero(mask_A);

	imshow("maskA",mask_A);
	imshow("maskB",mask_B);
}


void skeleton::thinning1(Mat init_image){
	Mat sub_result;
	init_image=~init_image;
	init_image.copyTo(sub_result);
	do{
		int count=0;
		for(int y=0;y<init_image.rows;y++){
			for(int x=0;x<init_image.cols;x++){
				int P1=init_image.at<uchar>(y,x);
				if((P1==0) && (y>0) && (x>0) && (y<init_image.rows-1) && (x<init_image.cols-1)){
					int AP1=0, BP1=0;
					Mat element(1,8,CV_8UC1);
					BP1+=element.at<uchar>(0,0)=(1-(init_image.at<uchar>(y-1,x)/255));  	//2
					BP1+=element.at<uchar>(0,1)=(1-(init_image.at<uchar>(y-1,x+1)/255));  	//3
					BP1+=element.at<uchar>(0,2)=(1-(init_image.at<uchar>(y,x+1)/255));		//4
					BP1+=element.at<uchar>(0,3)=(1-(init_image.at<uchar>(y+1,x+1)/255));	//5
					BP1+=element.at<uchar>(0,4)=(1-(init_image.at<uchar>(y+1,x)/255));		//6
					BP1+=element.at<uchar>(0,5)=(1-(init_image.at<uchar>(y+1,x-1)/255));	//7
					BP1+=element.at<uchar>(0,6)=(1-(init_image.at<uchar>(y,x-1)/255));		//8
					BP1+=element.at<uchar>(0,7)=(1-(init_image.at<uchar>(y-1,x-1)/255));	//9

					for(int i=0;i<8;i++){
						if( (element.at<uchar>(0,i)==0) && (element.at<uchar>(0,(i+1)%8)==1)){
							AP1++;
						}
					}
					if((AP1==1)&&(BP1>=2)&&(BP1<=6)&&(element.at<uchar>(0,0)*element.at<uchar>(0,2)*element.at<uchar>(0,4)==0)&&(element.at<uchar>(0,2)*element.at<uchar>(0,4)*element.at<uchar>(0,6)==0)){
						sub_result.at<uchar>(y,x)=255;
						count+=1;
					}
				}
			}
		}
		sub_result.copyTo(init_image);
		if(count==0)break;
		for(int y=0;y<init_image.rows;y++){
			for(int x=0;x<init_image.cols;x++){
				int P1=init_image.at<uchar>(y,x);
				if((P1==0) && (y>0) && (x>0) && (y<init_image.rows-1) && (x<init_image.cols-1)){
					int AP1=0, BP1=0;
					Mat element(1,8,CV_8UC1);
					BP1+=element.at<uchar>(0,0)=(1-(init_image.at<uchar>(y-1,x)/255));  	//2
					BP1+=element.at<uchar>(0,1)=(1-(init_image.at<uchar>(y-1,x+1)/255));  	//3
					BP1+=element.at<uchar>(0,2)=(1-(init_image.at<uchar>(y,x+1)/255));		//4
					BP1+=element.at<uchar>(0,3)=(1-(init_image.at<uchar>(y+1,x+1)/255));	//5
					BP1+=element.at<uchar>(0,4)=(1-(init_image.at<uchar>(y+1,x)/255));		//6
					BP1+=element.at<uchar>(0,5)=(1-(init_image.at<uchar>(y+1,x-1)/255));	//7
					BP1+=element.at<uchar>(0,6)=(1-(init_image.at<uchar>(y,x-1)/255));		//8
					BP1+=element.at<uchar>(0,7)=(1-(init_image.at<uchar>(y-1,x-1)/255));	//9

					for(int i=0;i<8;i++){
						if( (element.at<uchar>(0,i)==0) && (element.at<uchar>(0,(i+1)%8)==1)){
							AP1++;
						}
					}
					if((AP1==1)&&(BP1>=2)&&(BP1<=6)&&(element.at<uchar>(0,0)*element.at<uchar>(0,2)*element.at<uchar>(0,6)==0)&&(element.at<uchar>(0,0)*element.at<uchar>(0,4)*element.at<uchar>(0,6)==0)){
						sub_result.at<uchar>(y,x)=255;
						count+=1;
					}
				}
			}
		}
		sub_result.copyTo(init_image);
		if(count==0)break;
	}while(1);
	init_image=~init_image;
}

void skeleton::thinning2(Mat src){
	Mat skel(src.size(), CV_8UC1, cv::Scalar(0));
	Mat temp;
	Mat eroded;
	Mat element = getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
	bool done;
	do
	{
		erode(src, eroded, element);
		dilate(eroded, temp, element);
		subtract(src, temp, temp);
		bitwise_or(skel, temp, skel);
		eroded.copyTo(src);

		done = (cv::norm(src) == 0);
	} while (!done);
	skel.copyTo(src);
}

void skeleton::thinning3(Mat src){
	Mat dist_tran, labels,show_dist;
	distanceTransform(src, dist_tran, labels, CV_DIST_L2, 3, DIST_LABEL_CCOMP );

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
	//threshold(laplace, laplace, 120, 255, THRESH_BINARY_INV);
	adaptiveThreshold(laplace, laplace, 255, ADAPTIVE_THRESH_GAUSSIAN_C , THRESH_BINARY_INV, 19, 30);
	//erode(laplace, laplace, Mat(),Point(),2);
	//dilate(laplace, laplace, Mat(),Point(),2);
	imshow("laplace2",laplace);
}

int main(){
	cout<<"start"<<"\n";
	skeleton test;
	test.load();
	test.thres1=86;
	test.thres2=86;
	//test.voronoi();

	test.show();
	while(1){
		int key_pressed = cvWaitKey(1) & 255;
		if ( key_pressed == 27 ) break;
		if ( key_pressed == 46 ) test.thres1++;
		if ( key_pressed == 44 && test.thres1>=0) test.thres1--;
		if ( key_pressed == 93 ) test.thres2++;
		if ( key_pressed == 91 && test.thres2>=0) test.thres2--;
		test.voronoi();
		test.show();
		//test.track_skel();
	}
	//cout<<test.thres1<<" "<<test.thres2;
}
