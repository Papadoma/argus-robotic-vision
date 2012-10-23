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
	Canny(image, edges2, thres1, thres2, 3, false );
	threshold(depth,mask,1,255,THRESH_BINARY);
	//imshow("test",mask);
	bitwise_and(edges2,mask,edges2);

	imshow("before",mask);
	dilate(mask,mask,Mat(),Point(),1);
	erode(mask,mask,Mat(),Point(),1);
	imshow("after",mask);
}

void skeleton::load(){
	depth=imread("depth2.png",0);
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
	cout<<test.thres1<<" "<<test.thres1;
}
