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

	Point Head;
	Point L_Hand;
	Point R_Hand;
	Point L_Foot;
	Point R_Foot;


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
	//	Mat kernel(11,11,CV_8U,cv::Scalar(1));
	//morphologyEx(mask, mask, MORPH_CLOSE , kernel);
	medianBlur(mask, mask, 5);
	//erode(mask, mask, Mat(),Point(),5);
	//dilate(mask, mask, Mat(),Point(),5);

	imshow("mask",mask);



	//Mat skeleton(depth.rows,depth.cols,CV_8UC1);
	Mat skeleton;
	mask.copyTo(skeleton);

	double t = (double)getTickCount();
	thinning1(skeleton);
	t = (double)getTickCount() - t;

	imshow("custom", skeleton);
	//	imshow("erode/dilate", skeleton2);
	//	imshow("laplace", skeleton3);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	Mat skeleton2;
	skeleton.copyTo(skeleton2);

	findContours( skeleton2, contours, hierarchy,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_KCOS);
	//cvtColor( skeleton, skeleton, CV_GRAY2BGR );

	//vector<Point> contour_aprox;
	//approxPolyDP(contours[0], contour_aprox, 10, false);

	vector<vector<Point> >hull( contours.size() );
	for( int i = 0; i < (int)contours.size(); i++ )convexHull( Mat(contours[i]), hull[i], false );
	//drawContours( skeleton, hull, -1, Scalar(255));


	//Find possible limbs. The limb must be a skeleton's end
	vector<Point> limbs;
	for(int i=0;i<(int)hull[0].size();i++){
		int whites_area=0;
		whites_area+=skeleton.at<uchar>(hull[0][i].y-1,hull[0][i].x+1)/255;
		whites_area+=skeleton.at<uchar>(hull[0][i].y-1,hull[0][i].x)/255;
		whites_area+=skeleton.at<uchar>(hull[0][i].y-1,hull[0][i].x-1)/255;
		whites_area+=skeleton.at<uchar>(hull[0][i].y,hull[0][i].x+1)/255;
		whites_area+=skeleton.at<uchar>(hull[0][i].y,hull[0][i].x-1)/255;
		whites_area+=skeleton.at<uchar>(hull[0][i].y+1,hull[0][i].x+1)/255;
		whites_area+=skeleton.at<uchar>(hull[0][i].y+1,hull[0][i].x)/255;
		whites_area+=skeleton.at<uchar>(hull[0][i].y+1,hull[0][i].x-1)/255;
		//cout<<whites_area<<hull[0][i]<<"\n";
		if(whites_area==1)limbs.push_back(hull[0][i]); //found possible limbs
	}
	cout<<limbs.size()<<"\n";

	//Sort the limbs by y distance using bubblesort ;)
	for(int i=(int)limbs.size();i>1;i--){
		for(int j=0;j<i-1;j++){
			if(limbs[j].y>limbs[j+1].y){
				Point temp=limbs[j+1];
				limbs[j+1]=limbs[j];
				limbs[j]=temp;
			}
		}
	}

	//Define feet
	Point temp1, temp2;
	temp1=limbs.back();
	temp2=limbs.back();



	circle(skeleton, R_Foot, 3, Scalar(127), 2, 8, 0);



	//	for( int i = 0; i< contour_aprox.size(); i++ )
	//	{
	//		Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
	//		drawContours( skeleton1, contour_aprox, i, color, 1, 8, hierarchy, 0, Point() );
	//	}

	//		for(int i=0;i<(int)contour_aprox.size();i++){
	//			circle(skeleton, contour_aprox[i], 3, Scalar(0,255,0), 1, 8, 0);
	//			line(skeleton, contour_aprox[i], contour_aprox[(i+1)%contour_aprox.size()], Scalar(0,0,255), 1, 8);
	//		}



	imshow("contours", skeleton);

	//Mat mask_A, mask_B;
	//locate_upper_torso(mask,mask_A, mask_B);
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

	mask_A=Mat::zeros(177,154,CV_8UC1);
	mask_B=Mat::zeros(177,154,CV_8UC1);
	rectangle(mask_B, Point(25,0), Point(129,50), Scalar(255), CV_FILLED);		//head
	rectangle(mask_B, Point(0,50), Point(154,177), Scalar(255), CV_FILLED);	//torso

	rectangle(mask_A, Point(52,27), Point(102,77), Scalar(255), CV_FILLED);		//head
	rectangle(mask_A, Point(27,77), Point(127,177), Scalar(255), CV_FILLED);	//torso
	mask_B-=mask_A;
	int areaA=countNonZero(mask_A);


	int maxloc,radius=100;
	double best_score,search_score;


	Mat search_subimage, temp_result;;
	int center_x=src.cols/2;
	int center_y=src.rows/3;
	Rect search_mask(center_x-154/2,center_y-154/2,154,177);
	local(search_mask).copyTo(search_subimage);
	bitwise_and(mask_A,search_subimage,temp_result);
	best_score=countNonZero(temp_result);
	bitwise_and(mask_B,search_subimage,temp_result);
	best_score-=countNonZero(temp_result);
	best_score/=areaA;

	//rectangle(local, search_mask, Scalar(255));
	imshow("subimage",search_subimage);
	//	imshow("location",local);
	src.copyTo(local);


	bool searching=true;
	//search the best solution
	while(searching){
		float PI=3.14159265;
		Mat score(1,360,CV_32FC1);
		int temp_x, temp_y;
		for(int j=0;j<360;j++){
			temp_x=(int)((center_x+sin(j*PI/180)*radius)-search_mask.width/2) ;
			temp_y=(int)((center_y+cos(j*PI/180)*radius)-search_mask.height/2) ;
			if((temp_x<=local.cols-search_mask.width)&&(temp_x>=0))search_mask.x=temp_x;
			if((temp_y<=local.rows-search_mask.height)&&(temp_y>=0))search_mask.y=temp_y;

			local(search_mask).copyTo(search_subimage);

			bitwise_and(mask_A,search_subimage,temp_result);
			score.at<float>(0,j)=countNonZero(temp_result);
			imshow("test",temp_result);
			bitwise_and(mask_B,search_subimage,temp_result);

			score.at<float>(0,j)-=countNonZero(temp_result);
			score.at<float>(0,j)/=areaA;


			rectangle(local, search_mask, Scalar(127));
			line(local, Point(center_x,center_y), Point(temp_x+search_mask.width/2,temp_y+search_mask.height/2), Scalar(127));
			imshow("location",local);
			cvWaitKey(5);
			src.copyTo(local);
		}

		minMaxIdx(score,NULL,&search_score,NULL,&maxloc);
		if(search_score>best_score){
			center_x=(int)(center_x+sin(maxloc*PI/180)*radius) ;
			center_y=(int)(center_y+cos(maxloc*PI/180)*radius) ;
			best_score=search_score;
			radius/=2;
			cout<<"FOUND";
		}else{
			cout<<"not found"<<" "<<search_score<<" "<<best_score<<"\n";
			//radius*=2;
			if(radius==0)searching=false;

		}
	}

	//}

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
