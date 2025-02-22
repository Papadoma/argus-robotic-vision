#include <stdio.h>
#include <opencv.hpp>
//#include "opencv2/ocl/ocl.hpp"
//#include <skeltrack.h>
//#include <glib-object.h>

using namespace std;
using namespace cv;



class skeleton{
private:
	bool found_human;
	//

	Mat depth;
	Mat image;

	Point Head;
	Point L_Hand;
	Point R_Hand;
	Point L_Foot;
	Point R_Foot;

	Point Upper_torso;
	Point Lower_torso;

	void locate_upper_torso(Mat, Mat, Mat);
	vector<vector<Point> > segm_skel(vector<Point> , Mat );


public:
	skeleton();
	void load();
	void show();
	void find_extremas();
	void thinning1(Mat);
	void thinning2(Mat);
	void thinning3(Mat);
	int thres1;
	int thres2;

};

skeleton::skeleton(){
	found_human=false;
	Head=Point(0,0);
	L_Hand=Point(0,0);
	R_Hand=Point(0,0);
	L_Foot=Point(0,0);
	R_Foot=Point(0,0);
	Upper_torso=Point(0,0);
	Lower_torso=Point(0,0);
}

bool sort_points_by_y(Point i, Point j){return(i.y<j.y);}
bool sort_points_by_x(Point i, Point j){return(i.x<j.x);}

void skeleton::find_extremas(){
	equalizeHist(image, image);

	Mat mask;

	threshold(depth,mask,1,255,THRESH_BINARY);
	//	Mat kernel(11,11,CV_8U,cv::Scalar(1));
	//morphologyEx(mask, mask, MORPH_CLOSE , kernel);
	medianBlur(mask, mask, 7);

	imshow("Mask",mask);

	//find contour and convex of mask
	vector<vector<Point> > mask_contours;
	vector<Vec4i> mask_hierarchy;
	findContours( mask, mask_contours, mask_hierarchy,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );
	drawContours(mask, mask_contours, 0, Scalar(255), CV_FILLED);	//fill any inside holes, if any :P

	//	vector<vector<int> >mask_hull( mask_contours.size() );
	//	convexHull( Mat(mask_contours[0]), mask_hull[0] );
	//



	//prepare for skeleton forming
	Mat skeleton;
	mask.copyTo(skeleton);

	//form skeleton using thinning algorithm
	thinning1(skeleton);

	//find contour and convex of skeleton
	vector<vector<Point> > skel_contours;
	vector<Vec4i> skel_hierarchy;
	Mat skeleton2, skeleton_draw;
	skeleton.copyTo(skeleton2);	//keep skeleton intact, findcontours() changes the source
	findContours( skeleton2, skel_contours, skel_hierarchy,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE  );
	cvtColor( skeleton, skeleton_draw, CV_GRAY2BGR );	//skeleton_draw for colorful viewing purposes
	vector<vector<Point> >skel_hull( skel_contours.size() );
	vector<vector<int> >skel_hull2( skel_contours.size() );
	convexHull( Mat(skel_contours[0]), skel_hull[0], false );
	convexHull( Mat(skel_contours[0]), skel_hull2[0], true );

	drawContours(skeleton_draw, mask_contours, 0, Scalar(0,255,0), 1);

	vector <Vec4i> conv_def_idx;
	convexityDefects(skel_contours[0], skel_hull2[0], conv_def_idx);

	//Find skeleton contours approximation
	vector<Point>  approx_contours;
	approxPolyDP(skel_contours[0],approx_contours, skeleton.cols/20, true);

	//	//Draw skeleton contours approximation
	//	for(int i=0;i<(int)approx_contours.size();i++){
	//		circle(skeleton_draw, approx_contours[i], 2, Scalar(0,255,0), -1, 8, 0);
	//	}
	//	cout<<"approx"<<approx_contours.size()<<"\n";

	//Draw convex defects
	for(int i=0;i<(int)conv_def_idx.size();i++){
		Point start_p,end_p;
		start_p.x=(skel_contours[0][conv_def_idx[i][0]].x+skel_contours[0][conv_def_idx[i][1]].x)/2;
		start_p.y=(skel_contours[0][conv_def_idx[i][0]].y+skel_contours[0][conv_def_idx[i][1]].y)/2;
		end_p=skel_contours[0][conv_def_idx[i][2]];

		line(skeleton_draw, start_p, end_p, Scalar(255,255,0));
	}


	//Find possible limbs. The limb must be a skeleton's end. Using the convex hull of skeleton.
	vector<Point> limbs_buffer;
	for(int i=0;i<(int)skel_hull[0].size();i++){
		int whites_area=0;
		whites_area+=skeleton.at<uchar>(skel_hull[0][i].y-1,skel_hull[0][i].x+1)/255;
		whites_area+=skeleton.at<uchar>(skel_hull[0][i].y-1,skel_hull[0][i].x)/255;
		whites_area+=skeleton.at<uchar>(skel_hull[0][i].y-1,skel_hull[0][i].x-1)/255;
		whites_area+=skeleton.at<uchar>(skel_hull[0][i].y,skel_hull[0][i].x+1)/255;
		whites_area+=skeleton.at<uchar>(skel_hull[0][i].y,skel_hull[0][i].x-1)/255;
		whites_area+=skeleton.at<uchar>(skel_hull[0][i].y+1,skel_hull[0][i].x+1)/255;
		whites_area+=skeleton.at<uchar>(skel_hull[0][i].y+1,skel_hull[0][i].x)/255;
		whites_area+=skeleton.at<uchar>(skel_hull[0][i].y+1,skel_hull[0][i].x-1)/255;

		if(whites_area==1){
			circle(skeleton_draw, skel_hull[0][i], 4, Scalar(0,0,255), 1, 8, 0);
			limbs_buffer.push_back(skel_hull[0][i]); //found possible limbs
		}
	}
	//cout<<"Found: "<<limbs_buffer.size()<<" limbs"<<"\n";
//	cout<<limbs_buffer<<" ";
	//sort (limbs_buffer.begin(), limbs_buffer.end(), sort_points_by_y);

	vector <Point> lowerbody;
	vector <Point> upperbody;



//	cout<<limbs_buffer<<"\n";



	//Find which points are joints and not limbs
	vector<Point> temp_buffer=approx_contours;
	for(int i=0;i<(int)limbs_buffer.size();i++){
		for(int j=0;j<(int)temp_buffer.size();j++){
			if(limbs_buffer[i]==temp_buffer[j]){
				temp_buffer.erase(temp_buffer.begin()+j);
				break;
			}
		}
	}

	//Find which points are joints and related to convex defects
	vector<Point> joints_buffer;
	for(int j=0;j<(int)temp_buffer.size();j++){
		for(int i=0;i<(int)conv_def_idx.size();i++){

			if(skel_contours[0][conv_def_idx[i][2]]==temp_buffer[j]){
				joints_buffer.push_back(temp_buffer[j]);
				//temp_buffer.erase(temp_buffer.begin()+j);
				conv_def_idx.erase(conv_def_idx.begin()+i);
				break;
			}
		}
	}
	//cout<<joints_buffer;

	//	//Find presence (a joint could appear more than once)
	//	vector<int> presence;
	//	for(int i=0;i<(int)joints_buffer.size();i++){
	//		int count=0;
	//		for(int j=0;j<(int)joints_buffer.size();j++){
	//			if(joints_buffer[i]==joints_buffer[j]){
	//				count++;
	//			}
	//
	//		}
	//		presence.push_back(count);
	//	}
	//	cout<<presence;

	//Find upper and lower torso

	//Draw skeleton joints
	for(int i=0;i<(int)joints_buffer.size();i++){
		circle(skeleton_draw, joints_buffer[i], 2, Scalar(0,255,0), -1, 8, 0);
	}



	imshow("Skeleton", skeleton_draw);

	//Mat mask_A, mask_B;
	//locate_upper_torso(mask,mask_A, mask_B);
}

vector<vector<Point> > skeleton::segm_skel(vector<Point> contours, Mat skeleton){
	vector<vector<Point> > segmented;
	vector <Point> buffer;
	Mat skeleton_draw;
	cvtColor( skeleton, skeleton_draw, CV_GRAY2BGR );

	while(!contours.empty()){
		int whites_area=0;
		Point mark=contours.back();
		contours.pop_back();
		whites_area+=skeleton.at<uchar>(mark.y-1,mark.x+1)/255;
		whites_area+=skeleton.at<uchar>(mark.y-1,mark.x)/255;
		whites_area+=skeleton.at<uchar>(mark.y-1,mark.x-1)/255;
		whites_area+=skeleton.at<uchar>(mark.y,mark.x+1)/255;
		whites_area+=skeleton.at<uchar>(mark.y,mark.x-1)/255;
		whites_area+=skeleton.at<uchar>(mark.y+1,mark.x+1)/255;
		whites_area+=skeleton.at<uchar>(mark.y+1,mark.x)/255;
		whites_area+=skeleton.at<uchar>(mark.y+1,mark.x-1)/255;
		if(whites_area==3){
			circle(skeleton_draw, mark, 3, Scalar(0,255,255), 2, 8, 0);
			segmented.push_back(buffer);
			buffer.clear();
		}else{

			buffer.push_back(mark);
		}
		imshow("test",skeleton_draw);
	}

	return segmented;
}

//void skeleton::geodesic_dist(){
//
//}

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
	//test.find_extremas();

	test.show();
	while(1){
		int key_pressed = cvWaitKey(1) & 255;
		if ( key_pressed == 27 ) break;
		if ( key_pressed == 46 ) test.thres1++;
		if ( key_pressed == 44 && test.thres1>=0) test.thres1--;
		if ( key_pressed == 93 ) test.thres2++;
		if ( key_pressed == 91 && test.thres2>=0) test.thres2--;
		test.find_extremas();
		test.show();
		//test.track_skel();
	}
	//cout<<test.thres1<<" "<<test.thres2;
}
