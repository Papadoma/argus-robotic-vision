/////////////////////////////////////////////////////////////////
// Filename: classMotionDetector.cpp							/
// Author:   George Aprilis										/
// Date:     05.03.2011 20:04:13								/
//																/
// Description: class for Implementation of Motion Detection	/
//              using thresholding of absolute difference		/
//              between a sequence of frames. Returns 3 states:	/
//				Much Movement, Notion of Movement, and No Move- /
//				ment at all.									/
/////////////////////////////////////////////////////////////////

#include "MotionDetector.h"



/**
 * Class Constructor
 * Initializes varialbes used in detecting. Here is
 * were desirable threashold values will be determined. 
 */

MotionDetector::MotionDetector()
{
	N = 4;						//Buffer size depends on how "old" will be the frame that is
								//compaired with the current frame.
	diff_threshold = 45;		//threshold of absolute difference in value between 2 pixels
								//so that the pixels can be considered as "different" 
	motion_high_thres = 7500;	//Minimum mumber of pixels that must be "different" in the
								//two frames so that motion can be evaluated as Extensive
	motion_low_thres = 200;		//Minimum number of pixels that must be "differenet" in the
								//two frames so that motion can be evaluated as Slight
	buf = 0;
	tmp = 0;
	last = 0;
	count = 0;
	flagCounter = 0;

	cout << "Created MotionDetector instance" << endl;
}

/**
 * Class Destructor
 * Deallocates memory used for storing images
 */

MotionDetector::~MotionDetector()
{	
	//mike
	/* causes Opencv Error
	 * what if buf is empty???? 
	*/
	//george
	/* vazw ena if kai telos :P
	 */
	
	if ( buf != 0 ){
		for (int i = 0; i < N; i++){
		cvReleaseImage( &buf[i] );
		}
	}
	
	cout << "Destroying MotionDetector instance" << endl;
}



/********************** detectMotion ******************************
 * Type:		PUBLIC
 * Description: Using a N-sized buffer compares current given frame
 * 				with last - (N-1)th frame and calculates an the dif-
 * 				ference between the two frames giving value to a va-
 * 				riable integer "count" with the number of different
 * 				pixels. According to given thresholds, returns:
 * 					-1				Error in frame input
 * 					 0				Insignificant Motion
 * 					 1				Slight Motion
 * 					 2				Extensive Motion						
 * 				Function detectMotion() of a specific MotionDetector
 * 				is run in a loop.	  
 * @param		frame 	The current frame given as input 
 * @return 		integer Index of evaluation of Motion in N frames.
 ******************************************************************/

int MotionDetector::detectMotion(IplImage* frame)
{	
	int idx1, idx2, result = 0;
	//Check if frame was retrieved successfully
	if(!frame) return -1;
	//Allocate buffer if it's the first time
	if( buf == 0 ) {
            buf = (IplImage**)malloc(N*sizeof(buf[0]));
            memset( buf, 0, N*sizeof(buf[0]));
            
            for(int i=0 ; i<N ; i++){
				buf[i] = cvCreateImage( cvSize(frame->width, frame->height), IPL_DEPTH_8U, 1 );
				cvZero( buf[i] );
			}
    }
	
	//Make a temporary copy
	tmp = cvCloneImage(frame);
	cvSmooth( tmp , tmp , CV_GAUSSIAN , 21, 21); //Smooth image	
	
	idx1 = last;
	cvCvtColor( tmp, buf[last], CV_BGR2GRAY ); 	// convert frame to grayscale
	idx2 = (last + 1) % N; 						// index of (last - (N-1))th frame
	last = idx2;

	dif = buf[idx2];
	cvAbsDiff( buf[idx1], buf[idx2], dif );			// calculate absolute difference
	cvThreshold( dif, dif, diff_threshold, 255, CV_THRESH_BINARY ); // threshold frame
	cvErode( dif, dif, NULL, 4);

	
	//Buffer will be emty for the first frames so comparison will will be wrong
	//A safe value N+1 is chosen to start returning results
	if(flagCounter>N+1){
		
		//debug
		//cvShowImage("Difference" , dif);
		//cvShowImage("Camera" , frame);
		
		//counts value of non zero pixels in binary image
		count = cvRound(cvNorm( dif, 0, CV_L1, 0 )/255.); // divided with 255 because TRUE equals 255 in RGB
		
		if (count > motion_high_thres){
			result = 2;
		}
		else if (count > motion_low_thres){
			result = 1;
		}
		else{ 
			result = 0; 
		}
		
	}
	if (flagCounter < 2*N) flagCounter++; 	//checks because flagCounter isn't needed after first few frames
											//avoids getting huge values after long frame processing
	
	cvReleaseImage(&tmp);
	
	return result;
}

/*********************** getCount **************************
 * Type:		PUBLIC
 * Description: Returns the number of different pixels found
 * 				in a (thresholded) difference between frames.
 * 				According to this value, motion is evaluated
 * 				internally in detectMotion().
 * 				getCount is used for debugging.				   
 * @return 		integer count 
 ***********************************************************/

int MotionDetector::getCount()
{
	return count;
}


/******************* resetFlagCounter **********************
 * Type:		PUBLIC
 * Description: Function called in the ROS node, used to re-
 * 				set the flagCounter value, which causes the
 * 				algorithm to re-wait until the buffer of fra-
 * 				mes is full and results can be trusted again.		   
 * @return 		void 
 ***********************************************************/

void MotionDetector::resetFlagCounter()
{
	this->flagCounter = 0;
}


/********************** getDiffImg **************************
 * Type:		PUBLIC
 * Description: Returns the last image of the buffer of frames
 * 				which holds the difference image. Being called
 * 				in the ROS Node after the agorithm has run,
 * 				retrieves the result image for debugging.		   
 * @return 		integer count 
 ***********************************************************/

IplImage* MotionDetector::getDiffImg(){
	if ( buf!=0 )
		return buf[last];
	else
		return 0;
}
