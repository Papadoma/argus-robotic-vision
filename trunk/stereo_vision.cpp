#include "cv.h" 
#include "highgui.h" 
#include <stdio.h>  
// A Simple Camera Capture Framework 
int main() {
  CvCapture* capture_left = cvCaptureFromCAM( 0 );
  if ( !capture_left ) {
    fprintf( stderr, "ERROR: could not initialize camera 0 \n" );
    getchar();
    return -1;
  }
  CvCapture* capture_right = cvCaptureFromCAM( 1 );
  if ( !capture_right ) {
    fprintf( stderr, "ERROR: could not initialize camera 1 \n" );
    getchar();
    return -1;
  }
  
  int status1=cvGetCaptureProperty(capture_left, CV_CAP_PROP_FPS);
  int status2=cvGetCaptureProperty(capture_right, CV_CAP_PROP_FPS);
  printf("fps cam0: %d cam1: %d\n",status1,status2);
  
  
  // Create a window in which the captured images will be presented
  cvNamedWindow( "camera_left", CV_WINDOW_AUTOSIZE );
  cvNamedWindow( "camera_right", CV_WINDOW_AUTOSIZE );
  // Show the image captured from the camera in the window and repeat
  IplImage* frame_left;
  IplImage* frame_right;
  int i=0;
  while ( 1 ) {
    // Get one frame
    printf("iteration %d\n",i++);
    printf("cvGrabFrame left\n");
    cvGrabFrame( capture_left );
    printf("cvGrabFrame right\n");
    cvGrabFrame( capture_right );
    printf("cvRetrieveFrame left\n");
    IplImage* frame_left = cvRetrieveFrame( capture_left );
    printf("cvRetrieveFrame right\n");
    IplImage* frame_right = cvRetrieveFrame( capture_right );
    
    
   /* printf("%d\n",status);
   */ 
    if ( !(frame_left||frame_right)) {
      fprintf( stderr, "ERROR: frame_left is null...\n" );
      getchar();
      break;
    }

    printf("cvShowImage\n");
    cvShowImage( "camera_left", frame_left );
    cvShowImage( "camera_right", frame_right );
    
    // Do not release the frame!
    //If ESC key pressed, Key=0x10001B under OpenCV 0.9.7(linux version),
    //remove higher bits using AND operator
    printf("cvWaitKey\n");
    if ( (cvWaitKey(10) & 255) == 27 ) break;
  }
  // Release the capture device housekeeping
  cvReleaseCapture( &capture_right );
  cvReleaseCapture( &capture_left );
  cvDestroyWindow( "camera_left" );
  cvDestroyWindow( "camera_right" );
  return 0;
}

