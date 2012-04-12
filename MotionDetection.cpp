/*#############################################################
 * MotionDetection.cpp
 *
 * File Description :
 *	ROS Node of motion detection, vision package 
 *	
 * Contents :	completeTransition()
 * 				getGeneralParams()
 * 				getMotionParams()
 * 				getTimerParams()
 * 				motionCallback()
 * 				imageCallback()
 * 				spin()
 * 				startTransition()
 *
 * Author : Aprilis George
 *
 * Date :	24-10-2011
 *
 * Change History :
 *
 *#############################################################
 */
 
 #include "MotionDetection.h"
 
 using namespace std;
 
 //constructor
MotionDetection::MotionDetection() :	_nh(),
										testMotion("victimDirection" , "testMotion")
{
	 //initialize motion detector 
	 _motionDetector		=	new MotionDetector();
	 
	 //Get Motion Detector Parameters
	getMotionParams();
	
	// Get General Parameters, such as frame width & height , camera id
	getGeneralParams();
	
	//memory will be allocated in the imageCallback
	motionFrame = 0;
	
	extraFrame = cvCreateImage( cvSize(frameWidth,frameHeight), IPL_DEPTH_8U, 3 );
	
	//Declare publisher and advertise topic where algorithm results are posted
	_victimDirectionPublisher = _nh.advertise<vision_communications::victimIdentificationDirectionMsg>("victimDirection", 10);
	//Declare motion service Server
	_motionServer = _nh.advertiseService("motionArmService", &MotionDetection::motionArmSrvCallback, this);
	
	//Advertise topics for debugging if we are in debug mode
	if (debugMotion)
	{
		_motionDiffPublisher = image_transport::ImageTransport(_nh).advertise("motionDiff", 1);
		_motionFrmPublisher = image_transport::ImageTransport(_nh).advertise("motionFrm", 1);
	}
	
	//subscribe to input image's topic
	//image_transport::ImageTransport it(_nh);
	std::string transport = "theora";
	_frameSubscriber = image_transport::ImageTransport(_nh).subscribe(imageTopic , 1, &MotionDetection::imageCallback, this );
	
	//initialize states - robot starts in STATE_OFF 
	curState = STATE_OFF;
	prevState = STATE_OFF;
	
	//Initialize mutex lock
	motionLock = PTHREAD_MUTEX_INITIALIZER;
	
	//initialize flag used to sync the callbacks
	isMotionFrameUpdated = false;
	
	//initialize state Managing Variables
	motionNowON 	= false;
	
	ROS_INFO("[MotionNode] : Created Motion Detection instance");
}


MotionDetection::~MotionDetection()
{
	ROS_INFO("[MotionNode] : Destroying Motion Detection instance");
	
	pthread_mutex_destroy(&motionLock);
	
	delete _motionDetector;
	cvReleaseImage( &extraFrame);
	
}

//***************************************//
//         Get the timer parameters      //
//***************************************//
void MotionDetection::getTimerParams()
{
	// Get the MotionTime parameter if available
	if (_nh.hasParam("motionTime")) {
		_nh.getParam("motionTime", motionTime);
		ROS_INFO_STREAM("motionTime : " << motionTime);
	}
	else {
		ROS_ERROR("[MotionNode] : Parameter motionTime not found. Using Default");
		motionTime = 0.05;
	}
}


//**************************************//
// 	Get parameters referring to view	// 
//		and frame characteristics		//
//**************************************//
void MotionDetection::getGeneralParams()
{
	// Get the motionDummy parameter if available;
	if (_nh.hasParam("motionDummy")) {
		_nh.getParam("motionDummy", motionDummy);
		ROS_INFO("motionDummy: %d", motionDummy);
	}
	else {
		ROS_ERROR("[MotionNode] : Parameter motionDummy not found. Using Default");
		motionDummy = false;
	}
	
	// Get the debugMotion parameter if available;
	if (_nh.hasParam("debugMotion")) {
		_nh.getParam("debugMotion", debugMotion);
		ROS_INFO_STREAM("debugMotion : " << debugMotion);
	}
	else {
		ROS_ERROR("[MotionNode] : Parameter debugMotion not found. Using Default");
		debugMotion = true;
	}
	
	// Get the Height parameter if available;
	if (_nh.hasParam("height")) {
		_nh.getParam("height", frameHeight);
		ROS_INFO_STREAM("height : " << frameHeight);
	}
	else {
		ROS_ERROR("[MotionNode] : Parameter frameHeight not found. Using Default");
		frameHeight = DEFAULT_HEIGHT;
	}
	
	// Get the listener's topic;
	if (_nh.hasParam("imageTopic")) {
		_nh.getParam("imageTopic", imageTopic);
		ROS_INFO_STREAM("imageTopic : " << imageTopic);
	}
	else {
		ROS_ERROR("[MotionNode] : Parameter imageTopic not found. Using Default");
		imageTopic = "/vision/image";
	}
	
	// Get the Width parameter if available;
	if (_nh.hasParam("width")) {
		_nh.getParam("width", frameWidth);
		ROS_INFO_STREAM("width : " << frameWidth);
	}
	else {
		ROS_ERROR("[MotionNode] : Parameter frameWidth not found. Using Default");
		frameWidth = DEFAULT_WIDTH;
	}
}

//**************************************//
// 		Get parameters referring to 	// 
//		motion detection algorithm 		//
//**************************************//	
void MotionDetection::getMotionParams()
{
	// Get the buffer size parameter if available;
	if (_nh.hasParam("motionBuffer")) {
		_nh.getParam("motionBuffer", _motionDetector->N);
		ROS_INFO_STREAM("motionBuffer : " << _motionDetector->N);
	}
	else {
		ROS_ERROR("[MotionNode] : Parameter motionBuffer not found. Using Default");
		_motionDetector->N = 4;
	}
	
	// Get the difference threshold parameter if available;
	if (_nh.hasParam("motionDiffThres")) {
		_nh.getParam("motionDiffThres", _motionDetector->diff_threshold);
		ROS_INFO_STREAM("motionDiffThres : " << _motionDetector->diff_threshold);

	}
	else {
		ROS_ERROR("[MotionNode] : Parameter motionDiffThres not found. Using Default");
		_motionDetector->diff_threshold = 45;
	}
	
	// Get the motion high threshold parameter if available;
	if (_nh.hasParam("motionHighThres")) {
		_nh.getParam("motionHighThres", _motionDetector->motion_high_thres);
		ROS_INFO_STREAM("motionHighThres : " << _motionDetector->motion_high_thres);
	}
	else {
		ROS_ERROR("[MotionNode] : Parameter motionHighThres not found. Using Default");
		_motionDetector->motion_high_thres = 7500;
	}
	
	// Get the motion low threshold parameter if available;
	if (_nh.hasParam("motionLowThres")) {
		_nh.getParam("motionLowThres", _motionDetector->motion_low_thres);
		ROS_INFO_STREAM("motionLowThres : " << _motionDetector->motion_low_thres);
	}
	else {
		ROS_ERROR("[MotionNode] : Parameter motionLowThres not found. Using Default");
		_motionDetector->motion_low_thres = 200;
	}
}

//**************************************//
//      The Image callback function     //
//**************************************//
void MotionDetection::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
	int res = -1;
	
	sensor_msgs::CvBridge bridge;
	motionFrame = bridge.imgMsgToCv(msg, "bgr8");
	motionFrameTimestamp = msg->header.stamp;
		
	if ( motionFrame == NULL )			
	{               
		ROS_ERROR("[motionNode] : No more Frames or something went wrong with bag file");
		ros::shutdown();
		return;
	}
	
	//try to lock motion variables
	res = pthread_mutex_trylock(&motionLock);
	
	//if lock was successful: update frame, set boolean and unlock
	if(res == 0){
		memcpy( extraFrame->imageData , motionFrame->imageData , motionFrame->imageSize );
		isMotionFrameUpdated = true;
		pthread_mutex_unlock(&motionLock);
	}
}

//**************************************//
//      The callback function           //
//**************************************// 
void MotionDetection::motionCallback(const ros::TimerEvent&)
{	
	///NOTE: source code of motionCallback moved at motionDetectAndPost
	
	if(!motionNowON){
		return;
	}

	motionDetectAndPost();
}

bool MotionDetection::motionArmSrvCallback(std_srvs::Empty::Request&, std_srvs::Empty::Response&){
	
	_motionDetector->resetFlagCounter();
	
	for(int i=0 ; i<( 4 * _motionDetector->N ); i++)
	{
			motionDetectAndPost();
	}
	
	return true;
}

void MotionDetection::motionDetectAndPost()
{
	
	int retries = 0;
	while (true)
	{
		pthread_mutex_lock(&motionLock);
		if (isMotionFrameUpdated)
			break;
		else
		{
			//if input frame is not yet set
			//sleep for a while to let
			//imageCallback()  to catch up
			pthread_mutex_unlock(&motionLock);
			
			usleep(50 * 1000);
			
			if (retries > 10)
			{
				ROS_INFO("[motionNode] : Timed out waiting for motionFrame!");
				return;
			}
			retries++;
		}
	}
	
	//create message of Motion Detector
	vision_communications::victimIdentificationDirectionMsg motionMessage;
	
	if (motionDummy)
	{
		/*
		 * Motion Dummy Message
		*/
		int temp = 1;
		switch (temp)
		{
			case 0:
				motionMessage.probability = 0;
				break;
			case 1:
				motionMessage.probability = 0.5;
				break;
			case 2:
				motionMessage.probability = 1;
				break;
			default:
				motionMessage.probability = -1;
				ROS_INFO("Unable to get frame for motion detection");
		}
		motionMessage.x = 0;
		motionMessage.y = 0;
		motionMessage.area = 1000;
		motionMessage.header.frame_id="Motion";
		motionMessage.type = vision_communications::victimIdentificationDirectionMsg::MOTION;
		motionMessage.header.stamp = ros::Time::now();
		testMotion.checkoutMsg(motionMessage);
		_victimDirectionPublisher.publish(motionMessage);
		
		//dummy delay
		usleep(1000 * 60);
	}
	else
	{
		/*
		 * Motion Message
		*/
		
		//do detection and examine result cases
		switch (_motionDetector->detectMotion(extraFrame))
		{
			case 0:
				motionMessage.probability = 0;
				break;
			case 1:
				motionMessage.probability = 0.51;
				break;
			case 2:
				motionMessage.probability = 1;
				break;
			default:
				motionMessage.probability = -1;
				ROS_INFO("Unable to get frame for motion detection");
		}
		if(motionMessage.probability > 0.1){
			motionMessage.x = -1;
			motionMessage.y = -1;
			motionMessage.area = _motionDetector->getCount();
			motionMessage.header.frame_id = "headCamera";
			motionMessage.type = vision_communications::victimIdentificationDirectionMsg::MOTION;
			motionMessage.header.stamp = ros::Time::now();
			testMotion.checkoutMsg(motionMessage);
			_victimDirectionPublisher.publish(motionMessage);
			
			//Added for counting the messages sent to dataFusion (for diagnostics)
			motionCounter.request.count = 1;
			if (!motionClient.call(motionCounter))
				ROS_ERROR("Message for motion detection was sent, but could not increment counter");
		}
		
		if (debugMotion){
			cv::WImageBuffer3_b motionDiff;
			motionDiff.SetIpl( (IplImage*)cvClone( _motionDetector->getDiffImg() ) );
			sensor_msgs::ImagePtr msgMotionDiff = sensor_msgs::CvBridge::cvToImgMsg(motionDiff.Ipl() , "mono8");
			_motionDiffPublisher.publish(msgMotionDiff);
			
			cv::WImageBuffer3_b motionFrm;
			motionFrm.SetIpl( (IplImage*)cvClone( extraFrame ) );
			sensor_msgs::ImagePtr msgMotionFrm = sensor_msgs::CvBridge::cvToImgMsg(motionFrm.Ipl() , "bgr8");
			_motionFrmPublisher.publish(msgMotionFrm);			
		}
	}
	
	//reset the flag
	isMotionFrameUpdated = false;	
	//and unlock before leaving
	pthread_mutex_unlock(&motionLock);
}

void MotionDetection::spin()
{
	getTimerParams();
	motionTimer = _nh.createTimer(ros::Duration(motionTime), &MotionDetection::motionCallback , this);

	motionTimer.start();
	ros::spin();
}

//*************************************//
//       State Manager                 //
//*************************************//
void MotionDetection::startTransition(int newState){
	switch(newState){
		case 0:
			curState = STATE_OFF;
			break;
		case 1:
			curState = STATE_EXPLORATION;
			break;
		case 2:
			curState = STATE_IDENTIFICATION;
			break;
		case 3:
			curState = STATE_ARM_APPROACH;
			break;
		case 4:
			curState = STATE_ARM_SCAN;
			break;
		case 5:
			curState = STATE_ARM_SEARCH_COMPLETED;
			break; 
		case 6:
			curState = TELEOPERATION_ANY_STATE;
			break;
		case 7:
			curState = TELEOPERATION_ANY_STATE;
			break;
		case 8:
			curState = TELEOPERATION_ANY_STATE;
			break;
		case 9:
			curState = TELEOPERATION_ANY_STATE;
			break;
		case 10:
			curState = STATE_TERMINATING;
			break;
		default:
			ROS_ERROR("[MotionNode-StateManager] : Wrong state type");
			break;
	}
	
	//check if motion algorithm should be running now
	motionNowON		=	( curState == STATE_ARM_SEARCH_COMPLETED );
	
	//everytime state changes, Motion Detector needs to be reset so that it will
	//discard frames from previous calls in buffer.
	if(motionNowON){
		_motionDetector->resetFlagCounter();
	}
	
		//shutdown if the robot is switched off
	if (curState == STATE_TERMINATING){
		ros::shutdown();
		return;
	}
	
	prevState=curState;

	transitionComplete(newState); //this needs to be called everytime a node finishes transition
}

void MotionDetection::completeTransition(void){
	ROS_INFO("[MotionNode] : Transition Complete");
}


//************************************//
//       Main Function                //
//************************************//
int main(int argc, char** argv)
{	
	ros::init(argc,argv,"MotionNode");
	
	MotionDetection* motionDetection = new MotionDetection();
	
	motionDetection->spin();
	
	delete motionDetection;
		
	return 0;	
}
