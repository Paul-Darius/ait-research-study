#include <iostream>
#include <stdio.h>

 #include "opencv2/objdetect/objdetect.hpp"
 #include "opencv2/highgui/highgui.hpp"
 #include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

double min_face_size=10;
double max_face_size=300;

 /** Function Headers */
 Mat detectAndDisplay( Mat frame, string str, int frame_number);
 string NumberToString ( int Number );

 /** Global variables */
 String face_cascade_name = "haarcascade_frontalface_alt.xml";
 String profile_cascade_name = "haarcascade_profileface.xml";
 
 CascadeClassifier face_cascade;
 CascadeClassifier profile_cascade;

void help()
{
	printf("This software takes the video contained in the directory MBK_Videos, detect all the faces there and save it in the directory Result.");
}

int main(int, char**)
{
	help();
	
	if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	if( !profile_cascade.load( profile_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };	
	
	const string str="test.avi";
    VideoCapture cap(str); // open the default camera with argument 0. If we want to open a file, just replace the 0 by str and adjust str.
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    Mat edges;
    namedWindow("Result",WINDOW_NORMAL);
	namedWindow("Initial",WINDOW_NORMAL);
	VideoWriter outputVideo;

	const string NAME = "Result/testcopy.avi";
	int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC)); //Uncomment if using file instead of cam
	
	// Acquire input size
       
    double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
    double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video
    Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));

	outputVideo.open(NAME, ex /*CV_FOURCC('P','I','M','1') FOR WEBCAM PURPOSE */ /* Use ex if I am using a file instead of webcam*/, cap.get(CV_CAP_PROP_FPS), frameSize, true);
	Mat frame;
	int frame_number = 1;
    while(1)
    {
        cap >> frame; // get a new frame from camera
        imshow("Initial", frame);
        outputVideo.write(detectAndDisplay(frame, "Result", frame_number));
        if(waitKey(10) >= 0) break;
        frame_number++;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}

Mat detectAndDisplay( Mat frame, string str, int frame_number)
{
  std::vector<Rect> faces;
  std::vector<Rect> faces_profile;
  Mat frame_gray;
  Mat Rect_ROI;
  cvtColor( frame, frame_gray, CV_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );

  //-- Detect faces
   face_cascade.detectMultiScale( frame, faces, 1.2, 2, 0|CV_HAAR_SCALE_IMAGE, Size(min_face_size, min_face_size),Size(max_face_size, max_face_size) );
   profile_cascade.detectMultiScale( frame, faces_profile, 1.2, 2, 0|CV_HAAR_SCALE_IMAGE, Size(min_face_size, min_face_size),Size(max_face_size, max_face_size) );
 // Draw circles on the detected faces
 // First for front faces
  
  for( int i = 0; i < faces.size(); i++ )
	{
		Rect_ROI = Mat(frame,faces[i]);
		string str = "Result/result"+NumberToString(frame_number)+NumberToString(i)+".jpg";
		imwrite(str,Rect_ROI);
		rectangle(frame, faces[i],Scalar( 255, 0, 255 ),3,8,0);
	}

   for( int i = 0; i < faces_profile.size(); i++ )
	{
		Rect_ROI = Mat(frame,faces[i]);
		string str = "Result/resultProfile"+NumberToString(frame_number)+NumberToString(i)+".jpg";
		imwrite(str,Rect_ROI);
		rectangle(frame, faces_profile[i],Scalar( 255, 0, 255 ),3,8,0);
	}
  //-- Show what you got
  imshow( str, frame );
  return frame;
 }
 
 string NumberToString ( int Number )
{
	stringstream ss;
	ss << Number;
	return ss.str();
}
