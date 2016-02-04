#include <iostream>
#include <stdio.h>

 #include "opencv2/objdetect/objdetect.hpp"
 #include "opencv2/highgui/highgui.hpp"
 #include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

 /** Function Headers */
 Mat detectAndDisplay( Mat frame, string str );

 /** Global variables */
 String face_cascade_name = "haarcascade_frontalface_alt.xml";
 String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
 CascadeClassifier face_cascade;
 CascadeClassifier eyes_cascade;

void help()
{
	printf("H");
}

int main(int, char**)
{
	help();
	
	if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	
	
	const string str="test.avi";
    VideoCapture cap(str); // open the default camera with argument 0. If we want to open a file, just replace the 0 by str and adjust str.
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    Mat edges;
    namedWindow("normal",WINDOW_NORMAL);

	VideoWriter outputVideo;

	const string NAME = "testcopy.avi";
	int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC)); Uncomment if using file instead of cam
	
	// Acquire input size
       
    double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
    double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video
    Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));

	outputVideo.open(NAME, ex /*CV_FOURCC('P','I','M','1') FOR WEBCAM PURPOSE */ /* Use ex if I am using a file instead of webcam*/, cap.get(CV_CAP_PROP_FPS), frameSize, true);
	Mat frame;
    for(;;)
    {
        cap >> frame; // get a new frame from camera
        outputVideo.write(detectAndDisplay(frame, "normal"));
        if(waitKey(30) >= 0) break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}

Mat detectAndDisplay( Mat frame, string str )
{
  std::vector<Rect> faces;
  Mat frame_gray;

  cvtColor( frame, frame_gray, CV_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );

  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

  for( size_t i = 0; i < faces.size(); i++ )
  {
    Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
    ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

    Mat faceROI = frame_gray( faces[i] );
    std::vector<Rect> eyes;

    //-- In each face, detect eyes
    eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

    for( size_t j = 0; j < eyes.size(); j++ )
     {
       Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
       int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
       circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
     }
  }
  //-- Show what you got
  imshow( str, frame );
  return frame;
 }
