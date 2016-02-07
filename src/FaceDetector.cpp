#include <iostream>
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

/** Constants used for detectAndDisplay */ 

double min_face_size=10;
double max_face_size=300;

 /** Function Headers */
 Mat detectAndDisplay( Mat frame, string window_name, int frame_number, string video_string);
 string NumberToString ( int Number );

 /** Global variables */
 String face_cascade_name = "src/others/haarcascade_frontalface_alt.xml";
 String profile_cascade_name = "src/others/haarcascade_profileface.xml";
 
 CascadeClassifier face_cascade;
 CascadeClassifier profile_cascade;

void help()
{
	cout << "This software is supposed to be use indirectly, within the script generate_database.sh. Please read the readme file." << endl;
}

int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		help();
		return -1;
	}
	string video_string_with_slashes = argv[1];
	string video_string=video_string_with_slashes;
	for (int i=0; i<video_string.size();i++)
	{
		if (video_string[i] == '/' || video_string[i] == ' ')
			video_string[i]='-';
	}

	if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	if( !profile_cascade.load( profile_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };	

    VideoCapture cap(video_string_with_slashes);
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    Mat edges;
    namedWindow("Result",WINDOW_NORMAL);
	namedWindow("Initial",WINDOW_NORMAL);
	VideoWriter outputVideo;
	
	string command1 = "mkdir Database/"+video_string+";";
	string command2 = "mkdir Database/"+video_string+"/video;";
	system(command1.c_str());
	system(command2.c_str());


	const string VIDEO_NAME = "Database/"+video_string+"/video/"+"detected.avi";
	
	
	int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));
	       
    double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
    double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video
    Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));

	outputVideo.open(VIDEO_NAME, ex /*CV_FOURCC('P','I','M','1') FOR WEBCAM PURPOSE */ /* Use ex if I am using a file instead of webcam*/, cap.get(CV_CAP_PROP_FPS), frameSize, true);
	Mat frame;
	int frame_number = 1;
    while(1)
    {
        cap >> frame; // get a new frame from camera
        imshow("Initial", frame);
        outputVideo.write(detectAndDisplay(frame, "Result", frame_number, video_string));
        if(waitKey(10) >= 0) break;
        frame_number++;
    }
    cout << "Done" << endl;
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}

Mat detectAndDisplay( Mat frame, string window_name, int frame_number, string video_string)
{
  std::vector<Rect> faces;
  std::vector<Rect> faces_profile;
  Mat frame_gray;
  Mat Rect_ROI;
  cvtColor( frame, frame_gray, CV_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );
  string filename;
  //-- Detect faces
   face_cascade.detectMultiScale( frame, faces, 1.2, 2, 0|CV_HAAR_SCALE_IMAGE, Size(min_face_size, min_face_size),Size(max_face_size, max_face_size) );
   profile_cascade.detectMultiScale( frame, faces_profile, 1.2, 2, 0|CV_HAAR_SCALE_IMAGE, Size(min_face_size, min_face_size),Size(max_face_size, max_face_size) );

  for( int i = 0; i < faces.size(); i++ )
	{
		Rect_ROI = Mat(frame,faces[i]);
		filename = "Database/"+video_string+"/"+"Frame"+NumberToString(frame_number)+"Face"+NumberToString(i)+".jpg";
		imwrite(filename,Rect_ROI);
		rectangle(frame, faces[i],Scalar( 255, 0, 255 ),3,8,0);
	}

   for( int i = 0; i < faces_profile.size(); i++ )
	{
		Rect_ROI = Mat(frame,faces[i]);
		filename = "Database/"+video_string+"/"+"|Profile|Frame"+NumberToString(frame_number)+"Face"+NumberToString(i)+".jpg";
		imwrite(filename,Rect_ROI);
		rectangle(frame, faces_profile[i],Scalar( 255, 0, 255 ),3,8,0);
	}
  //-- Show what you got
  imshow( window_name, frame );
  return frame;
 }
 
 string NumberToString ( int Number )
{
	stringstream ss;
	ss << Number;
	return ss.str();
}
