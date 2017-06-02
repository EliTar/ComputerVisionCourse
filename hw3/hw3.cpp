#include <iostream>
#include <string>
#include <vector>
#include <numeric>

#include <opencv2/opencv.hpp>

using std::string;
using std::endl;
using std::cout;
using std::vector;

using cv::VideoCapture;
using cv::Point;
using cv::Scalar;
using cv::Size;
using cv::RNG;
using cv::Mat;
using cv::Ptr;
using cv::BackgroundSubtractor;
using cv::KalmanFilter;

Point meanOfRegion(vector<Point> vec)
{
    Point sum = std::accumulate( vec.begin(), vec.end(), Point(0, 0));
    return Point(sum.x / vec.size(), sum.y / vec.size());
}

int main()
{
    // Run the command:
    // https://ask.fedoraproject.org/en/question/9111/sticky-what-plugins-do-i-need-to-install-to-watch-movies-and-listen-to-music/
    // sudo dnf install gstreamer{1,}-{ffmpeg,libav,plugins-{good,ugly,bad{,-free,-nonfree}}} --setopt=strict=0
    VideoCapture vid("../videos/bugs12.mp4");

    Mat frame, fgMaskMOG2;

    Ptr<BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
    pMOG2 = cv::createBackgroundSubtractorMOG2(500, 50, false);

    while(vid.grab())
    {
        vid.retrieve(frame);
        pMOG2->apply(frame, fgMaskMOG2);

        Mat element = getStructuringElement(cv::MORPH_RECT, Size(3, 3));
        
        cv::dilate(fgMaskMOG2, fgMaskMOG2, Mat(), Point(-1,-1), 1);
		morphologyEx(fgMaskMOG2, fgMaskMOG2, cv::MORPH_CLOSE, element);
        cv::erode(fgMaskMOG2, fgMaskMOG2, Mat(), Point(-1,-1), 1);
        
        RNG     rng(12345);
        double  contSum = 0,
                contAvgSize;
        Mat     temp = fgMaskMOG2.clone(),
                drawing = Mat::zeros(frame.rows, frame.cols, CV_8UC3);

        vector<vector<Point> > contours; 
        vector<vector<Point> > chosenContours;
        vector<Point> blobLocation;

        cv::findContours(temp, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

        for(int i = 0; i < contours.size(); i++)
			contSum += cv::contourArea(contours[i]);

		double contAvg = contSum / contours.size();

        for(int i = 0; i < contours.size(); i++)
		{
            // Making sure a color won't be black, but still nice and colorfull
			Scalar color = Scalar(rng.uniform(50, 255), rng.uniform(50, 255), rng.uniform(50, 255));

            if(cv::contourArea(contours[i]) > contAvg / 4)
            {
                chosenContours.push_back(contours[i]);
			    cv::drawContours(drawing, contours, i, color, CV_FILLED);
            }
		}

        for(int i = 0; i < chosenContours.size(); i++)
        {
            blobLocation.push_back( meanOfRegion(chosenContours[i]) );
            circle(drawing, meanOfRegion(chosenContours[i]), 3, Scalar(0, 0, 255), CV_FILLED);
        }

        imshow("Frame", frame);
        imshow("Contours", drawing);
        imshow("FG Mask MOG 2", fgMaskMOG2);
        cv::waitKey(0);
    }

    return 0;
}