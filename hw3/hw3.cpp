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
    VideoCapture vid("../videos/bug00.mp4");

    Mat frame, fgMaskMOG2;

    Ptr<BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
    pMOG2 = cv::createBackgroundSubtractorMOG2(500, 50, false);

    int stateSize = 6;
    int measSize = 4;
    int contrSize = 0;
    unsigned int type = CV_32F;

    cv::KalmanFilter kf(stateSize, measSize, contrSize, type);
    cv::Mat state(stateSize, 1, type);  // [x,y,v_x,v_y,w,h]
    cv::Mat meas(measSize, 1, type);    // [z_x,z_y,z_w,z_h]

    // Transition State Matrix A
    // Note: set dT at each processing step!
    // [ 1 0 dT 0  0 0 ]
    // [ 0 1 0  dT 0 0 ]
    // [ 0 0 1  0  0 0 ]
    // [ 0 0 0  1  0 0 ]
    // [ 0 0 0  0  1 0 ]
    // [ 0 0 0  0  0 1 ]
    cv::setIdentity(kf.transitionMatrix);

    // Measure Matrix H
    // [ 1 0 0 0 0 0 ]
    // [ 0 1 0 0 0 0 ]
    // [ 0 0 0 0 1 0 ]
    // [ 0 0 0 0 0 1 ]
    kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
    kf.measurementMatrix.at<float>(0) = 1.0f;
    kf.measurementMatrix.at<float>(7) = 1.0f;
    kf.measurementMatrix.at<float>(16) = 1.0f;
    kf.measurementMatrix.at<float>(23) = 1.0f;

    // Process Noise Covariance Matrix Q
    // [ Ex   0   0     0     0    0  ]
    // [ 0    Ey  0     0     0    0  ]
    // [ 0    0   Ev_x  0     0    0  ]
    // [ 0    0   0     Ev_y  0    0  ]
    // [ 0    0   0     0     Ew   0  ]
    // [ 0    0   0     0     0    Eh ]
    kf.processNoiseCov.at<float>(0) = 1e-2;
    kf.processNoiseCov.at<float>(7) = 1e-2;
    kf.processNoiseCov.at<float>(14) = 5.0f;
    kf.processNoiseCov.at<float>(21) = 5.0f;
    kf.processNoiseCov.at<float>(28) = 1e-2;
    kf.processNoiseCov.at<float>(35) = 1e-2;

    // Measures Noise Covariance Matrix R
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));
    // <<<< Kalman Filter

    double ticks = 0;
    bool found = false;

    int notFoundCount = 0;

    vector<Point> kfPoints;

    // Grab one frame for BG Substruction to start working
    vid.grab();
    vid.retrieve(frame);
    pMOG2->apply(frame, fgMaskMOG2);

    while(vid.grab())
    {
        double precTick = ticks;
        ticks = (double) cv::getTickCount();

        double dT = (ticks - precTick) / cv::getTickFrequency(); //seconds
        dT = 0.03;

        vid.retrieve(frame);
        Mat res = frame.clone();
        pMOG2->apply(frame, fgMaskMOG2);

        if (found)
        {
            // >>>> Matrix A
            kf.transitionMatrix.at<float>(2) = dT;
            kf.transitionMatrix.at<float>(9) = dT;
            // <<<< Matrix A

            cout << "dT:" << endl << dT << endl;

            state = kf.predict();
            cout << "State post:" << endl << state << endl;

            cv::Rect predRect;
            predRect.width = state.at<float>(4);
            predRect.height = state.at<float>(5);
            predRect.x = state.at<float>(0) - predRect.width / 2;
            predRect.y = state.at<float>(1) - predRect.height / 2;

            cv::Point center;
            center.x = state.at<float>(0);
            center.y = state.at<float>(1);
            cv::circle(res, center, 2, CV_RGB(255,0,0), -1);

            cv::rectangle(res, predRect, CV_RGB(255,0,0), 2);

            kfPoints.push_back(center);
        }

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

        cv::Rect bugRect = cv::boundingRect(chosenContours[0]);

        cv::drawContours(res, chosenContours, 0, CV_RGB(20,150,20), 1);
        cv::rectangle(res, bugRect, CV_RGB(0,255,0), 2);

        // cv::Point center;
        // center.x = bugRect.x + bugRect.width / 2;
        // center.y = bugRect.y + bugRect.height / 2;
        cv::circle(res, blobLocation[0], 2, CV_RGB(20,150,20), -1);

        std::stringstream sstr;
        sstr << "(" << blobLocation[0].x << "," << blobLocation[0].y << ")";
        cv::putText(res, sstr.str(),
                    cv::Point(blobLocation[0].x + 3, blobLocation[0].y - 3),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(20,150,20), 2);

        // <<<<< Detection result

        // >>>>> Kalman Update
        if (blobLocation.size() == 0)
        {
            notFoundCount++;
            cout << "notFoundCount:" << notFoundCount << endl;
            if( notFoundCount >= 100 )
            {
                found = false;
            }
            /*else
                kf.statePost = state;*/
        }
        else
        {
            notFoundCount = 0;

            meas.at<float>(0) = blobLocation[0].x;
            meas.at<float>(1) = blobLocation[0].y;
            meas.at<float>(2) = (float)bugRect.width;
            meas.at<float>(3) = (float)bugRect.height;

            if (!found) // First detection!
            {
                // >>>> Initialization
                kf.errorCovPre.at<float>(0) = 1; // px
                kf.errorCovPre.at<float>(7) = 1; // px
                kf.errorCovPre.at<float>(14) = 1;
                kf.errorCovPre.at<float>(21) = 1;
                kf.errorCovPre.at<float>(28) = 1; // px
                kf.errorCovPre.at<float>(35) = 1; // px

                state.at<float>(0) = meas.at<float>(0);
                state.at<float>(1) = meas.at<float>(1);
                state.at<float>(2) = 0;
                state.at<float>(3) = 0;
                state.at<float>(4) = meas.at<float>(2);
                state.at<float>(5) = meas.at<float>(3);
                // <<<< Initialization

                kf.statePost = state;
                
                found = true;
            }
            else
                kf.correct(meas); // Kalman Correction

            cout << "Measure matrix:" << endl << meas << endl;
        }

        for(Point p : kfPoints)
        {
            circle(frame, p, 3, Scalar(0, 0, 255), CV_FILLED);
        }

        imshow("Frame", frame);
        imshow("Test", res);
        imshow("Contours", drawing);
        imshow("FG Mask MOG 2", fgMaskMOG2);
        cv::waitKey(0);
    }

    return 0;
}