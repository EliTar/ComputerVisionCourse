#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

using std::string;
using std::endl;
using std::cout;
using std::vector;

using cv::VideoCapture;
using cv::Mat;
using cv::Ptr;
using cv::BackgroundSubtractor;
using cv::findContours;

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

        imshow("Frame", frame);
        imshow("FG Mask MOG 2", fgMaskMOG2);
        cv::waitKey(0);
    }

    return 0;
}