#include <iostream>
#include <vector>
#include <random>

#include "slic.h"

#include <opencv2/opencv.hpp>

using std::cout;
using std::endl;

using std::vector;

using cv::imshow;
using cv::imread;
using cv::Mat;
using cv::Scalar;
using cv::Point;
using cv::Rect;

int main()
{
    Mat testImage = cv::imread("../images/liberty_test.jpg");
    SLIC slic;
    int estSuperpixelsNum = 400;
    bool debugMode = true;

    ////
    // Step 1: Compute input image fragments
    ////

    // Find the Super Pixels in the image. The parameter is defined above
    // The GetLabel method gives us 1-dim array with the pixel laybeling.

    slic.GenerateSuperpixels(testImage, estSuperpixelsNum);
    Mat superPixels = slic.GetImgWithContours(Scalar(0,0,255));
    int* label = slic.GetLabel();

    // Translation of the array into a Mat object, the same size as the image
    // only with label number insted of pixel values.

    Mat labeledImage = Mat::zeros(cv::Size(640, 480), CV_32S);
    int imageRows = labeledImage.rows;
    int imageCols = labeledImage.cols;
    for(int i = 0; i < imageRows; i++)
    {
        for(int j = 0; j < imageCols; j++)
        {
            labeledImage.at<int>(i , j) = label[i * imageCols + j];
        }
    }

    // Finding the minimum and maximum values of the labeling. minimum is 0

    double  maxLabel,
            minLabel;

    minMaxLoc(labeledImage, &minLabel, &maxLabel);

    ////
    // Step 2: Choose several random pixels for each fregment
    ////

    // First, we calculate how many pixels have each label.
    // We do this to know how many repesentatives we should pick for each label.

    // axis shceme for the image:
    // -y
    // ^
    // |
    // |
    // |
    // ------> x

    vector<vector<Point>> pixelsLabeled(maxLabel + 1);

    for(int i = 0; i < imageRows; i++)
    {
        for(int j = 0; j < imageCols; j++)
        {
            // cout << i << " " << j <<  " " << labeledImage.at<int>(i , j) << endl;
            pixelsLabeled[labeledImage.at<int>(i , j)].push_back(Point(j, i));
            // The points coordinates are reversed to maintain our axis.
        }
    }

    cout << minLabel << " " << maxLabel << endl;
    cout << pixelsLabeled.size() << endl;

    // Choose sqrt(pointsLabeldWith(i)) points randomally

    std::random_device rd;
    std::mt19937 gen(rd());
    vector<vector<Point>> patchRepresentatives(maxLabel + 1);

    for(int i = 0; i < maxLabel + 1; i++)
    {
        int currentSize = pixelsLabeled[i].size();
        int toChoose = round(sqrt(currentSize)) / 2;
        std::uniform_int_distribution<> dis(0, currentSize - 1);

        for(int j = 0; j < toChoose; j++)
        {
            patchRepresentatives[i].push_back( pixelsLabeled[i][dis(gen)] );
        }
    }

    ////
    // Step 3: Determine patches, search fo closets patch in the example image.
    ////

    // We have to maintain the same size for all patches,
    // therefore, if one patch exceeds the image limits,
    // we will just ignore it.

    // Rect is defined by (x, y) width and height
    // (x,y) ------
    // |
    // |
    // |
    // |
    // |

    int patchRadius = 2;

    int sideLength = 2 * patchRadius + 1;
    vector<vector<Rect>> patches(maxLabel + 1);

    for(int i = 0; i < maxLabel + 1; i++)
    {
        for(Point p : patchRepresentatives[i])
        {
            if( p.x - patchRadius >= 0
            &&  p.y - patchRadius >= 0
            &&  p.x + patchRadius < imageCols
            &&  p.y + patchRadius < imageRows)
            {
                patches[i].push_back(Rect(p.x - patchRadius, p.y - patchRadius, sideLength, sideLength));
            }
        }
    }

    // Visualize the patches

    Mat vizPatch = Mat::zeros(cv::Size(640, 480), CV_8UC3);

    for(int i = 0; i < maxLabel + 1; i++)
    {
        for(Rect rec : patches[i])
        {
            for(int i = rec.x; i < rec.x + sideLength; i++)
            {
                for(int j = rec.y; j < rec.y + sideLength; j++)
                {
                    vizPatch.at<cv::Vec3b>(j,i) = testImage.at<cv::Vec3b>(j,i);
                }
            }
        }
    }

    cout << patches[5].size() << endl;


    imshow("w", vizPatch);
    imshow("ww", superPixels);
    cv::waitKey(0);
    // Step 1 : computing input image fragments

    return 0;
}

//    // Proves it works
//     for(int i = 0; i < testImage.rows; i++)
//     {
//         for(int j = 0; j < testImage.cols; j++)
//         {
//             cout << labled.at<int>(i , j) << endl;
//         }
//     }

//     Mat labled = Mat::zeros(cv::Size(640, 480), CV_8U);
//     for(int i = 0; i < labled.rows; i++)
//     {
//         for(int j = 0; j < labled.cols; j++)
//         {
//             labled.at<uchar>(i , j) = label[i * labled.cols + j];
//         }
//     }