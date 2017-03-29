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
using cv::Vec3b;

const int patchRadius = 2;

vector<vector<Rect>> RandomPatchesForEachLabel(Mat image, Mat imageLabels)
{
    // Defining some usful constants,
    // finding the minimum and maximum values of the labeling.

    const int   imageRows = image.rows,
                imageCols = image.cols;
    double      maxLabel,
                minLabel;

    minMaxLoc(imageLabels, &minLabel, &maxLabel);


    ////
    // Step 2: Choose several random pixels for each fregment
    ////

    // First, we calculate how many pixels each label has.
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
            pixelsLabeled[imageLabels.at<int>(i , j)].push_back(Point(j, i));
            // The points coordinates are reversed to maintain our axis.
        }
    }

    // cout << minLabel << " " << maxLabel << endl;
    // cout << pixelsLabeled.size() << endl;

    // Choose sqrt(pointsLabeldWith(i)) points randomally

    std::random_device rd;
    std::mt19937 gen(rd());
    vector<vector<Point>> patchRepresentatives(maxLabel + 1);

    for(int i = minLabel; i < maxLabel + 1; i++)
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
    // Step 3: Determine patches, search for closets patch in the example image.
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

    int sideLength = 2 * patchRadius + 1;
    vector<vector<Rect>> patches(maxLabel + 1);

    for(int i = minLabel; i < maxLabel + 1; i++)
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

    return patches;
}

Mat VisualizePatches(Mat image, vector<vector<Rect>> patches)
{
    int sideLength = 2 * patchRadius + 1;
    Mat vizPatch = Mat::zeros(cv::Size(image.cols, image.rows), CV_8UC3);

    cout << "hello" << endl;

    for(int i = 0; i < patches.size(); i++)
    {
        for(Rect rec : patches[i])
        {
            for(int i = rec.x; i < rec.x + sideLength; i++)
            {
                for(int j = rec.y; j < rec.y + sideLength; j++)
                {
                    vizPatch.at<cv::Vec3b>(j,i) = image.at<cv::Vec3b>(j,i);
                }
            }
        }
    }

    return vizPatch;
}



int main()
{
    Mat trainImage = cv::imread("../images/liberty_train.jpg");
    Mat trainLabels = cv::imread("../images/liberty_train_labels.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat testImage = cv::imread("../images/liberty_test.jpg");

    trainLabels.convertTo(trainLabels, CV_32S);

    SLIC slic;
    int estSuperpixelsNum = 400;

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

    ////
    // Step 2 + 3 combined, see RandomPatchesForEachLabel
    ////

    vector<vector<Rect>> testPatches = RandomPatchesForEachLabel(testImage, labeledImage);
    Mat vizTestPatch = VisualizePatches(testImage, testPatches);

    // Patches for the training image
    vector<vector<Rect>> trainPatches = RandomPatchesForEachLabel(trainImage, trainLabels);
    Mat vizTrainPatch = VisualizePatches(testImage, trainPatches);


    ////
    // Step 3
    ////

    // Calulation of color difference for patches.
    // We use the CIE76 method to do this (CIE 1976)

    vector<vector<double>> distancePerPixel(testPatches.size());
    Mat trainImageLab;
    Mat testImageLab;

    cvtColor(trainImage, trainImageLab, CV_BGR2Lab);
    cvtColor(testImage, testImageLab, CV_BGR2Lab);

    // for(int i = 0; i < testPatches.size(); i++)
    // {
    //     for(Rect testSquare : testPatches[i])
    //     {
    //         for(int j = 0; j < trainPatches.size(); j++)
    //         {
    //             double smallestDistance = DBL_MAX;
    //             double currentDistance;
    //             for(Rect trainSquare : trainPatches[j])
    //             {
    //                 trainImageLab.at<Vec3b>()



    //                  currentDistance = CalcPixelDistance(testImage, testSquare, trainImage, trainSquare);
    //                  if(currentDistance < smallestDistance)
    //                  {
    //                     smallestDistance = currentDistance;
    //                  }
    //             }
    //             distancePerPixel[i].push_back(smallestDistance);
    //         }
    //     }
    // }

    // vector<double> fragmentDistacne(testPatches.size());
    // for(i = 0; i < testPatches.size(); i++)
    // {
    //     // Choose the median value of all minimum distances
    //     fragmentDistacne[i] =   std::nth_element(distancePerPixel[i].begin(),
    //                             distancePerPixel[i].begin() + distancePerPixel[i].size() / 2,
    //                             distancePerPixel[i].end());
    // }

    // Normalize

    imshow("ww", trainImage);
    imshow("w", vizTrainPatch);
    cv::waitKey(0);

    //imshow("w", vizTrainPatch);
    // Step 1 : computing input image fragments

    return 1;
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




// // Finding the minimum and maximum values of the labeling. minimum is 0

//     double  maxLabel,
//             minLabel;

//     minMaxLoc(labeledImage, &minLabel, &maxLabel);

//     ////
//     // Step 2: Choose several random pixels for each fregment
//     ////

//     // First, we calculate how many pixels have each label.
//     // We do this to know how many repesentatives we should pick for each label.

//     // axis shceme for the image:
//     // -y
//     // ^
//     // |
//     // |
//     // |
//     // ------> x

//     vector<vector<Point>> pixelsLabeled(maxLabel + 1);

//     for(int i = 0; i < imageRows; i++)
//     {
//         for(int j = 0; j < imageCols; j++)
//         {
//             // cout << i << " " << j <<  " " << labeledImage.at<int>(i , j) << endl;
//             pixelsLabeled[labeledImage.at<int>(i , j)].push_back(Point(j, i));
//             // The points coordinates are reversed to maintain our axis.
//         }
//     }

//     cout << minLabel << " " << maxLabel << endl;
//     cout << pixelsLabeled.size() << endl;

//     // Choose sqrt(pointsLabeldWith(i)) points randomally

//     std::random_device rd;
//     std::mt19937 gen(rd());
//     vector<vector<Point>> patchRepresentatives(maxLabel + 1);

//     for(int i = 0; i < maxLabel + 1; i++)
//     {
//         int currentSize = pixelsLabeled[i].size();
//         int toChoose = round(sqrt(currentSize)) / 2;
//         std::uniform_int_distribution<> dis(0, currentSize - 1);

//         for(int j = 0; j < toChoose; j++)
//         {
//             patchRepresentatives[i].push_back( pixelsLabeled[i][dis(gen)] );
//         }
//     }

//     ////
//     // Step 3: Determine patches, search for closets patch in the example image.
//     ////

//     // We have to maintain the same size for all patches,
//     // therefore, if one patch exceeds the image limits,
//     // we will just ignore it.

//     // Rect is defined by (x, y) width and height
//     // (x,y) ------
//     // |
//     // |
//     // |
//     // |
//     // |

//     int patchRadius = 2;

//     int sideLength = 2 * patchRadius + 1;
//     vector<vector<Rect>> patches(maxLabel + 1);

//     for(int i = 0; i < maxLabel + 1; i++)
//     {
//         for(Point p : patchRepresentatives[i])
//         {
//             if( p.x - patchRadius >= 0
//             &&  p.y - patchRadius >= 0
//             &&  p.x + patchRadius < imageCols
//             &&  p.y + patchRadius < imageRows)
//             {
//                 patches[i].push_back(Rect(p.x - patchRadius, p.y - patchRadius, sideLength, sideLength));
//             }
//         }
//     }