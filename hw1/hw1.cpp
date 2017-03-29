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
const int sideLength = 2 * patchRadius + 1;

Mat PaintInAverageColor(Mat image, Mat imageLabels)
{
    const int   imageRows = image.rows,
                imageCols = image.cols;
    double      maxLabel,
                minLabel;

    minMaxLoc(imageLabels, &minLabel, &maxLabel);
    vector<vector<Vec3b>> pixelValues(maxLabel + 1);
    for(int i = 0; i < imageRows; i++)
    {
        for(int j = 0; j < imageCols; j++)
        {
            pixelValues[imageLabels.at<int>(i , j)].push_back(image.at<Vec3b>(i, j));
        }
    }

    vector<int> sumA(maxLabel + 1, 0);
    vector<int> sumB(maxLabel + 1, 0);
    vector<int> sumC(maxLabel + 1, 0);

    vector<int> howMany(maxLabel + 1, 0);

    for(int i = 0; i < pixelValues.size(); i++)
    {
        for(int j = 0; j < pixelValues[i].size(); j++)
        {
            sumA[i] += pixelValues[i][j][0];
            sumB[i] += pixelValues[i][j][1];
            sumC[i] += pixelValues[i][j][2];
            howMany[i]++;
        }
    }

    vector<Vec3b> avgPixels(maxLabel + 1);
    for(int i = 0; i < pixelValues.size(); i++)
    {
        avgPixels[i][0] = sumA[i] / howMany[i];
        avgPixels[i][1] = sumB[i] / howMany[i];
        avgPixels[i][2] = sumC[i] / howMany[i];
    }

    // vector<Vec3b> pixelAvg(maxLabel + 1);

    // for(int i = 0; i < maxLabel + 1; i++)
    // {
    //     Vec3b sum = std::accumulate(pixelValues[i].begin(), pixelValues[i].end(), 0);
    //     // Vec3b avg = sum / pixelValues[i].size();
    //     // pixelAvg[i] = avg;
    // }

    Mat newImage = Mat::zeros(cv::Size(image.cols, image.rows), CV_8UC3);

    for(int i = 0; i < imageRows; i++)
    {
        for(int j = 0; j < imageCols; j++)
        {
            newImage.at<Vec3b>(i , j) = avgPixels[imageLabels.at<int>(i , j)];
        }
    }

    return newImage;
}

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

    // axis shceme for the image, we choose this because of OpenCV's "at" function.

    // -x
    // ^
    // |
    // |
    // |
    // ------> y

    vector<vector<Point>> pixelsLabeled(maxLabel + 1);

    for(int i = 0; i < imageRows; i++)
    {
        for(int j = 0; j < imageCols; j++)
        {
            // cout << i << " " << j <<  " " << labeledImage.at<int>(i , j) << endl;
            pixelsLabeled[imageLabels.at<int>(i , j)].push_back(Point(i, j));
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
        int toChoose = round(sqrt(currentSize));
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
            &&  p.x + patchRadius < imageRows
            &&  p.y + patchRadius < imageCols)
            {
                patches[i].push_back(Rect(p.x - patchRadius, p.y - patchRadius, sideLength, sideLength));
            }
        }
    }

    return patches;
}

Mat VisualizePatches(Mat image, vector<vector<Rect>> patches)
{
    Mat vizPatch = Mat::zeros(cv::Size(image.cols, image.rows), CV_8UC3);

    for(int i = 0; i < patches.size(); i++)
    {
        for(Rect rec : patches[i])
        {
            for(int i = rec.x; i < rec.x + sideLength; i++)
            {
                for(int j = rec.y; j < rec.y + sideLength; j++)
                {
                    vizPatch.at<cv::Vec3b>(i,j) = image.at<cv::Vec3b>(i,j);
                }
            }
        }
    }

    return vizPatch;
}

double subSquare(uchar a, uchar b)
{
    return (a - b) * (a - b);
}

double Cie76Compare(Vec3b first, Vec3b second)
{
    double differences =    subSquare(first[0], second[0])
                            +  subSquare(first[1], second[1])
                            +  subSquare(first[2], second[2]);
    return sqrt(differences);
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

    cout << "Random patches test: " << endl;

    vector<vector<Rect>> testPatches = RandomPatchesForEachLabel(testImage, labeledImage);
    Mat vizTestPatch = VisualizePatches(testImage, testPatches);

    cout << "Random patches train: " << endl;

    // Patches for the training image
    vector<vector<Rect>> trainPatches = RandomPatchesForEachLabel(trainImage, trainLabels);
    Mat vizTrainPatch = VisualizePatches(testImage, trainPatches);


    ////
    // Step 3
    ////

    // Calulation of color difference for patches.
    // We use the CIE76 method to do this (CIE 1976)

    //vector<vector<double>> distancePerPixel(testPatches.size());
    Mat trainImageLab;
    Mat testImageLab;

    vector<vector<vector<double>>> distancePerPixel(testPatches.size(), vector<vector<double>>(trainPatches.size()));
    // Test patch, Color, the according distances


    cvtColor(trainImage, trainImageLab, CV_BGR2Lab);
    cvtColor(testImage, testImageLab, CV_BGR2Lab);

    cout << "Calculate patch distances: " << endl;

    for(int i = 0; i < testPatches.size(); i++)
    {
        for(Rect testSquare : testPatches[i])
        {
            for(int j = 0; j < trainPatches.size(); j++)
            {
                double smallestDistance = DBL_MAX;
                double currentDistance;
                double smallestColorDistance = DBL_MAX;

                for(Rect trainSquare : trainPatches[j])
                {
                    double currenPatchDistance = 0;
                    for(int x = 0; x < sideLength; x++)
                    {
                        for(int y = 0; y < sideLength; y++)
                        {

                            Vec3b trainPixel = trainImageLab.at<Vec3b>(trainSquare.x + x, trainSquare.y + y);
                            Vec3b testPixel = testImageLab.at<Vec3b>(testSquare.x + x, testSquare.y + y);
                            currenPatchDistance += Cie76Compare(trainPixel, testPixel);
                        }
                    }
                    if(currenPatchDistance < smallestColorDistance)
                    {
                        smallestColorDistance = currenPatchDistance;
                    }
                }
                distancePerPixel[i][j].push_back(smallestColorDistance);
            }
        }
    }

    cout << "Calculate fregment distances: " << endl;

    vector<vector<double>> fragmentDistance(testPatches.size(), vector<double>(trainPatches.size()));
    for(int i = 0; i < testPatches.size(); i++)
    {
        for(int j = 0; j < trainPatches.size(); j++)
        {
            // Choose the median value of all minimum distances
            std::nth_element(   distancePerPixel[i][j].begin(),
                                distancePerPixel[i][j].begin() + distancePerPixel[i][j].size() / 2,
                                distancePerPixel[i][j].end());

            fragmentDistance[i][j] = distancePerPixel[i][j][distancePerPixel[i].size() / 2];
        }
    }


    // Normalize
    double  maxVal = -1,
            minVal = DBL_MAX;

    for(int i = 0; i < fragmentDistance.size(); i++)
    {
        for(int j = 0; j < trainPatches.size(); j++)
        {
            if(fragmentDistance[i][j] > maxVal)
            {
                maxVal = fragmentDistance[i][j];
            }
            if(fragmentDistance[i][j] < minVal)
            {
                minVal = fragmentDistance[i][j];
            }
        }
    }
    vector<vector<double>> normalizedFregmentColorDistance(testPatches.size(), vector<double>(trainPatches.size()));

    for(int i = 0; i < fragmentDistance.size(); i++)
    {
        for(int j = 0; j < trainPatches.size(); j++)
        {
            normalizedFregmentColorDistance[i][j] =
                        (fragmentDistance[i][j] - minVal) /
                        (maxVal - minVal);
        }
    }

    // for(int i = 0; i < fragmentDistance.size(); i++)
    // {
    //     for(int j = 0; j < trainPatches.size(); j++)
    //     {
    //         cout << normalizedFregmentColorDistance[i][j] << " ";
    //     }
    //     cout << endl;
    // }
    // cout << maxVal << " " << minVal << endl;


    ////
    // Step 4: we now have the mapping! huryy!!!!
    ////

    Mat avgColoredImage = PaintInAverageColor(testImage, labeledImage);

    Mat grabCutMask = Mat::zeros(cv::Size(avgColoredImage.cols, avgColoredImage.rows), CV_8U);

    int t = 0.01;

    for(int i = 0; i < avgColoredImage.rows; i++)
    {
        for(int j = 0; j < avgColoredImage.cols; j++)
        {
            double currentLable = labeledImage.at<int>(i, j);
            double currentCut = normalizedFregmentColorDistance[currentLable][3];
            if(currentCut < t)
                grabCutMask.at<uchar>(i, j) = cv::GC_FGD;
            else if(currentCut >= t && currentCut < 0.5)
                grabCutMask.at<uchar>(i, j)  = cv::GC_PR_FGD;
            else if(currentCut >= 0.5 && currentCut < 1 - t)
                grabCutMask.at<uchar>(i, j)  = cv::GC_PR_BGD;
            else if(currentCut >= 1 - t)
                grabCutMask.at<uchar>(i, j)  = cv::GC_BGD;
        }
    }

    Mat background;
    Mat foreground;

    cout << "Grab Cut: " << endl;

    grabCut(avgColoredImage, grabCutMask, Rect(1, 1, 480, 640), background, foreground, 10);

    cv::compare(grabCutMask, cv::GC_PR_FGD, grabCutMask, cv::CMP_EQ);

    Mat foregroundIm(avgColoredImage.size(), CV_8UC3, Scalar(255,255,255));
    avgColoredImage.copyTo(foregroundIm, grabCutMask);

    imshow("w", foregroundIm);

    // trainLabels = cv::imread("../images/liberty_train_labels.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    // imshow("w", trainLabels);

    //imshow("w", vizTrainPatch);
    //imshow("ww", trainImage);
    //imshow("www", avgColoredImage);
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