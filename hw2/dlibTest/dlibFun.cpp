// First run sudo dnf install dlib-devel-18.18-4.fc25.x86_64
// Run libx11-dev

// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.  
    

    This example is essentially just a version of the face_landmark_detection_ex.cpp
    example modified to use OpenCV's VideoCapture object to read from a camera instead 
    of files.


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>
#include <numeric> // std::iota

// using namespace dlib;
// using namespace std;

using std::vector;
using std::cout;
using std::endl;
using std::cerr;
using std::string;
using std::to_string;

using cv::Mat;
using cv::Point2f;
using cv::imshow;
using cv::waitKey;
using cv::Scalar;
using cv::Rect;
using cv::Size;
using cv::imread;
using cv::imshow;
using cv::waitKey;

enum faceRegionEnum {
    leftEye,
    rightEye,
    mouth
};

// For this function we changed dlib/image_processing/../image_processing/full_object_detection.h:112:28
// .parts to be public, so we can access the points.

vector<Point2f> pushActualPointsFromRegion(vector<Point2f> points, vector<int> toChoose)
{
    vector<Point2f> chosenPoints;

    for(int num : toChoose)
    {
        chosenPoints.push_back(points[num]);
    }

    return chosenPoints;
}

vector<float> pushAxisOfPoints(vector<Point2f> points, vector<int> toChoose, string axis = "x")
{
    vector<float> chosenPoints;

    for(int num : toChoose)
    {
        if(axis == "x")
            chosenPoints.push_back(points[num].x);
        if(axis == "y")
            chosenPoints.push_back(points[num].y);
    }

    return chosenPoints;
}

Rect extractRectFromFacialLandmark(vector<Point2f> facialLandmark)
{
    vector<int> bottom(17);
    std::iota(bottom.begin(), bottom.end(), 0);
    vector<float> bottomValues = pushAxisOfPoints(facialLandmark, bottom, "y");
    
    // for(float f : bottomValues)
    // {
    //     cout << f << " ";
    // }
    // cout << endl;

    vector<int> top(10);
    std::iota(top.begin(), top.end(), 17);
    vector<float> topValues = pushAxisOfPoints(facialLandmark, top, "y");

    vector<int> right(8);
    std::iota(right.begin(), right.end(), 9);
    vector<float> rightValues = pushAxisOfPoints(facialLandmark, right, "x");
    
    vector<int> left(8);
    std::iota(left.begin(), left.end(), 0);
    vector<float> leftValues = pushAxisOfPoints(facialLandmark, left, "x");
    
    float topValue = topValues[ std::distance( topValues.begin(), std::min_element(topValues.begin(), topValues.end()) ) ];
    float bottomValue = bottomValues[ std::distance( bottomValues.begin(), std::max_element(bottomValues.begin(), bottomValues.end()) ) ];
    float leftValue = leftValues[ std::distance( leftValues.begin(), std::min_element(leftValues.begin(), leftValues.end()) ) ];
    float rightValue = rightValues[ std::distance( rightValues.begin(), std::max_element(rightValues.begin(), rightValues.end()) ) ];
    
    return Rect(leftValue - 20, topValue - 20, rightValue - leftValue + 40, bottomValue - topValue + 40);
}

vector<Point2f> convertDlibShapeToOpenCV(dlib::full_object_detection objectDet, Rect& outputRect)
{
    vector<Point2f> cvParts;
    dlib::rectangle dlibRect = objectDet.get_rect();

    for(dlib::point p : objectDet.parts)
    {
        Point2f cvPoint{p(0), p(1)};
        cvParts.push_back(cvPoint);
    }
    
    outputRect = Rect(dlibRect.left(), dlibRect.top(), dlibRect.width(), dlibRect.height());

    return cvParts;
}

Point2f centroidOfRegion(vector<Point2f> facialLandmark, faceRegionEnum faceRegion )
{
    int regionSize;
    int regionStartPoint;

    switch(faceRegion) {
        case faceRegionEnum::leftEye:   regionSize = 6;
                                        regionStartPoint = 36;
                                        break;
        case faceRegionEnum::rightEye:  regionSize = 6;
                                        regionStartPoint = 42;
                                        break;
        case faceRegionEnum::mouth:      regionSize = 19;
                                        regionStartPoint = 48;
                                        break;
        default: // The chin area
                                        regionSize = 17;
                                        regionStartPoint = 0;
    }

    vector<int> regionPoints(regionSize);
    std::iota(regionPoints.begin(), regionPoints.end(), regionStartPoint);

    vector<Point2f> actualPoints = pushActualPointsFromRegion(facialLandmark, regionPoints);

    Point2f sum  = std::accumulate(actualPoints.begin(), actualPoints.end(), Point2f(0.0f, 0.0f) );

    return Point2f(sum.x / regionSize, sum.y / regionSize);
}

vector<Mat> alignImageFaces(Mat image)
{
    try
    {
        dlib::image_window win;

        // Load face detection and pose estimation models.
        dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
        dlib::shape_predictor pose_model;
        dlib::deserialize("../shape_predictor_68_face_landmarks.dat") >> pose_model;

        cv::Mat temp = image.clone();
        // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
        // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
        // long as temp is valid.  Also don't do anything to temp that would cause it
        // to reallocate the memory which stores the image as that will make cimg
        // contain dangling pointers.  This basically means you shouldn't modify temp
        // while using cimg.
        dlib::cv_image<dlib::bgr_pixel> cimg(temp);

        // Detect faces 
        vector<dlib::rectangle> faces = detector(cimg);
        // Find the pose of each face.
        vector<dlib::full_object_detection> shapes;
        for (unsigned long i = 0; i < faces.size(); ++i)
            shapes.push_back(pose_model(cimg, faces[i]));

        // Convert the faces detected by dlib to something OpenCV can deal with.
        vector<vector<Point2f>> facialLandmarks(shapes.size());

        for(int i = 0; i < shapes.size(); i++)
        {
            Rect dummyRect;
            facialLandmarks[i] = convertDlibShapeToOpenCV(shapes[i], dummyRect);
        }

        // The locations of the facial landmarks visually presented:
        // https://github.com/cmusatyalab/openface/blob/master/images/dlib-landmark-mean.png

        vector<Mat> alignedFaces;

        if(facialLandmarks.size() > 0)
        {
            // for(int i = 0; i < facialLandmarks[0].size(); i++)
            // {
            //     circle(myImage, facialLandmarks[0][i], 3, Scalar(0, 0, 255));
            //     string objectTitle = std::to_string(i);
            //     cv::putText(myImage, objectTitle, facialLandmarks[0][i], cv::FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 255, 0), 0.5);
            // }

            for(vector<Point2f> face : facialLandmarks)
            {
                // circle(myImage, centroidOfRegion(face, faceRegionEnum::leftEye), 3, Scalar(0, 255, 0));
                // circle(myImage, centroidOfRegion(face, faceRegionEnum::rightEye), 3, Scalar(0, 255, 0));
                // circle(myImage, centroidOfRegion(face, faceRegionEnum::mouth), 3, Scalar(0, 255, 0));

                vector<Point2f> dstPoints = {Point2f(50, 60), Point2f(75, 120), Point2f(100, 60)};
                vector<Point2f> srcPoints = {centroidOfRegion(face, faceRegionEnum::leftEye)
                                                , centroidOfRegion(face, faceRegionEnum::mouth)
                                                , centroidOfRegion(face, faceRegionEnum::rightEye)};

                Mat affineTransformation = getAffineTransform(srcPoints, dstPoints);
                
                // cout << affineTrans << endl;
                Mat transformedFace;
                warpAffine(image, transformedFace, affineTransformation, Size(150, 175));

                alignedFaces.push_back(transformedFace);
            }

            return alignedFaces;

        }
        else
        {
            cerr << "No faces Detected! returning empty array" << endl;
            return vector<Mat>();
        }
    }
    catch(dlib::serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(std::exception& e)
    {
        cout << e.what() << endl;
    }
}


int main()
{
    Mat image = imread("../../images/bigBang.jpg");

    vector<Mat> faces = alignImageFaces(image);

    for(Mat face : faces)
    {
        imshow("w", face);
        waitKey(0);
    }
    
    return 0;
}


int main1()
{
    try
    {
        cv::VideoCapture cap(1);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

        dlib::image_window win;

        // Load face detection and pose estimation models.
        dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
        dlib::shape_predictor pose_model;
        dlib::deserialize("../shape_predictor_68_face_landmarks.dat") >> pose_model;

        // Grab and process frames until the main window is closed by the user.
        while(!win.is_closed())
        {
            // Grab a frame
            cv::Mat temp;
            cap >> temp;
            // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
            // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
            // long as temp is valid.  Also don't do anything to temp that would cause it
            // to reallocate the memory which stores the image as that will make cimg
            // contain dangling pointers.  This basically means you shouldn't modify temp
            // while using cimg.
            dlib::cv_image<dlib::bgr_pixel> cimg(temp);

            // Detect faces 
            vector<dlib::rectangle> faces = detector(cimg);
            // Find the pose of each face.
            vector<dlib::full_object_detection> shapes;
            for (unsigned long i = 0; i < faces.size(); ++i)
                shapes.push_back(pose_model(cimg, faces[i]));

            vector<vector<Point2f>> faceLandmarks(shapes.size());
            Mat myImage = temp.clone();
            Rect face;

            for(int i = 0; i < shapes.size(); i++)
            {
                faceLandmarks[i] = convertDlibShapeToOpenCV(shapes[i], face);
            }

            // https://github.com/cmusatyalab/openface/blob/master/images/dlib-landmark-mean.png

            if(faceLandmarks.size() > 0)
            {
                // for(int i = 0; i < faceLandmarks[0].size(); i++)
                // {
                //     circle(myImage, faceLandmarks[0][i], 3, Scalar(0, 0, 255));
                //     string objectTitle = std::to_string(i);
                //     cv::putText(myImage, objectTitle, faceLandmarks[0][i], cv::FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 255, 0), 0.5);
                // }

                Rect faceTry = extractRectFromFacialLandmark(faceLandmarks[0]);
                // cout << faceTry.x << " " << faceTry.y << " " << faceTry.width << " " << faceTry.height << endl;
                cv::rectangle(myImage, face, Scalar(255, 0, 0));
                cv::rectangle(myImage, faceTry, Scalar(0, 0, 255));
                circle(myImage, centroidOfRegion(faceLandmarks[0], faceRegionEnum::leftEye), 3, Scalar(0, 255, 0));
                circle(myImage, centroidOfRegion(faceLandmarks[0], faceRegionEnum::rightEye), 3, Scalar(0, 255, 0));
                circle(myImage, centroidOfRegion(faceLandmarks[0], faceRegionEnum::mouth), 3, Scalar(0, 255, 0));

                vector<Point2f> desiredPoints = {Point2f(50, 60), Point2f(75, 120), Point2f(100, 60)};
                vector<Point2f> actualPoints = {centroidOfRegion(faceLandmarks[0], faceRegionEnum::leftEye)
                                                , centroidOfRegion(faceLandmarks[0], faceRegionEnum::mouth)
                                                , centroidOfRegion(faceLandmarks[0], faceRegionEnum::rightEye)};

                //Mat affineTrans = getAffineTransform(desiredPoints, actualPoints);
                Mat affineTrans = getAffineTransform(actualPoints, desiredPoints);
                
                cout << affineTrans << endl;

                Mat transformedFace;
                warpAffine(temp, transformedFace, affineTrans, Size(150, 175));

                imshow("w", myImage);
                imshow("ww", transformedFace);
                waitKey(1);
            }

            // Display it all on the screen
            win.clear_overlay();
            win.set_image(cimg);
            win.add_overlay(render_face_detections(shapes));
        }
    }
    catch(dlib::serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(std::exception& e)
    {
        cout << e.what() << endl;
    }
}