
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

using std::cout;
using std::vector;
using std::endl;
using std::string;
using std::to_string;

using cv::Mat;
using cv::Rect;
using cv::imshow;
using cv::Size;
using cv::Scalar;
using cv::imread;
using cv::Point;
using cv::waitKey;
using cv::Point2f;


// Draw a rectangle surronding the objects and adding their title accordingly

Mat drawObjectsWithTitles(Mat image, vector<Rect> objects, string objectName)
{
    CV_Assert(objects.size() != 0);

    Mat drawnImage = image.clone();

    for(int i = 0; i < objects.size(); i++)
    {
        string objectTitle = objectName + " " + std::to_string(i);
        cv::rectangle(drawnImage, objects[i], Scalar(0, 255, 0));
        cv::putText(drawnImage, objectTitle, Point(objects[i].x, objects[i].y), cv::FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
    }

    return drawnImage;
}

vector<Mat> resizeAndReturnFaces(Mat image, vector<Rect> faceBoxes, int squareSide)
{
    CV_Assert(faceBoxes.size() > 0);

    vector<Mat> actualFaces;

    for(Rect faceBox : faceBoxes)
    {
        Mat face(image, faceBox);
        resize(face, face, Size(squareSide, squareSide));
        actualFaces.push_back(face);
    }

    return actualFaces;
}

vector<vector<Rect>> assignEyesToFace(vector<Rect> faceBoxes, vector<Rect> eyeBoxes)
{
    CV_Assert(faceBoxes.size() > 0 && eyeBoxes.size() > 0);

    vector<vector<Rect>> eyesAssigned(faceBoxes.size());

    // http://stackoverflow.com/questions/29120231/how-to-verify-if-rect-is-inside-cvmat-in-opencv

    for(int i = 0; i < faceBoxes.size(); i++)
    {
        for(Rect eye : eyeBoxes)
        {
            if( (faceBoxes[i] | eye) == faceBoxes[i] )
            {
                eyesAssigned[i].push_back(eye);
            }
        }
    }

    return eyesAssigned;
}

vector<vector<Rect>> convertEyesToFaceSpace(vector<Rect> faceBoxes, vector<vector<Rect>> imageSpaceEyes)
{
    CV_Assert(faceBoxes.size() > 0 && imageSpaceEyes.size() > 0);

    vector<vector<Rect>> eyesConvertd(imageSpaceEyes.size());

    for(int i = 0; i < imageSpaceEyes.size(); i++)
    {
        if(imageSpaceEyes[i].size() > 0)
        {
            for(Rect eye : imageSpaceEyes[i])
            {
                Rect eyeConvertd = Rect(eye.x - faceBoxes[i].x, eye.y - faceBoxes[i].y, eye.width, eye.height);
                eyesConvertd[i].push_back(eyeConvertd);
            }
        }
    }

    return eyesConvertd;
}

Point2f returnCenterOfRect(Rect r)
{
    return Point2f(r.x + r.width / 2, r.y + r.height / 2);
}

int main()
{
    Mat image = imread("../fam.jpg");

    cv::CascadeClassifier facesDetector("../haarcascade_frontalface_default.xml");
    cv::CascadeClassifier eyesDetector("../haarcascade_eye_tree_eyeglasses.xml");

    vector<Rect>    faces,
                    eyes;

    facesDetector.detectMultiScale(image, faces, 1.1, 15, 0, Size(80, 80));
    eyesDetector.detectMultiScale(image, eyes, 1.1, 10, 0);

    Mat drawing = drawObjectsWithTitles(image, faces, "Face");
    //drawing = drawObjectsWithTitles(drawing, eyes, "Eye");

    vector<Mat> actualFaces = resizeAndReturnFaces(image, faces, 150);

    for(Mat f : actualFaces)
    {
        cv::imshow("w", f);
        cv::waitKey(0);
    }

    vector<vector<Rect>> eyesAssigned = assignEyesToFace(faces, eyes);

    for(int i = 0; i < faces.size(); i++)
    {
        if(eyesAssigned[i].size() > 0)
        {
            drawing = drawObjectsWithTitles(drawing, eyesAssigned[i], "Eye " + to_string(i) );
            Rect eye = eyesAssigned[i][0];
            circle(drawing, Point(eye.x + eye.width / 2, eye.y + eye.height / 2), 1, Scalar(0, 0, 255));
        }
    }

   
    cv::imshow("w", drawing);
    // cv::imshow("ww", transformedFace);
    cv::waitKey(0);

    return 0;
}