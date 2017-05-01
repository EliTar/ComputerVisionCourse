
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

    // https://github.com/mesutpiskin/OpenCvObjectDetection/tree/master/haarcascades

    cv::CascadeClassifier facesDetector("../haarcascade_frontalface_default.xml");
    cv::CascadeClassifier eyesDetector("../haarcascade_eye_tree_eyeglasses.xml");
    cv::CascadeClassifier mouthDetector("../haarcascade_mcs_mouth.xml");

    vector<Rect>    faces,
                    eyes,
                    mouth;

    facesDetector.detectMultiScale(image, faces, 1.1, 15, 0, Size(80, 80));
    eyesDetector.detectMultiScale(image, eyes, 1.1, 10, 0);
    mouthDetector.detectMultiScale(image, mouth, 1.2, 25, 0);

    Mat drawing = drawObjectsWithTitles(image, faces, "Face");
    drawing = drawObjectsWithTitles(drawing, mouth, "Mouth");
    drawing = drawObjectsWithTitles(drawing, eyes, "Eye");

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

    vector<vector<Rect>> eyesConverted = convertEyesToFaceSpace(faces, eyesAssigned);

    int tryFaceNum = 0;

    Mat tryFace = actualFaces[tryFaceNum].clone();
    vector<Rect> faceEye, faceMouth;
    eyesDetector.detectMultiScale(tryFace, faceEye, 1.1, 10, 0);
    mouthDetector.detectMultiScale(tryFace, faceMouth, 1.1, 15, 0);

    vector<Point2f> desiredPoints = {Point2f(50, 60), Point2f(75, 80), Point2f(100, 60)};
    vector<Point2f> actualPoints = {returnCenterOfRect(faceEye[0]),
                                    Point2f( (faceEye[0].x + faceEye[0].width / 2 + faceEye[1].x + faceEye[1].width / 2) / 2,
                                    (faceEye[0].y + faceEye[0].height / 2 + faceEye[1].y + faceEye[1].height / 2) / 2 + 20) ,
                                    returnCenterOfRect(faceEye[1])};

    Mat tryFacedrawing = drawObjectsWithTitles(tryFace, faceEye, "Eye");
    tryFacedrawing = drawObjectsWithTitles(tryFacedrawing, faceMouth, "Mouth");
    
    imshow("w", tryFacedrawing);
    waitKey(0);

    desiredPoints = {Point2f(50, 60), Point2f(75, 120), Point2f(100, 60)};
    actualPoints = {returnCenterOfRect(faceEye[0]),
                    returnCenterOfRect(faceMouth[0]),
                    returnCenterOfRect(faceEye[1])};
    
    for(int i = 0; i < desiredPoints.size(); i++)
    {
        circle(tryFace, desiredPoints[i], 2, Scalar(0, 0, 255));
        circle(tryFace, actualPoints[i], 2, Scalar(0, 255, 0));
    }

    cv::rectangle(tryFace, faceEye[0], Scalar(255, 0, 0));
    cv::rectangle(tryFace, faceEye[1], Scalar(255, 0, 0));
    
    imshow("w", tryFace);
    waitKey(0);

    for(int i = 0; i < faces.size(); i++)
    {
        if(eyesAssigned[i].size() > 0)
        {
            drawing = drawObjectsWithTitles(drawing, eyesAssigned[i], "Eye " + to_string(i) );
            Rect eye = eyesAssigned[i][0];
            circle(drawing, Point(eye.x + eye.width / 2, eye.y + eye.height / 2), 1, Scalar(0, 0, 255));
        }
    }

    Mat affineTrans = getAffineTransform(desiredPoints, actualPoints);
    // Mat affineTrans = getAffineTransform(actualPoints, desiredPoints);

    cout << affineTrans << endl;
    
    Mat transformedFace;

    warpAffine(actualFaces[tryFaceNum], transformedFace, affineTrans, actualFaces[0].size());

    cv::imshow("www", drawing);
    cv::imshow("ww", transformedFace);
    cv::imshow("w", actualFaces[tryFaceNum]);
    
    cv::waitKey(0);

    return 0;
}