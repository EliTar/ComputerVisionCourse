#include <iostream>
#include <vector>
#include <random>

#include <opencv2/opencv.hpp>

typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

using std::vector;
using std::string;
using std::cout;
using std::endl;

using cv::imshow;
using cv::imread;

using cv::Mat;
using cv::Scalar;
using cv::Point;
using cv::Rect;
using cv::Range;
using cv::Vec3b;
using cv::Size;
using cv::InputArray;
using cv::OutputArray;

using cv::Point2f;
using cv::Point2i;
using cv::Point3f;
using cv::Vec3f;
using cv::Mat1i;
using cv::Mat1f;
using cv::Mat1b;
using cv::Mat3b;
using cv::Mat3f;

#define UTILITY if (false)

const int patchRadius = 3;
const int sideLength = 2 * patchRadius + 1;

void printTimeSinceLastCall(const char* message)
{
	static s64 freq = static_cast<int>(cv::getTickFrequency());
	static s64 last = cv::getTickCount();

	s64 curr = cv::getTickCount();
	s64 delta = curr - last;
	// double deltaMs = (double)delta / freq * 1000;
	double deltaSeconds = (double)delta / freq;

	// printf("%s: %.2f %.2f\n", message, deltaMs, deltaSeconds);
	printf("%s: %.2f\n", message, deltaSeconds);

	last = curr;
}

//
// SLIC Super Pixels Code
//

void VisualizeSuperPixels(
		const Mat &labels,
		Mat &dst,
		const vector<Vec3b> &colors,
		const vector<Point2i> &centers,
		bool showEdges = false,
		bool showCenters = true)
{
	CV_Assert(labels.type() == CV_32S);

	dst.create(labels.size(), CV_8UC3);
	auto labelIt = labels.begin<s32>();
	auto dstIt = dst.begin<cv::Vec3b>();

	auto labelEnd = labels.end<s32>();

	while (labelIt != labelEnd) {
		*dstIt = colors[*labelIt];

		labelIt++;
		dstIt++;
	}

	if (showEdges) {
		Mat1b edges{labels.size()};

		edges.setTo(0);

		for (int r = 0; r < labels.rows - 1; r++) {
			for (int c = 0; c < labels.cols - 1; c++) {
				int curr = labels.at<s32>(r, c);
				int right = labels.at<s32>(r, c + 1);
				int down = labels.at<s32>(r + 1, c);

				if (curr != right || curr != down) {
					edges(r, c) = 255;
				}
			}
		}

		dst.setTo(255, edges);
	}

	if (showCenters) {
		for (Point2i point : centers) {
			cv::circle(dst, point, 2, Scalar{0, 0, 0}, -1);
		}
	}
}

void SegmentSuperPixelsSLIC(
		InputArray _imgBGR,
		OutputArray _dst,
		vector<Point2i> *outKernelPositions,
		vector<Vec3b> *outKernelColors,
		int regionSize,
		float smoothness,
		float stopLimit = 0.01)
{
	CV_Assert(_imgBGR.type() == CV_8UC3);
	Mat3b imgBGR = _imgBGR.getMat();
	Mat3b imgLAB;
	cv::cvtColor(imgBGR, imgLAB, cv::COLOR_BGR2Lab);

	vector<Point2i> kernelPosition;
	vector<Vec3b> kernelColor;
	vector<Vec3b> kernelColorBGR;

	int rows = imgLAB.rows;
	int cols = imgLAB.cols;

	int N = rows * cols;
	int S = regionSize;

	//
	// Place initial kernels
	//

	{
		int paddingY = ((rows - 1) % S) / 2;
		int paddingX = ((cols - 1) % S) / 2;

		for (int y = paddingY; y < rows; y += S) {
			for (int x = paddingX; x < cols; x += S) {
				kernelPosition.push_back(Point2i{x, y});
				kernelColor.push_back(imgLAB(y, x));
			}
		}
	}

	cv::Mat3f colors;
	imgLAB.convertTo(colors, CV_32FC3);

	Mat1i segment{imgLAB.size()};
	Mat1f minDistance{imgLAB.size()};
	Mat display;

	int residualError = INT_MAX;

	while (residualError > stopLimit * S * kernelPosition.size()) {
		segment.setTo(-1);
		minDistance.setTo(rows * cols);
		residualError = 0;

		//
		// Assign each pixel to closest kernelPosition
		//

		for (unsigned int i = 0; i < kernelPosition.size(); i++) {
			Point2i kPosition{kernelPosition[i]};
			Point3f kColor{kernelColor[i]};

			Range rowRange{std::max(kPosition.y - S, 0), std::min(kPosition.y + S, rows)};
			Range colRange{std::max(kPosition.x - S, 0), std::min(kPosition.x + S, cols)};

			Mat rectColor{colors, rowRange, colRange};
			Mat1i rectSegment{segment, rowRange, colRange};
			Mat1f rectDistance{minDistance, rowRange, colRange};

			for (int r = 0; r < rectColor.rows; r++) {
				Vec3f *colorPtr = rectColor.ptr<cv::Vec3f>(r);
				float *distancePtr = rectDistance.ptr<float>(r);
				int *segmentPtr = rectSegment.ptr<int>(r);

				for (int c = 0; c < rectColor.cols; c++) {
					Vec3f pixelColor = *colorPtr;
					float currDist = 0;

					{
						float dx = pixelColor[0] - kColor.x;
						float dy = pixelColor[1] - kColor.y;
						float dz = pixelColor[2] - kColor.z;

						currDist += std::sqrt(dx * dx + dy * dy + dz * dz);
					}

					{
						float dx = c - S;
						float dy = r - S;

						currDist += smoothness / S * std::sqrt(dx * dx + dy * dy);
					}

					if (currDist < *distancePtr) {
						*distancePtr = currDist;
						*segmentPtr = i;
					}

					colorPtr++;
					distancePtr++;
					segmentPtr++;
				}
			}
		}

		//
		// Set each kernel position to the average of its pixels
		//

		for (unsigned int i = 0; i < kernelPosition.size(); i++) {
			Point2i kPosition = kernelPosition[i];
			Point3f kColor{kernelColor[i]};

			Range rowRange{std::max(kPosition.y - S, 0), std::min(kPosition.y + S + 1, rows)};
			Range colRange{std::max(kPosition.x - S, 0), std::min(kPosition.x + S + 1, cols)};

			Mat rectColor{colors, rowRange, colRange};
			Mat1i rectSegment{segment, rowRange, colRange};

			Point2f newPosition{0, 0};
			Point3f newColor{0, 0, 0};

			int count = 0;

			for (int r = 0; r < rectColor.rows; r++) {
				int *segmentPtr = rectSegment.ptr<int>(r);
				Point3f *colorPtr = rectColor.ptr<Point3f>(r);

				for (int c = 0; c < rectColor.cols; c++) {
					if (*segmentPtr == (int)i) {
						newPosition.x += c;
						newPosition.y += r;
						newColor += *colorPtr;

						count++;
					}
					segmentPtr++;
					colorPtr++;
				}
			}

			if (count > 0) {
				newPosition /= count;
				newColor /= count;

				Point2i lastPosition = kPosition;
				kernelPosition[i] = newPosition + Point2f{(float)colRange.start, (float)rowRange.start};
				kernelColor[i] = Vec3b{(u8)(newColor.x + 0.5f), (u8)(newColor.y + 0.5f), (u8)(newColor.z + 0.5f)};

				residualError += std::abs(kernelPosition[i].x - lastPosition.x) + std::abs(kernelPosition[i].y - lastPosition.y);
			}
		}

		cv::cvtColor(kernelColor, kernelColorBGR, cv::COLOR_Lab2BGR);
		VisualizeSuperPixels(segment, display, kernelColorBGR, kernelPosition);
		cv::imshow("Super Pixels", display);
		cv::waitKey(1);
	}

	//
	// Enforce pixel connectivity
	//

	Mat1i connectedSegment{imgLAB.size()};

	{
		connectedSegment.setTo(0);

		//
		// Add the biggest connected componnent of each super pixel
		//

		for (unsigned int i = 0; i < kernelPosition.size(); i++) {
			Point2i kPosition{kernelPosition[i]};
			Point3f kColor{kernelColor[i]};

			Range rowRange{std::max(kPosition.y - S, 0), std::min(kPosition.y + S, rows)};
			Range colRange{std::max(kPosition.x - S, 0), std::min(kPosition.x + S, cols)};

			Mat1i rectSegment{segment, rowRange, colRange};
			Mat1i rectConnectedSegment{connectedSegment, rowRange, colRange};
			Mat1b cluster = (rectSegment == i);

			Mat1i ccLabels;
			Mat1i ccStats;
			Mat _centroids;

			int ccCount = cv::connectedComponentsWithStats(cluster, ccLabels, ccStats, _centroids, 4, CV_32S);

			if (ccCount > 1) {
				int biggestCC = -1;
				int biggestCCSize = 0;

				for (int c = 1; c < ccCount; c++) {
					int currArea = ccStats(c, cv::CC_STAT_AREA);
					if (currArea > biggestCCSize) {
						biggestCC = c;
						biggestCCSize = currArea;
					}
				}

				Mat1b biggestCCMask = (ccLabels == biggestCC);

				cv::add(rectConnectedSegment, i + 1, rectConnectedSegment, biggestCCMask);
			}
		}

		//
		// Merge each of remaining pixels with its closest CC
		//

		Mat1i distanceLabels;
		Mat1b holes = (connectedSegment == 0);
		Mat _dist;

		cv::distanceTransform(
				holes,
				_dist,
				distanceLabels,
				cv::DIST_L2,
				3,
				cv::DIST_LABEL_PIXEL);

		Mat1i labelDict{1, N};

		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				if (holes(r, c) == 0) {
					labelDict(distanceLabels(r, c)) = segment(r, c);
				}
			}
		}

		for (int r = 0; r < rows; r++) {
			for (int c = 0; c < cols; c++) {
				connectedSegment(r, c) = labelDict(distanceLabels(r, c));
			}
		}
	}

	connectedSegment.copyTo(_dst);

	if (outKernelPositions) {
		*outKernelPositions = kernelPosition;
	}

	if (outKernelColors) {
		cv::cvtColor(kernelColor, kernelColorBGR, cv::COLOR_Lab2BGR);
		*outKernelColors = kernelColorBGR;
	}

	cv::cvtColor(kernelColor, kernelColorBGR, cv::COLOR_Lab2BGR);
	VisualizeSuperPixels(connectedSegment, display, kernelColorBGR, kernelPosition, true, false);
	cv::imshow("Super Pixels", display);
	cv::waitKey(1);
	cv::waitKey(1);

	// NOTE(Andrey): With one waitKey(1), the window sometimes doesn't update
}

Vec3b ConvertLabelToColor(int label)
{
	static vector<Vec3b> colors = {
		{0, 0, 0},
		{255, 0, 0},
		{0, 255, 0},
		{0, 0, 255},
		{255, 255, 0},
		{255, 0, 255},
		{0, 255, 255},
		{128, 128, 128},
		{255, 255, 255},
	};

	return colors[label % colors.size()];
}

Mat FragmentToColorDistVisualization(Mat image, Mat imageLabels, vector<vector<double>> distances, int label)
{
    Mat vizImage = Mat::zeros(cv::Size(image.cols, image.rows), CV_8UC3);

    for(int i = 0; i < imageLabels.rows; i++)
    {
        for(int j = 0; j < imageLabels.cols; j++)
        {
            Vec3b pixelColor = image.at<Vec3b>(i , j);
            double dist = distances[imageLabels.at<int>(i , j)][label];
            uchar a = round(pixelColor[0] * dist);
            uchar b = round(pixelColor[1] * dist);
            uchar c = round(pixelColor[2] * dist);

            vizImage.at<Vec3b>(i , j) = Vec3b(a, b, c);
        }
    }

    // vector<bool> imageLabels(size.)

    // for(int i = 0; i < imageLabels.rows; i++)
    // {
    //     for(int j = 0; j < imageLabels.cols; j++)
    //     {
    //         string text = (string)dist;
    //         int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
    //         double fontScale = 2;
    //         int thickness = 3;
    //         cv::Point textOrg(10, 130);
    //         cv::putText(img, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness,8);

    //         Vec3b pixelColor = image.at<Vec3b>(i , j);
    //         double dist = distances[imageLabels.at<int>(i , j)][label];
    //         uchar a = round(pixelColor[0] * dist);
    //         uchar b = round(pixelColor[1] * dist);
    //         uchar c = round(pixelColor[2] * dist);

    //         vizImage.at<Vec3b>(i , j) = Vec3b(a, b, c);
    //     }
    // }

    return vizImage;
}

Mat PaintLabelsTrainImage(Mat imageLabels)
{
    imageLabels.convertTo(imageLabels, CV_32S);
    Mat newImage = Mat::zeros(cv::Size(imageLabels.cols, imageLabels.rows), CV_8UC3);
    for(int i = 0; i < imageLabels.rows; i++)
    {
        for(int j = 0; j < imageLabels.cols; j++)
        {
            newImage.at<Vec3b>(i , j) = ConvertLabelToColor(imageLabels.at<int>(i , j));
        }
    }
    return newImage;
}

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

    for(unsigned int i = 0; i < pixelValues.size(); i++)
    {
        for(unsigned int j = 0; j < pixelValues[i].size(); j++)
        {
            sumA[i] += pixelValues[i][j][0];
            sumB[i] += pixelValues[i][j][1];
            sumC[i] += pixelValues[i][j][2];
            howMany[i]++;
        }
    }

    vector<Vec3b> avgPixels(maxLabel + 1);
    for(unsigned int i = 0; i < pixelValues.size(); i++)
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

vector<vector<Rect>> RandomPatchesForEachLabel(Mat image, Mat imageLabels, bool innerPatchesOnly = false)
{
    imageLabels.convertTo(imageLabels, CV_32S);

    const int   imageRows = image.rows,
                imageCols = image.cols;
    double      maxLabel,
                minLabel;

    // finding the minimum and maximum values of the labeling.
    minMaxLoc(imageLabels, &minLabel, &maxLabel);
    int sideLength = 2 * patchRadius + 1;

    ////
    // Step 2: Choose several random pixels for each fregment
    ////

	//
	// We build a list of the pixels of each label
	//

    vector<vector<Point>> pixelsByLabel(maxLabel + 1);
	vector<int> pixelCountByLabel(maxLabel + 1, 0);

	//
	// To be eligble for participation in the pixel loterry, pixels have to:
	//
	// 1) 18 years old or older
	// 2) Be further than patchRadius from image boundary
	// 3) Not be on the edge between segments
	//

    for(int r = patchRadius; r < imageRows - patchRadius; r++)
    {
        for(int c = patchRadius; c < imageCols - patchRadius; c++)
        {
			Point p{c, r};

			Point c1{c - patchRadius, r - patchRadius};
			Point c2{c - patchRadius, r + patchRadius};
			Point c3{c + patchRadius, r - patchRadius};
			Point c4{c + patchRadius, r + patchRadius};

			bool innerPatch = imageLabels.at<int>(p) == imageLabels.at<int>(c1) &&
					imageLabels.at<int>(p) == imageLabels.at<int>(c2) &&
					imageLabels.at<int>(p) == imageLabels.at<int>(c3) &&
					imageLabels.at<int>(p) == imageLabels.at<int>(c4);

			int label = imageLabels.at<int>(r , c);

			if (innerPatch || !innerPatchesOnly)
			{
				pixelsByLabel[label].push_back(Point(c, r));
			}
			pixelCountByLabel[label]++;
        }
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    vector<vector<Rect>> patches(maxLabel + 1);

	auto sampleCount = [](int pixelCount) {
		return round(sqrt(pixelCount) / 2);
	};

    for(int i = minLabel; i < maxLabel + 1; i++)
    {
        int currentSize = pixelsByLabel[i].size();

		if (currentSize == 0)
			continue;

        int toChoose = sampleCount(pixelCountByLabel[i]);
        std::uniform_int_distribution<> dis(0, currentSize - 1);

        for(int j = 0; j < toChoose; j++)
        {
			Point p = pixelsByLabel[i][dis(gen)];
			Rect patchRect{p.x - patchRadius, p.y - patchRadius, sideLength, sideLength};
            patches[i].push_back(patchRect);
        }
    }

	return patches;
}

// TODO: Check this function!!! likes to give exeptions. maybe its Vec3b?
// Fixed. Probably the problem was with acssesing outside the matrix...

Mat VisualizePatches(Mat image, vector<vector<Rect>> patches, int patchLabel = -1)
{
    Mat vizPatch = Mat::zeros(cv::Size(image.cols, image.rows), CV_8UC3);

    for(int i = 0; i < (int)patches.size(); i++)
    {
        if(patchLabel == i || patchLabel == -1)
        {
            for(Rect rec : patches[i])
            {
                for(int i = rec.x; i < rec.x + sideLength; i++)
                {
                    for(int j = rec.y; j < rec.y + sideLength; j++)
                    {
                        if( i < image.rows && j < image.cols )
                            vizPatch.at<cv::Vec3b>(i,j) = image.at<cv::Vec3b>(i,j);
                    }
                }
            }
        }
    }

    return vizPatch;
}

double PatchDistance(const Mat& a, const Mat& b)
{
	double distance = 0;

	for(int r = 0; r < a.rows; r++)
	{
		const Vec3b *firstPtr = a.ptr<Vec3b>(r);
		const Vec3b *secondPtr = b.ptr<Vec3b>(r);
		for(int c = 0; c < a.cols; c++)
		{
			Vec3b first = *firstPtr;
			Vec3b second = *secondPtr;

			double d0 = first[0] - second[0];
			double d1 = first[1] - second[1];
			double d2 = first[2] - second[2];

			double differences = d0 * d0 + d1 * d1 + d2 * d2;

			distance += sqrt(differences);

			firstPtr++;
			secondPtr++;
		}
	}

	return distance;
}

Mat DrawBorderFromLabels(Mat image, Mat imageLabels)
{
    imageLabels.convertTo(imageLabels, CV_32S);
    Mat newImage = Mat::zeros(cv::Size(image.cols, image.rows), CV_8UC3);

    int imageCols = image.cols;
    int imageRows = image.rows;

    for(int i = 1; i < imageRows - 1; i++)
    {
        for(int j = 1; j < imageCols - 1; j++)
        {
            bool boarderPixel = false;
            for(int x = -1; x < 2; x++)
            {
                for(int y = -1; y < 2; y++)
                {
                    if( imageLabels.at<int>(i + x , j + y) != imageLabels.at<int>(i , j) )
                        boarderPixel = true;
                }
            }
            if(boarderPixel)
                newImage.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
            else
                newImage.at<cv::Vec3b>(i, j) = image.at<cv::Vec3b>(i, j);
        }
    }

    return newImage;
}

Mat SubtructFregmentAverageColor(Mat image, Mat imageLabels)
{
    imageLabels.convertTo(imageLabels, CV_32S);
    Mat newImage = Mat::zeros(cv::Size(image.cols, image.rows), CV_8UC3);

    const int   imageRows = image.rows,
                imageCols = image.cols;
    double      maxLabel,
                minLabel;

    minMaxLoc(imageLabels, &minLabel, &maxLabel);

    vector<vector<Point>> pixelsLabeledPoints(maxLabel + 1);
    vector<vector<int>> pixelsLabeledValues(maxLabel + 1);

    for(int i = 0; i < imageRows; i++)
    {
        for(int j = 0; j < imageCols; j++)
        {
            pixelsLabeledPoints[imageLabels.at<int>(i , j)].push_back(Point(i, j));
            pixelsLabeledValues[imageLabels.at<int>(i , j)].push_back(image.at<Vec3b>(i, j)[0]);
            pixelsLabeledValues[imageLabels.at<int>(i , j)].push_back(image.at<Vec3b>(i, j)[1]);
            pixelsLabeledValues[imageLabels.at<int>(i , j)].push_back(image.at<Vec3b>(i, j)[2]);
        }
    }

    for(int i = 0; i < (int)pixelsLabeledPoints.size(); i++)
    {
        int fregmentAverage = (int) (accumulate( pixelsLabeledValues[i].begin(), pixelsLabeledValues[i].end(), 0.0) / (pixelsLabeledValues[i].size() * 3) );
        for(Point p : pixelsLabeledPoints[i])
        {
            Vec3b currentColor = image.at<Vec3b>(p.x , p.y);
            int a = currentColor[0] - fregmentAverage;
            int b = currentColor[1] - fregmentAverage;
            int c = currentColor[2] - fregmentAverage;
            if(a < 0)
                a = 0;
            if(b < 0)
                b = 0;
            if(c < 0)
                c = 0;
            newImage.at<Vec3b>(p.x , p.y) = Vec3b(a, b, c);
        }
    }

    return newImage;
}

void Usage()
{
	const char *message =
R"msg(
	Usage: hw1 [FILE_NAME]
	Please make sure you have the following files in the ../images folder:
	"[FILE_NAME]_train", "[FILE_NAME]_test" and "[FILE_NAME]_train_labels"
	with .jpg, .tif or .png endings.
	Good luck!
)msg";
	cout << message << endl;
}

Mat LoadImageWithSomeExtension(string basePath, int flags = cv::IMREAD_COLOR)
{
	static vector<string> extensions = {
		".tif",
		".jpg",
		".png",
	};

	for (auto ext : extensions)
	{
		Mat image = cv::imread(basePath + ext, flags);

		if (image.data)
			return image;
	}

	return Mat{};

}

void DecreaseImageContrast(Mat &image, int by = 2)
{
	cv::cvtColor(image, image, cv::COLOR_BGR2HSV);

	for (int r = 0; r < image.rows; r++) {
		for (int c = 0; c < image.cols; c++) {
			Vec3b color = image.at<Vec3b>(r, c);
			color[2] = 128 + (color[2] - 128) / by;
			image.at<Vec3b>(r, c) = color;
		}
	}

	cv::cvtColor(image, image, cv::COLOR_HSV2BGR);
}

int main(int argc, char *argv[])
{
	Mat trainImage;
	Mat trainLabels;
	Mat testImage;

	//
	// Open Images
	//

	string fileName;

	if (argc == 2)
	{
		fileName = string{argv[1]};
	}

	bool openedImages = false;

	while (!openedImages)
	{
		if (fileName == "")
		{
			std::cout << std::endl;
			std::cout << "\tEnter the basic filename: (giraffes, girl, dog, kayak, veg, zebra, ...):" << std::endl;
			std::cout << "?> ";
			std::cin >> fileName;
		}

		string folder = "../images/";

		string trainImagePath = folder + fileName + "_train";
		string trainLabelsPath = folder + fileName + "_train_labels";
		string testImagePath = folder + fileName + "_test";

		trainImage = LoadImageWithSomeExtension(trainImagePath);
		trainLabels = LoadImageWithSomeExtension(trainLabelsPath, CV_LOAD_IMAGE_GRAYSCALE);
		testImage = LoadImageWithSomeExtension(testImagePath);

		if (trainImage.empty() || trainLabels.empty() || testImage.empty())
		{
			std::cerr << endl;
			if (trainImage.empty())
				std::cerr << "\tERROR: Couldn't open file " + trainImagePath << endl;
			if (trainLabels.empty())
				std::cerr << "\tERROR: Couldn't open file " + trainLabelsPath << endl;
			if (testImage.empty())
				std::cerr << "\tERROR: Couldn't open file " + testImagePath << endl;

			std::cerr << "\tPlease make sure your files are there with the correct extension" << endl;

			fileName = "";
		}
		else
		{
			openedImages = true;
		}

	}

    //
    // Step 1: Compute input image fragments
    //


	Mat testLabels;

	SegmentSuperPixelsSLIC(
			testImage,
			testLabels,
			nullptr,
			nullptr,
			25,
			50,
			0.005);

    printTimeSinceLastCall("Generate Super Pixels");

	UTILITY
	{
		// Shows the superpixels formed on the image.
		Mat showMeThePixels = DrawBorderFromLabels(testImage, testLabels);
		cv::imshow("w", showMeThePixels);
		cv::waitKey(0);
	}


	//
	// Step 2: Select test and train patches
	//

    vector<vector<Rect>> testFragmentPatches = RandomPatchesForEachLabel(testImage, testLabels);

    printTimeSinceLastCall("Random patches test");

	UTILITY
	{
		// Show the chosen patches.

    	Mat vizTestPatch = VisualizePatches(testImage, testFragmentPatches, -1);
    	cv::imshow("w", vizTestPatch);
    	cv::waitKey(0);
	}

	const bool innerPatchesOnly = true;
    vector<vector<Rect>> trainPatches = RandomPatchesForEachLabel(trainImage, trainLabels, innerPatchesOnly);

    printTimeSinceLastCall("Random patches train");

	UTILITY
	{
		for (int i = -1; i < 5; i++) {
			Mat vizTrainPatch = VisualizePatches(trainImage, trainPatches, i);
			cv::imshow("w", vizTrainPatch);
			cv::waitKey(0);
		}
	}

	//
	// Step 3: Calculate distances between fragments and labels
	//

	// We convert the images to CIELAB colorspace, because
	// distances there better reflect human perception.

    Mat trainImageLab = trainImage.clone();
    Mat testImageLab = testImage.clone();;

	DecreaseImageContrast(trainImageLab, 2);
	DecreaseImageContrast(testImageLab, 2);

    cvtColor(trainImageLab, trainImageLab, CV_BGR2Lab);
    cvtColor(testImageLab, testImageLab, CV_BGR2Lab);

    // The calculation of the distances of each patch to each of the labels.
    // We iterate on each fragment.
    // For each one of them, we go through all the test patches in the fragment.
    // For each patch, we go through each color.
    // For each color, we go through the corasponding patches of this color.
    // Then we just calculate the distance.

    vector<vector<vector<double>>> distancePerPixel(testFragmentPatches.size(), vector<vector<double>>(trainPatches.size()));

    for(int i = 0; i < (int)testFragmentPatches.size(); i++)
    {
        for(Rect testPatch : testFragmentPatches[i])
        {
            for(int j = 0; j < (int)trainPatches.size(); j++)
            {
                double smallestColorDistance = DBL_MAX;

                for(Rect trainPatch : trainPatches[j])
                {
                    double currentPatchDistance = PatchDistance(Mat{trainImageLab, trainPatch}, Mat{testImageLab, testPatch});

                    if(currentPatchDistance < smallestColorDistance)
                    {
                        smallestColorDistance = currentPatchDistance;
                    }
                }
                distancePerPixel[i][j].push_back(smallestColorDistance);
            }
        }
    }

    printTimeSinceLastCall("Calculate patch distances");

    // Choose the median value of all minimum distances.
    // A robust voting scheme as explaind in the paper.

    vector<vector<double>> fragmentDistance(testFragmentPatches.size(), vector<double>(trainPatches.size()));

    for(int i = 0; i < (int)testFragmentPatches.size(); i++)
    {
        for(int j = 0; j < (int)trainPatches.size(); j++)
        {
            if(distancePerPixel[i][j].size() != 0)
            {
                std::nth_element(   distancePerPixel[i][j].begin(),
                                    distancePerPixel[i][j].begin() + distancePerPixel[i][j].size() / 2,
                                    distancePerPixel[i][j].end());

                fragmentDistance[i][j] = distancePerPixel[i][j][distancePerPixel[i].size() / 2];
            }
        }
    }

    printTimeSinceLastCall("Calculate fragment distances");

    // We normalize each fragment distace to the range of [0, 1]
    // as asked in the assigment.

    double  maxVal = -1,
            minVal = DBL_MAX;

    for(int i = 0; i < (int)fragmentDistance.size(); i++)
    {
        for(int j = 0; j < (int)trainPatches.size(); j++)
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

    vector<vector<double>> normalizedFregmentColorDistance(testFragmentPatches.size(), vector<double>(trainPatches.size()));

    for(int i = 0; i < (int)fragmentDistance.size(); i++)
    {
        for(int j = 0; j < (int)trainPatches.size(); j++)
        {
            normalizedFregmentColorDistance[i][j] =
                        (fragmentDistance[i][j] - minVal) /
                        (maxVal - minVal);
        }
    }

    printTimeSinceLastCall("Normalize Distances");

    // Normalize each fragment distance.
    // After a discussion with Hagit and a test we've made,
    // we found out that normalizing each of those distances as well,
    // in the range of the maximum and minimum distances of the particular fragment
    // gives better results.
    // When we consider the grab cut algorithm -
    // it's much more logical for the algorithm to have a more relevant view of each distance -
    // meaning, if a certain distance is 0.1 it makes a big difference if it's normalized
    // globally or locally, patch-wise.

    for(int i = 0; i < (int)testFragmentPatches.size(); i++)
    {
		auto &distances = normalizedFregmentColorDistance[i];

		double maxVal = *std::max_element(distances.begin(), distances.end());
		double minVal = *std::min_element(distances.begin(), distances.end());

        for(int j = 0; j < (int)trainPatches.size(); j++)
        {
            normalizedFregmentColorDistance[i][j] =
                        (normalizedFregmentColorDistance[i][j] - minVal) /
                        (maxVal - minVal);
        }
    }

    printTimeSinceLastCall("Normalize each fregment distance");

    UTILITY
    {
        for(int i = 0; i< (int)trainPatches.size(); i++)
        {
            Mat vizFrag = FragmentToColorDistVisualization(testImage, testLabels, normalizedFregmentColorDistance, i);
            imshow("ww", vizFrag);
            cv::waitKey(0);
        }
    }

    UTILITY
    {
        Mat justShowTheLabeles = PaintLabelsTrainImage(trainLabels);
        imshow("w", justShowTheLabeles);
        cv::waitKey(0);
    }

	//
    // Step 4: We now have the mapping! Hurray!!!!
	//

    Mat avgColoredImage = PaintInAverageColor(testImage, testLabels);

    vector<Mat> foregroundImages(trainPatches.size());
    vector<vector<vector<int>>> pixelVotes(avgColoredImage.rows, vector<vector<int>>(avgColoredImage.cols) );

    for(int labelNumber = 0; labelNumber < (int)trainPatches.size(); labelNumber++)
    {
        Mat grabCutMask = Mat::zeros(avgColoredImage.size(), CV_8U);

        for(int i = 0; i < avgColoredImage.rows; i++)
        {
            for(int j = 0; j < avgColoredImage.cols; j++)
            {
                double currentLabel = testLabels.at<int>(i, j);
                double currentCut = normalizedFregmentColorDistance[currentLabel][labelNumber];

                if(currentCut < 0.05)
                    grabCutMask.at<uchar>(i, j)  = cv::GC_PR_FGD;
                else if(currentCut >= 0.05)
                    grabCutMask.at<uchar>(i, j)  = cv::GC_PR_BGD;

            }
        }

        Mat background;
        Mat foreground;

		cv::grabCut(avgColoredImage, grabCutMask, Rect(1, 1, 64, 64), background, foreground, 8);
		grabCutMask = (grabCutMask == cv::GC_PR_FGD);

        // This sets pixels that are equal to 255.

        string grabCutString = "Grab Cut " + std::to_string(labelNumber);
        printTimeSinceLastCall(grabCutString.c_str());

        foregroundImages[labelNumber] = Mat{avgColoredImage.size(), CV_8UC3, Scalar{255,255,255}};
        testImage.copyTo(foregroundImages[labelNumber], grabCutMask);

		for(int i = 0; i < avgColoredImage.rows; i++)
		{
			for(int j = 0; j < avgColoredImage.cols; j++)
			{
				if(grabCutMask.at<uchar>(i, j) == 255)
				{
					pixelVotes[i][j].push_back(labelNumber);
				}
			}
		}
    }

    Mat finalLabeling = Mat::zeros(avgColoredImage.size(), CV_8U);

    for(int i = 0; i < avgColoredImage.rows; i++)
    {
        for(int j = 0; j < avgColoredImage.cols; j++)
        {
			int finalIndex;

            if(pixelVotes[i][j].size() == 0)
            {
				int fragment = testLabels.at<int>(i, j);
				auto &distances = normalizedFregmentColorDistance[fragment];
				int index = std::min_element(distances.begin(), distances.end()) - distances.begin();

                finalIndex = index;
            }
            else if(pixelVotes[i][j].size() == 1)
            {
                finalIndex = pixelVotes[i][j][0];
            }
            else
            {
				int fragment = testLabels.at<int>(i, j);
				auto &distances = normalizedFregmentColorDistance[fragment];

                int index = pixelVotes[i][j][0];
                for(int k : pixelVotes[i][j])
                {
                    if (distances[k] < distances[index] )
					{
						index = k;
					}
                }

				finalIndex = index;
            }

			finalLabeling.at<uchar>(i, j) = finalIndex;
        }
    }

    printTimeSinceLastCall("Final Labeling");

    for(int i = 0; i < (int)trainPatches.size(); i++)
    {
		string windowName = "Foreground ";
		windowName += std::to_string(i);
        imshow(windowName, foregroundImages[i]);
    }

    Mat finalViz = PaintLabelsTrainImage(finalLabeling);
    Mat finalVizBoarder = DrawBorderFromLabels(testImage, finalLabeling);

    imshow("Contours", finalVizBoarder);
    imshow("Final Labeling", finalViz);

    while(cv::waitKey(0) != 'q')
    { }

    return 0;
}
