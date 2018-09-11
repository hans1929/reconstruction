#include <stdio.h>
#include <iostream>
#include <iomanip>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
#include "functions.h"

using namespace cv;
using namespace std;

//function get keypoints and matches
void getKeyAndMatch(Mat  img_a, Mat img_b, 
					vector<KeyPoint> &keypoint_a, 
					vector<KeyPoint> &keypoint_b, 
					vector<DMatch> &match)//get keypionts and matches from a set of two images
{
	//Detect the keypoints using SIFT
	//int minHessian = 800;
	//SurfFeatureDetector detector(minHessian);
	SiftFeatureDetector detector;
	detector.detect( img_a, keypoint_a );
	detector.detect( img_b, keypoint_b );

	//Calculate descriptors (feature vectors)
	//SurfDescriptorExtractor extractor;
	SiftDescriptorExtractor extractor;
	Mat descriptors_a, descriptors_b;
	extractor.compute( img_a, keypoint_a, descriptors_a );
	extractor.compute( img_b, keypoint_b, descriptors_b );

	//Matching descriptor vectors with a brute force matcher
	BFMatcher matcher(NORM_L2);
	matcher.match( descriptors_a, descriptors_b, match );
}//end function get keypoints and matches

// function Convert keypoints into Point2f
void cvtKeytoP2f(vector<KeyPoint> keypoint_a, 
				 vector<KeyPoint> keypoint_b, 
				 vector<Point2f> &point2f_a, 
				 vector<Point2f> &point2f_b, 
				 vector<DMatch> match)
{
	for (std::vector<cv::DMatch>::const_iterator it= match.begin(); it!= match.end(); ++it)
	{
		// Get the position of the a keypoints
		float x= keypoint_a[it->queryIdx].pt.x;
		float y= keypoint_a[it->queryIdx].pt.y;
		point2f_a.push_back(cv::Point2f(x,y));
		// Get the position of b keypoints
		x= keypoint_b[it->trainIdx].pt.x;
		y= keypoint_b[it->trainIdx].pt.y;
		point2f_b.push_back(cv::Point2f(x,y));
	}
}//end function Convert keypoints into Point2f

//function get good match
void getGoodMatch(vector<DMatch> matche, vector<uchar> inlier, vector<DMatch>& good_matche)
{
	// extract the surviving (inliers) good matches
	std::vector<uchar>::const_iterator itIn3= inlier.begin();
	std::vector<cv::DMatch>::const_iterator itM3= matche.begin();
	// for all matches
	for ( ;itIn3!= inlier.end(); ++itIn3, ++itM3) 
	{
		if (*itIn3)
		{ // it is a valid match
			good_matche.push_back(*itM3);
		}
	}
}//end function get good match



/**************************************** main function *********************************/
int main( int argc, char** argv )
{
/************************************* 1st & 2nd image *********************************/
	//interesting points detection
	//iphone
	Mat K = (Mat_<double>(3, 3) <<7.0784196432192323e+002, 0., 3.0550000000000000e+002,
								  0., 7.0784196432192323e+002, 4.0750000000000000e+002, 
								  0., 0., 1.);//intrinsic matrix
	Mat D = (Mat_<double>(5, 1) <<4.0478810531698453e-002, 2.3369118136386455e-001, 0.,
								  0., -8.2338413527898235e-001);//distortion matrix
	


	//read & show the image
	Mat img_1 = imread( "img1.jpg", CV_LOAD_IMAGE_GRAYSCALE );
	Mat img_2 = imread( "img2.jpg", CV_LOAD_IMAGE_GRAYSCALE );
	//Mat img_1 = imread( "desk_left.jpg", CV_LOAD_IMAGE_GRAYSCALE );
	//Mat img_2 = imread( "desk_right.jpg", CV_LOAD_IMAGE_GRAYSCALE );
	imshow("original_left", img_1 );
	imshow("original_right", img_2 );
	//if invalid read
	if( !img_1.data || !img_2.data )
	{ return -1; }

	//detect interesting points and point matches
	vector<KeyPoint> keypoints_1, keypoints_2;
	vector< DMatch > matches_12;
	getKeyAndMatch(img_1, img_2, keypoints_1, keypoints_2, matches_12);


	//Convert keypoints into Point2f based on matches
	std::vector<Point2f> points1, points2;
	cvtKeytoP2f(keypoints_1, keypoints_2, points1, points2, matches_12);//points1 and points2 matches in every entry

	std::cout <<"The interesting points detection is done\nThe number of keypoints of image 1(2) is"<< points1.size() << std::endl; 

	//estimate fundamental matrix using RANSAC
	// Compute F matrix using RANSAC
	std::vector<uchar> inliers_12(points1.size(),0);
	cv::Mat F_12 = cv::findFundamentalMat(
		cv::Mat(points1),cv::Mat(points2), // matching points
		inliers_12,      // match status (inlier ou outlier)  
		CV_FM_RANSAC, // RANSAC method
		1,            // distance to epipolar line
		0.99);        // confidence probability

	// extract the surviving (inliers) good matches
	vector <DMatch> good_matches_12;
	getGoodMatch(matches_12, inliers_12, good_matches_12);
	
	//extract the good keypoints in the format of Point2f
	vector <Point2f> good_points1, good_points2;
	cvtKeytoP2f(keypoints_1, keypoints_2, good_points1, good_points2, good_matches_12);//good_points1 and good_points2 matches in every entry

	std::cout << "After using RANSAC, the number of matched points between image 1 & 2 (after cleaning) is: " << good_matches_12.size() << std::endl;

	Mat img_matches12;//Draw good matches
	drawMatches( img_1, keypoints_1, img_2, keypoints_2, good_matches_12, img_matches12 );

	imshow("Good_Matches12", img_matches12 );  //show good matches after ransac

	//std::cout<<"The fundamental matrix is\n"<<F_12<<std::endl;

	//Using partial camera calibration to find camera matrix
	Mat E = K.t() * F_12 * K;//essential matrix
	//cout << "The essential matrix is:\n"<<E << endl;
	SVD svd(E);
	Matx33d W(0, -1, 0,
			  1, 0, 0,
			  0, 0, 1);//HZ 9.13
	Mat_<double> R1 = svd.u * Mat(W) * svd.vt;//R1
	Mat_<double> r2;
	Rodrigues(R1, r2);

	//cout <<"the second camera's R is \n"<< r2<<endl;
	if (!CheckCoherentRotation(R1))
		return 0;
	Mat_<double> u3 = -svd.u.col(2); //u3
	//cout <<"the second camera's t is \n"<< u3<<endl;
	Matx34d P(1,0,0,0,
			  0,1,0,0,
			  0,0,1,0);//first (original) camera (R|t)
	Matx34d P1( R1(0, 0), R1(0, 1), R1(0, 2), u3(0),
				R1(1, 0), R1(1, 1), R1(1, 2), u3(1),
				R1(2, 0), R1(2, 1), R1(2, 2), u3(2) );//second camera (R|t)
	Mat M = K * Mat(P);//first (original) camera matrix
	Mat M2 = K * Mat(P1);//second (original) camera matrix
/************************************* end 1st & 2nd image *****************************/


	      
			  
	//Triangulation for image 1 & 2
	Mat points3D_12_H;//the triangulated homogeneous coords
	triangulatePoints(M, M2, trackedPoint_1, trackedPoint_2, points3D_12_H);
	//store to txt file
	FileStorage file("points3D_12_H.txt", FileStorage::WRITE);
	file<< "points3D_12_H" << points3D_12_H.t();
	file.release();
		FileStorage filex("points3D_12_H.xml", FileStorage::WRITE);
	filex<< "points3D_12_H" << points3D_12_H.t();
	filex.release();
	//normalization
	Mat points3D_12(3, points3D_12_H.cols, CV_32FC1);//3D world coords
	normalization3D(points3D_12_H, points3D_12);//normalize and the 3D coords
	//store to txt file
	FileStorage file2("points3D_12.txt", FileStorage::WRITE);
	file2<< "points3D_12" << points3D_12.t();
	file2.release();


	


	waitKey(0);
	return 0;
}//end main



