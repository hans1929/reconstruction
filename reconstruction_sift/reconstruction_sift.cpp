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
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace cv;
using namespace std;

//function check valid rotation matrix
bool CheckCoherentRotation(Mat_<double>& R) {
	if (fabsf(determinant(R)) - 1.0 > 1e-07) {
		cerr << "det(R) != +-1.0, this is not a rotation matrix" << endl;
		return false;
	}
	return true;
}//end function check valid rotation matrix

//function normalization
void normalization3D(Mat point3d_H, Mat & point3d)
{
	//normalization
	point3d_H.convertTo(point3d_H,CV_32F);
	point3d_H.row(0) = point3d_H.row(0) / point3d_H.row(3);
	point3d_H.row(1) = point3d_H.row(1) / point3d_H.row(3);
	point3d_H.row(2) = point3d_H.row(2) / point3d_H.row(3);
	point3d_H.row(0).copyTo(point3d.row(0));
	point3d_H.row(1).copyTo(point3d.row(1));
	point3d_H.row(2).copyTo(point3d.row(2));
}//end function triangulate

/**************************************** main function *********************************/
int main( int argc, char** argv )
{
	//iphone's intrinsic parameters
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
    //Mat img_1 = imread( "tsukuba_l.png", CV_LOAD_IMAGE_GRAYSCALE );
    //Mat img_2 = imread( "tsukuba_r.png", CV_LOAD_IMAGE_GRAYSCALE );
	imshow("original_left", img_1 );
	imshow("original_right", img_2 );
	//if invalid read
	if( !img_1.data || !img_2.data )
	{ return -1; }

	//detect interesting points and point matches
	vector<KeyPoint> keyPoints_1, keyPoints_2;
    //Detect the keypoints using SIFT
    SiftFeatureDetector detector;
	detector.detect( img_1, keyPoints_1 );
	detector.detect( img_2, keyPoints_2 );
    cout<<"the size of keypoint_1 is: "<<keyPoints_1.size()<<endl;
    cout<<"the size of keypoint_2 is: "<<keyPoints_2.size()<<endl;
    //Calculate descriptors (feature vectors, each is 1*128)
	SiftDescriptorExtractor extractor;
	Mat descriptors_1, descriptors_2;
	extractor.compute( img_1, keyPoints_1, descriptors_1 );
	extractor.compute( img_2, keyPoints_2, descriptors_2 );
    
    //Matching descriptor vectors with a brute force matcher
	vector< DMatch > matches_12;
    BFMatcher matcher(NORM_L2);
	matcher.match( descriptors_1, descriptors_2, matches_12 );
    cout<<"the size of matches_12 is: "<<matches_12.size()<<endl;
    
    //convert matched keypoints to point2f
    vector<Point2f> matchPoints1, matchPoints2;
    for (vector<cv::DMatch>::const_iterator it= matches_12.begin(); it!= matches_12.end(); ++it)
	{
		// Get the position of the keypoints of img1
		float x= keyPoints_1[it->queryIdx].pt.x;
		float y= keyPoints_1[it->queryIdx].pt.y;
		matchPoints1.push_back(cv::Point2f(x,y));
		// Get the position of keypoints of img2
		x= keyPoints_2[it->trainIdx].pt.x;
		y= keyPoints_2[it->trainIdx].pt.y;
		matchPoints2.push_back(cv::Point2f(x,y));
	}// now matchPoints2 and matchPoints2 matches in every entry
    cout <<"The interesting points detection is done\nThe number of matched keypoints of img1 & img2 is"<< matchPoints1.size() << endl; 

    //estimate fundamental matrix using RANSAC
	// Compute F matrix using RANSAC
	vector<uchar> inliers_12(matchPoints1.size(), 0);
	Mat F_12 = cv::findFundamentalMat(
		Mat(matchPoints1), Mat(matchPoints2), // matching points
		inliers_12,      // match status (inlier ou outlier)  
		CV_FM_RANSAC, // RANSAC method
		1,            // distance to epipolar line
		0.99);        // confidence probability

	// extract the surviving (inliers) good matches
	vector<DMatch> good_matches_12;
    vector<uchar>::const_iterator it_inliers = inliers_12.begin();
    vector<DMatch>::const_iterator it_matches= matches_12.begin();
    for (; it_inliers != inliers_12.end(); ++it_inliers, ++it_matches) {
        if (*it_inliers)
		{ // it is a valid match
			good_matches_12.push_back(*it_matches);
		}
    }

    //convert the good match keypoints to the format of Point2f
	vector <Point2f> good_matchPoints1, good_matchPoints2;
    for (vector<cv::DMatch>::const_iterator it= good_matches_12.begin(); it!= good_matches_12.end(); ++it)
	{
		// Get the position of the keypoints of img1
		float x= keyPoints_1[it->queryIdx].pt.x;
		float y= keyPoints_1[it->queryIdx].pt.y;
		good_matchPoints1.push_back(cv::Point2f(x,y));
		// Get the position of keypoints of img2
		x= keyPoints_2[it->trainIdx].pt.x;
		y= keyPoints_2[it->trainIdx].pt.y;
		good_matchPoints2.push_back(cv::Point2f(x,y));
	}// now good_matchPoints1 and good_matchPoints2 matches in every entry
    cout << "After using RANSAC, the number of good matched points between image 1 & 2 (after cleaning) is: " << good_matches_12.size() << std::endl;

    //Draw good matches
    Mat img_matches12; //the image that shows the matches
    drawMatches( img_1, keyPoints_1, img_2, keyPoints_2, good_matches_12, img_matches12 );
    imshow("Good_Matches12", img_matches12 );  //show good matches after ransac

    cout<<"The fundamental matrix F is\n"<<F_12<<endl;



    //Using partial camera calibration to find camera matrix
	Mat E = K.t() * F_12 * K;//essential matrix
	cout << "The essential matrix is:\n"<< E << endl;
	SVD svd(E);
	Matx33d W(0, -1, 0,
			  1, 0, 0,
			  0, 0, 1);//HZ result9.19
	Mat_<double> R2 = svd.u * Mat(W) * svd.vt;//R2
	//Mat_<double> r2;
	//Rodrigues(R2, r2);
    cout <<"the second camera's Rotation matrix R is \n"<< R2<<endl;
	if (!CheckCoherentRotation(R2))
		return 0;
	Mat_<double> u2 = -svd.u.col(2); //u2
	//cout <<"the second camera's Translation matrix t is \n"<< u2<<endl;
	Matx34d P1(1,0,0,0,
			  0,1,0,0,
			  0,0,1,0);//first (original) camera (R|t == I|0)
	Matx34d P2( R2(0, 0), R2(0, 1), R2(0, 2), u2(0),
				R2(1, 0), R2(1, 1), R2(1, 2), u2(1),
				R2(2, 0), R2(2, 1), R2(2, 2), u2(2) );//second camera (R|t == R2|u2)
	Mat M = K * Mat(P1);//first (original) camera matrix
	Mat M2 = K * Mat(P2);//second (original) camera matrix


    //Triangulation for image 1 & 2
	Mat points3D_12_H;//the triangulated homogeneous coords
	triangulatePoints(M, M2, good_matchPoints1, good_matchPoints2, points3D_12_H);
	//store to txt file
/*	FileStorage file("points3D_12_H.txt", FileStorage::WRITE);
	file<< "points3D_12_H" << points3D_12_H.t();
	file.release();
		FileStorage filex("points3D_12_H.xml", FileStorage::WRITE);
	filex<< "points3D_12_H" << points3D_12_H.t();
	filex.release();*/
	//normalization
	Mat points3D_12(3, points3D_12_H.cols, CV_32FC1);//3D world coords
	normalization3D(points3D_12_H, points3D_12);//normalize and the 3D coords (dim: 3 * cols)
    //store to txt file
	FileStorage file2("points3D_12.txt", FileStorage::WRITE);
	file2<< "points3D_12" << points3D_12.t();
	file2.release();

    

    //convert to pcl format points
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    cloud->width    = points3D_12.cols;
    cloud->height   = 1;
    cloud->is_dense = false;
    cloud->points.resize (cloud->width * cloud->height);
    cout<<"the size of the pcl points is: "<<cloud->points.size()<<endl;
    for (size_t i = 0; i < cloud->points.size (); ++i)
    {
        cloud->points[i].x = points3D_12.at<float>(0, i);
        cloud->points[i].y = points3D_12.at<float>(1, i);
        cloud->points[i].z = points3D_12.at<float>(2, i);
    }
    //save to pcl format pcd
    pcl::io::savePCDFileASCII ("test_pcd.pcd", *cloud);
    std::cerr << "Saved " << cloud->points.size () << " data points to test_pcd.pcd." << std::endl;

    //visualize points using pcl
    //pcl::io::loadPCDFile ("test_pcd.pcd", *cloud);
    pcl::visualization::CloudViewer viewer("Cloud Viewer");
    viewer.showCloud(cloud);
    

    waitKey(0);
    return 0;
}