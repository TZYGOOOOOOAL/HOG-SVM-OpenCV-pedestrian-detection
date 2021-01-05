# ifndef __TEST_H__
# define __TEST_H__


#include <iostream>
#include <string>
#include <vector>
#include <opencv.hpp>

using namespace std;
using namespace cv;

void test();
void test_one_image_on_my_method(Mat &img, HOGDescriptor &hd, Ptr<ml::SVM> &svm, vector<Rect> &bboxes, 
	vector<double> &confs, const Size &step_size, const Size &pad_size, const double &det_scale, const double &conf_th, const double &nms_th);
void test_one_image_on_one_scale(Mat &img, HOGDescriptor &hd, Ptr<ml::SVM> &svm, vector<Rect> &bboxes, vector<double> &confs, 
	const Size &step_size, const Size &pad_size);
void test_one_image_on_cv_method(Mat &img, HOGDescriptor &hd, vector<Rect> &bboxes, vector<double> &confs,
	const Size &step_size, const Size &pad_size, const double &det_scale, const double &conf_th, const double &nms_th);
void nms(vector<Rect> &bboxes, vector<double> &confs, const double &conf_th = 0.5, const double &nms_th = 0.5);
# endif
