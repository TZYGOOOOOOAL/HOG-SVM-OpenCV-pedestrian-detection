# ifndef __EVAL_H__
# define __EVAL_H__

#include <iostream>
#include <string>
#include <vector>
#include <opencv.hpp>

using namespace std;
using namespace cv;

void eval_valid(Ptr<ml::SVM> &svm = ml::SVM::create());

void eval_classify(Mat &predicts, Mat &labels);
void eval_detect(const vector<vector<Rect>> &predicts, const vector<vector<Rect>> &labels, const double &iou_th);
void eval_detect_one_img(const vector<Rect> &predict_bboxes, const vector<Rect> &label_bboxes, const double &iou_th, int &TP, double &p);
# endif