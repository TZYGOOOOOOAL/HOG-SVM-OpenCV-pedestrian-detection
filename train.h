# ifndef __TRAIN_H__
# define __TRAIN_H__


#include <iostream>
#include <string>
#include <vector>
#include <opencv.hpp>

using namespace std;
using namespace cv;

void train();
void train_hard();
void create_feature_and_label_matrix(vector<string> &img_paths, Mat &feature_mat, Mat &label_mat, int label_val);

void get_hard_examples(Ptr<ml::SVM> &svm, vector<string> &neg_img_paths, Mat &hard_feature_mat, Mat &hard_label_mat, bool save);
# endif
