# ifndef __MY_HOG_H__
# define __MY_HOG_H__

#include <iostream>
#include <string>
#include <vector>
#include <opencv.hpp>

using namespace std;
using namespace cv;

cv::HOGDescriptor get_hog_desriptor(Size patch_size, Size block_size, Size block_stride, Size cell_size, int hist_bins);
vector<float> get_one_patch_hog_features(Mat &img, HOGDescriptor& hd, Size win_stride);
vector<float> convert_svm_detector(const Ptr<ml::SVM>& svm);
# endif
