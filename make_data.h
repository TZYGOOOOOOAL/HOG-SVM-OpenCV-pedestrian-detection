# ifndef __MAKE_DATA_H__
# define __MAKE_DATA_H__

#include <iostream>
#include <string>
#include <vector>
#include <opencv.hpp>
#include <cstdlib>
#include <ctime>

using namespace std;
using namespace cv;

void make_data();
void _make_data(string pos_raw_img_dir, string pos_img_save_dir, string neg_raw_img_dir, string neg_img_save_dir, Size patch_size, float pos_neg_ratio);
void crop_positive_images(const vector<string> &img_paths, Size patch_size, string save_dir);
void crop_one_positive_image(Mat &img, string img_path, Size patch_size, string save_dir);
void random_split_negative_images(const vector<string> &img_paths, Size patch_size, int num_patch_per_img, string save_dir);
void random_split_one_negative_image(Mat &img, string img_path, Size patch_size, int num_patch_per_img, string save_dir);
string get_neg_save_path(string img_path, int x, int y);

# endif
