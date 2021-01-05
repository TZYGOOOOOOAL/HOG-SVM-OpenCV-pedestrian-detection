# ifndef __UTILS_H__
# define __UTILS_H__

#include <opencv.hpp>
#include <direct.h>
#include <io.h>
#include <fstream>
#include <string>
#include <vector>
#include <ctime>

using namespace std;
using namespace cv;

class Timer{
private:
	double start_time;
public:
	Timer();
	double get_run_time(string desc="", bool reset=false, bool show=true);
	void reset();
};


bool is_exits(string path);
bool make_dir(string dir_path);
string path_join(string path1, string path2);
static string my_split_path(const string &path, string mode);
string get_ext(string path);
string get_filename(string path);
string get_basename(string path);
string get_dirname(string path);
bool is_file(string path);
bool is_dir(string path);

vector<string> get_all_files(string path, vector<string> formats = vector<string>());
vector<string> get_child_files(string path, vector<string> formats = vector<string>());
vector<string> get_child_dirs(string path);
void save_mat_data(Mat &m, string save_path);
Mat load_mat_data(string load_path);
vector<vector<Rect>> parse_annotations(const vector<string> &img_paths, const string &annotation_dir);
vector<Rect> parse_one_annotation(const string &annotation_path);
Mat visual_bboxes(Mat &img, vector<Rect> &bboxes, vector<double> confs, Scalar &color, int &thickness, bool show_now = false);
string get_no_repeat_save_path(string save_path);
cv::Ptr<ml::SVM> load_svm(string svm_path);
double cvt_to_conf(const double &dist);
double bbox_iou(const Rect &bbox1, const Rect &bbox2);
vector<int> argsort_d(const vector<double>& v);

# endif