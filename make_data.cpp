#include "make_data.h"
#include "utils.h"
#include "Config.h"
extern Config config;

void make_data()
{
	// 制作训练样本
	_make_data(config.TRAIN_POS_RAW_IMG_DIR, config.TRAIN_POS_IMG_DIR,
		config.TRAIN_NEG_RAW_IMG_DIR, config.TRAIN_NEG_IMG_DIR, config.PATCH_SIZE,
		config.POS_NEG_RATIO);

	// 制作验证样本
	_make_data(config.VALID_POS_RAW_IMG_DIR, config.VALID_POS_IMG_DIR,
		config.VALID_NEG_RAW_IMG_DIR, config.VALID_NEG_IMG_DIR, config.PATCH_SIZE,
		config.POS_NEG_RATIO);

}

void _make_data(string pos_raw_img_dir, string pos_img_save_dir,
	string neg_raw_img_dir, string neg_img_save_dir, 
	Size patch_size, float pos_neg_ratio)
{
	// 制作正样本
	if (!is_dir(pos_img_save_dir))
		make_dir(pos_img_save_dir);

	vector<string> pos_files = get_child_files(pos_raw_img_dir);
	crop_positive_images(pos_files, patch_size, pos_img_save_dir);
	int num_pos = pos_files.size();

	// 制作负样本 
	if (!is_dir(neg_img_save_dir))
		make_dir(neg_img_save_dir);
	int num_neg = pos_neg_ratio * num_pos;
	vector<string> neg_files = get_child_files(neg_raw_img_dir);
	int num_patch_per_neg_img = num_neg / neg_files.size();
	random_split_negative_images(neg_files, patch_size, num_patch_per_neg_img, neg_img_save_dir);
	
	return;
}

// 中心裁剪正样本
void crop_positive_images(const vector<string> &img_paths,
	Size patch_size, string save_dir)
{
	for (int i = 0; i < img_paths.size(); i++)
	{
		string img_path = img_paths[i];
		Mat img = imread(img_path);
		crop_one_positive_image(img, img_path, patch_size, save_dir);
	}
	return;
}

// 中心裁剪单幅正样本图像
void crop_one_positive_image(Mat &img, string img_path,
	Size patch_size, string save_dir)
{
	if (img.empty())
		img = imread(img_path);
	
	if (img.cols < patch_size.width || img.rows < patch_size.height)
	{
		cout << img_path + " too small to crop !!!" << endl;
		return;
	}

	int x_start = int(img.cols / 2.0 - patch_size.width / 2.0);
	int y_start = int(img.rows / 2.0 - patch_size.height / 2.0);

	Mat patch;
	img(Rect(x_start, y_start, patch_size.width, patch_size.height)).copyTo(patch);

	string save_path = path_join(save_dir, get_filename(img_path)) + ".jpg";

	imwrite(save_path, patch);
	cout << "positive sample save path: " + save_path << endl;
	return;
}


// 剪裁负样本
void random_split_negative_images(const vector<string> &img_paths, 
	Size patch_size, int num_patch_per_img, string save_dir)
{
	for (int i = 0; i < img_paths.size(); i++)
	{
		string img_path = img_paths[i];
		Mat img = imread(img_path);
		random_split_one_negative_image (img, img_path, patch_size, num_patch_per_img, save_dir);
	}
}

// 剪裁单个负样本图像
void random_split_one_negative_image(Mat &img, string img_path,
	Size patch_size, int num_patch_per_img, string save_dir)
{
	srand((unsigned)time(NULL));
	if (img.empty())
		img = imread(img_path);

	int x_min = 0, x_max = img.cols - patch_size.width;
	int y_min = 0, y_max = img.rows - patch_size.height;
	int x, y;
	Mat patch(patch_size.height, patch_size.width, CV_8UC3);

	for (int i = 0; i < num_patch_per_img; i++)
	{
		x = (rand() % (x_max - x_min + 1)) + x_min;
		y = (rand() % (y_max - y_min + 1)) + y_min;

		img(Rect(x, y, patch_size.width, patch_size.height)).copyTo(patch);

		string save_path = get_neg_save_path(img_path, x, y);
		save_path = path_join(save_dir, save_path);
		imwrite(save_path, patch);
		cout << "negative sample save path: " + save_path << endl;
	}

	return;
}

string get_neg_save_path(string img_path, int x, int y)
{
	return get_filename(img_path) + "_" + to_string(x) + "_" + to_string(y) + ".jpg";
}