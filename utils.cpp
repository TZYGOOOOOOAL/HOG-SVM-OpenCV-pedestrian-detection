#include "utils.h"



// 文件存在
bool is_exits(string path)
{
	return !bool(_access(path.c_str(), 0));
}

// 创建1级目录
bool make_dir(string dir_path)
{
	if (is_exits(dir_path) || is_file(dir_path))
		return false;
	_mkdir(dir_path.c_str());
	return true;
}

// 路径拼接
// https://blog.csdn.net/jiratao/article/details/9764679?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.control
string path_join(string path1, string path2)
{
	char path_buffer[_MAX_PATH];
	_makepath(path_buffer, NULL, path1.c_str(), path2.c_str(), NULL);
	return string(path_buffer);
}


static string my_split_path(const string &path, string mode)
{
	char drive[_MAX_DRIVE];
	char dir[_MAX_DIR];
	char _fname[_MAX_FNAME];
	char _ext[_MAX_EXT];

	_splitpath(path.c_str(), drive, dir, _fname, _ext);

	if (mode == "fname")
		return string(_fname);
	if (mode == "ext")
		return string(_ext);
	if (mode == "drive")
		return string(drive);
	if (mode == "dir")
		return string(drive) + dir;
	if (mode == "basename")
		return string(_fname) + _ext;
	return path;
}

// 后缀
string get_ext(string path){
	return my_split_path(path, "ext");
}

// 文件名
string get_filename(string path){
	return my_split_path(path, "fname");
}

// basename
string get_basename(string path){
	return my_split_path(path, "basename");
}

// dir name
string get_dirname(string path){
	return my_split_path(path, "dir");
}

// 是文件
bool is_file(string path)
{
	return is_exits(path) && !get_ext(path).empty();
}

// 是目录
bool is_dir(string path)
{
	return is_exits(path) && get_ext(path).empty();
}

// 所有文件
vector<string> get_all_files(string path, vector<string> formats)
{
	vector<string> file_paths = get_child_files(path, formats);
	vector<string> dir_paths = get_child_dirs(path);

	// 递归查找所有子目录
	vector<string> temp_paths;
	for (int i = 0; i < dir_paths.size(); i++)
	{
		temp_paths = get_all_files(dir_paths[i], formats);
		file_paths.insert(file_paths.end(), temp_paths.begin(), temp_paths.end());
	}

	return file_paths;
}


// 目录下所有子文件
vector<string> get_child_files(string path, vector<string> formats)
{
	if (formats.empty())
		formats.emplace_back("");

	//文件句柄    
	intptr_t hFile = 0;
	vector<string> file_paths;

	//文件信息    
	struct _finddata_t fileinfo;
	string p;

	// 查找
	for (int i = 0; i < formats.size(); i++)
	{
		string format = formats[i];
		if ((hFile = _findfirst(p.assign(path).append(string("/*") + format).c_str(), &fileinfo)) != -1)
		{
			do
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					file_paths.push_back(p.assign(path).append("/").append(fileinfo.name));

			} while (_findnext(hFile, &fileinfo) == 0);

			_findclose(hFile);
		}
	}
	
	return file_paths;
}


vector<string> get_child_dirs(string path)
{
	//文件句柄    
	intptr_t  hFile = 0;
	vector<string> dir_paths;

	//文件信息    
	struct _finddata_t fileinfo;
	string p;

	// 所有文件和目录
	if ((hFile = _findfirst(p.assign(path).append("/*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			// 是目录
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				{
					dir_paths.emplace_back(p.assign(path).append("/").append(fileinfo.name));
				}
			}
		} while (_findnext(hFile, &fileinfo) == 0);

		_findclose(hFile);
	}

	return dir_paths;
}

// 保存矩阵数值
void save_mat_data(Mat &m, string save_path)
{
	FileStorage fs(save_path, FileStorage::WRITE);
	fs << "data" << m;
}

Mat load_mat_data(string load_path)
{
	FileStorage fs(load_path, FileStorage::READ);
	Mat m;
	fs["data"] >> m;
	return m;
}

// 解析标注文件
vector<vector<Rect>> parse_annotations(const vector<string> &img_paths, const string &annotation_dir)
{
	vector<vector<Rect>> bboxes;
	for (int i = 0; i < img_paths.size(); i++)
	{
		string annotation_path = path_join(annotation_dir, get_filename(img_paths[i]) + ".txt");
		assert (is_file(annotation_path));
		bboxes.emplace_back(parse_one_annotation(annotation_path));
	}
		
	return bboxes;
}

vector<Rect> parse_one_annotation(const string &annotation_path)
{
	assert(is_file(annotation_path));

	vector<Rect> v_rect;

	std::ifstream input(annotation_path);
	string dataline;
	while (!input.eof())
	{
		// 获取一行
		getline(input, dataline);
		if (dataline.find("Bounding box") == 0)
		{
			int start_pos = dataline.find(": (");
			int xy[4] = { 0, 0, 0, 0 };
			int idx = start_pos + 3;
			char c = dataline[idx];

			for (int xy_idx = 0; xy_idx < 4; xy_idx++)
			{
				// 提取坐标数字
				while (c >= '0' && c <= '9')
				{
					xy[xy_idx] *= 10;
					xy[xy_idx] += c - '0';
					idx++;
					c = dataline[idx];
				}

				// 跳过非数字
				while (c < '0' || c > '9')
				{
					if (idx == dataline.size() - 1)
						break;
					idx++;
					c = dataline[idx];
				}
			}

			// 转换为Rect
			//cout << xy[0] << ' ' << xy[1] << ' ' << xy[2] << ' ' << xy[3] << endl;
			v_rect.emplace_back(Rect(xy[0], xy[1], xy[2] - xy[0] + 1, xy[3] - xy[1] + 1));
		}
	}
	return v_rect;
}


// 可视化
Mat visual_bboxes(Mat &img, vector<Rect> &bboxes, vector<double> confs,
	Scalar &color, int &thickness, bool show_now)
{
	Mat img_new = img.clone();

	for (int i = 0; i < bboxes.size(); i++)
	{
		// 画框
		rectangle(img_new, bboxes[i], color, thickness);

		// 标记置信度
		if (!confs.empty())
		{
			string conf_text = to_string(confs[i]);
			conf_text.erase(conf_text.begin() + 4, conf_text.end());
			Size text_box = cv::getTextSize(conf_text, FONT_HERSHEY_SIMPLEX, 0.6, thickness, nullptr);
			cv::putText(img_new, conf_text, Point2i(bboxes[i].x, bboxes[i].y - int(0.3 * text_box.height)),
				FONT_HERSHEY_SIMPLEX, 0.6, color, thickness);
		}
	}

	if (show_now)
	{
		imshow("result", img_new);
		cv::waitKey();
		destroyWindow("result");
	}

	return img_new;
}

// 获得不重复保存路径
string get_no_repeat_save_path(string save_path)
{
	if (is_file(save_path))
		cout << "WARNING : save_path is Exist !!!" << endl;
	else
		return save_path;

	string dir = get_dirname(save_path);
	string ext = get_ext(save_path);
	string filename = get_filename(save_path);
	
	while (is_file(save_path))
	{
		save_path = path_join(dir, filename + "_" + to_string(rand()) + ext);
	}

	return save_path;
}


// 加载svm模型
cv::Ptr<ml::SVM> load_svm(string svm_path)
{
	Ptr<ml::SVM> svm = ml::SVM::load<ml::SVM>(svm_path);
	return svm;
}

// svm距离转置信度
double cvt_to_conf(const double &dist)
{
	return (1.0 / (1.0 + exp(-dist)));
}

// iou
double bbox_iou(const Rect &bbox1, const Rect &bbox2)
{
	Rect I = bbox1 & bbox2;
	Rect U = bbox1 | bbox2;
	return I.area() / (U.area() + 1e-12);
}

// 
vector<int> argsort_d(const vector<double>& v)
{
	vector<int> idxs(v.size(), 0);
	for (int i = 0; i < v.size(); ++i)
		idxs[i] = i;

	std::sort(idxs.begin(), idxs.end(),
		[&v](int pos1, int pos2) {return (v[pos1] > v[pos2]); });

	return idxs;
}

Timer::Timer()
{
	start_time = double(clock());
}

double Timer::get_run_time(string desc, bool reset, bool show)
{
	double run_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
	if (show)
		cout << desc + " run time = " << run_time << "(s)" << endl;
	if (reset)
		start_time = double(clock());
	return run_time;
}

void Timer::reset()
{
	start_time = double(clock());
	return;
}
