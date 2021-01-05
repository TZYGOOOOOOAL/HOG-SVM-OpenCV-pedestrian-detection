#include "test.h"
#include "Config.h"
#include "utils.h"
#include "my_hog.h"
#include "eval.h"

extern Config config;

void test()
{
	Timer timer;

	/*** 获取数据 ***/
	// 图像数据
	vector<string> img_paths = get_child_files(config.TEST_IMG_DIR);
	//vector<string> img_paths({ "data/test/crop_000002.png" });

	// 标签
	vector<vector<Rect>> labels;
	if (config.TEST_WITH_LABEL)
		labels = parse_annotations(img_paths, config.TEST_TXT_DIR);

	// hog 检测器
	HOGDescriptor hd = get_hog_desriptor(config.PATCH_SIZE, config.HOG_BLOCK_SIZE, config.HOG_BLOCK_STRIDE,
		config.HOG_CELL_SIZE, config.HOG_BINS);

	// 所有结果
	vector<vector<Rect>> predicts;

	/************************* 自己实现检测方法 ******************************/
	if (!config.USE_CV_METHOD)
	{
		// 加载SVM模型
		if (!is_file(config.SVM_MODEL_PATH))
		{
			cerr << "SVM model NOT Exits !!!";
			return;
		}
		Ptr<ml::SVM> svm = load_svm(config.SVM_MODEL_PATH);

		// 遍历每幅图像
		timer.reset();
		for (int i = 0; i < img_paths.size(); i++)
		{
			//if (get_basename(img_paths[i]) != "crop_000002.png")
			//	continue;
			vector<Rect> bboxes;
			vector<double> confs;
			Mat img = imread(img_paths[i]);

			cout << "Test " << img_paths[i] << endl;
			test_one_image_on_my_method(img, hd, svm, bboxes, confs, config.DETECT_STEP,
				config.DETECT_PAD, config.DETECT_SCALE, config.DETECT_CONF_TH, config.DETECT_NMS_TH);
			predicts.push_back(bboxes);

			// 可视化
			if (config.VISUAL_TEST_RESULT || config.SAVE_RESULT)
			{
				if (config.TEST_WITH_LABEL)
					img = visual_bboxes(img, labels[i], vector<double>(), config.VISUAL_LABEL_COLOR, config.VISUAL_THICKNESS, false);
				img = visual_bboxes(img, bboxes, confs, config.VISUAL_PREDICT_COLOR, config.VISUAL_THICKNESS, config.VISUAL_TEST_RESULT);

				if (config.SAVE_RESULT)
				{
					make_dir(config.RESULT_SAVE_DIR);
					string result_save_path = path_join(config.RESULT_SAVE_DIR, get_filename(img_paths[i]) + ".jpg");
					cv::imwrite(result_save_path, img);
				}
			}
		}

		double total_time = timer.get_run_time("", false, false);
		cout << "my method test time per img : " << total_time / img_paths.size() << "(s)" << endl;
	}

	/*************************** OpenCV自带方法 ********************************/
	else
	{
		cout << "Loading Test Model ..." << endl;
		// 加载检测模型
		if (config.USE_DEFAULT_MODEL){
			hd.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
		}
		else
		{
			// 加载已转换好的，若无则make新的
			//if (is_file(config.HOG_DETECT_MODEL_PATH))
				//hd.load(config.HOG_DETECT_MODEL_PATH);
			//else
			{
				if (is_file(config.SVM_MODEL_PATH))
				{
					Ptr<ml::SVM> svm = load_svm(config.SVM_MODEL_PATH);

					// OpenCV只支持线性核
					if (svm->getKernelType() != ml::SVM::KernelTypes::LINEAR)
					{
						cerr << "SVM NOT Linear Kernel  !!!";
						return;
					}
					vector<float> hog_detector = convert_svm_detector(svm);
					hd.setSVMDetector(hog_detector);

					// 保存转换结果
					hd.save(config.HOG_DETECT_MODEL_PATH);
				}

				// 模型路径不存在
				else
				{
					cerr << "SVM model :" << config.SVM_MODEL_PATH << " Not Exist !!!";
					return;
				}

			}
		}

		// 遍历每幅图像
		timer.reset();
		cout << "Start Testing !!!" << endl;
		for (int i = 0; i < img_paths.size(); i++)
		//labels.erase(labels.begin() + 10, labels.end());
		//for (int i = 0; i < 10; i++)
		{
			vector<Rect> bboxes;
			vector<double> confs;
			Mat img = imread(img_paths[i]);

			cout << "Test " << img_paths[i] << endl;
			test_one_image_on_cv_method(img, hd, bboxes, confs, config.DETECT_STEP,
				config.DETECT_PAD, config.DETECT_SCALE, config.DETECT_CONF_TH, config.DETECT_NMS_TH);
			predicts.push_back(bboxes);
				
			// 可视化
			if (config.VISUAL_TEST_RESULT || config.SAVE_RESULT)
			{
				if (config.TEST_WITH_LABEL)
					img = visual_bboxes(img, labels[i], vector<double>(), config.VISUAL_LABEL_COLOR, config.VISUAL_THICKNESS, false);
				img = visual_bboxes(img, bboxes, confs, config.VISUAL_PREDICT_COLOR, config.VISUAL_THICKNESS, config.VISUAL_TEST_RESULT);

				if (config.SAVE_RESULT)
				{
					make_dir(config.RESULT_SAVE_DIR);
					string result_save_path = path_join(config.RESULT_SAVE_DIR, get_filename(img_paths[i]) + ".jpg");
					cv::imwrite(result_save_path, img);
				}
			}
				
		}
		double total_time = timer.get_run_time("", false, false);
		cout << "opencv detect test time per img : " << total_time / img_paths.size() << "(s)" << endl;
	}

	/*** 评价结果 ***/
	if (config.TEST_WITH_LABEL)
	{
		eval_detect(predicts, labels, config.DETECT_IOU_TH);
	}
}


// 自己实现的方法
void test_one_image_on_my_method(Mat &img, HOGDescriptor &hd, Ptr<ml::SVM> &svm, vector<Rect> &bboxes, vector<double> &confs,
	const Size &step_size, const Size &pad_size, const double &det_scale, const double &conf_th, const double &nms_th)
{
	// 最大尺度
	double max_scale = pow(det_scale, 10);

	// 最多20个尺度
	double s = max_scale;
	for (int i = 0; i < 20; i++)
	{ 
		s /= det_scale;

		if (int(img.rows*s) < hd.winSize.height || int(img.cols*s) < hd.winSize.width)
			break;

		Mat scale_img;
		resize(img, scale_img, Size(0, 0), s, s);

		vector<Rect> one_img_bboxes;
		vector<double> one_img_confs;

		test_one_image_on_one_scale(scale_img, hd, svm, one_img_bboxes, one_img_confs, step_size, pad_size);

		// bboxes 还原到原尺度上
		for (int bbox_idx = 0; bbox_idx < one_img_bboxes.size(); bbox_idx++)
		{
			one_img_bboxes[bbox_idx].x = int(one_img_bboxes[bbox_idx].x / s) - pad_size.width;
			one_img_bboxes[bbox_idx].y = int(one_img_bboxes[bbox_idx].y / s) - pad_size.height;
			one_img_bboxes[bbox_idx].width = int(one_img_bboxes[bbox_idx].width / s);
			one_img_bboxes[bbox_idx].height = int(one_img_bboxes[bbox_idx].height / s);
		}

		bboxes.insert(bboxes.end(), one_img_bboxes.begin(), one_img_bboxes.end());
		confs.insert(confs.end(), one_img_confs.begin(), one_img_confs.end());
	}

	// 后处理
	nms(bboxes, confs, conf_th, nms_th);

	return;
}

// 单尺度预测
void test_one_image_on_one_scale(Mat &img, HOGDescriptor &hd, Ptr<ml::SVM> &svm,
	vector<Rect> &bboxes, vector<double> &confs,const Size &step_size,const Size &pad_size)
{
	int pad_v = pad_size.height;
	int pad_h = pad_size.width;
	cv::copyMakeBorder(img, img, pad_v, pad_v, pad_h, pad_h, cv::BORDER_REFLECT101);

	int win_h = hd.winSize.height;
	int win_w = hd.winSize.width;
	int step_h = step_size.height;
	int step_w = step_size.width;
	Mat feature_mat, predict_mat;

	vector<vector<float>> features;
	vector<Rect> ROI_bboxes;

	for (int i = 0; i < img.rows - win_h; i += step_h)
	{
		for (int j = 0; j < img.cols - win_w; j += step_w)
		{
			Rect ROI_rect(j, i, win_w, win_h);
			Mat ROI = img(ROI_rect).clone();
			ROI_bboxes.push_back(ROI_rect);
			features.emplace_back(get_one_patch_hog_features(ROI, hd, Size(8,8)));
		}
	}
	
	// 构建数据矩阵
	feature_mat = Mat(features.size(), features[0].size(), CV_32FC1);
	for (int i = 0; i < features.size(); i++)
	{
		for (int j = 0; j < features[i].size(); j++)
			feature_mat.at<float>(i, j) = features[i][j];
	}

	// 预测 输出原始值（到决策面距离）
	svm->predict(feature_mat, predict_mat, ml::StatModel::Flags::RAW_OUTPUT);

	// 将所有大于0的样本（正样本）转换置信度，并保存
	for (int i = 0; i < predict_mat.rows; i++)
	{
		double dist = (double)(predict_mat.at<float>(i, 0));
		if (dist < 0)  // 有坑 < 0 才是正样本
		{
			confs.push_back(cvt_to_conf(-dist));
			bboxes.push_back(ROI_bboxes[i]);
			//Mat ROI = img(ROI_bboxes[i]).clone();
			//imshow("", ROI);
			//waitKey();
		}
	}
	return;
}


void test_one_image_on_cv_method(Mat &img, HOGDescriptor &hd, vector<Rect> &bboxes, vector<double> &confs, 
	const Size &step_size, const Size &pad_size, const double &det_scale, const double &conf_th, const double &nms_th)
{
	// scale 表示 图像不断缩小 1/scale 直至图像大小 <= 窗口大小（上限64次迭代）
	// 3: 到边界距离， 4：步长，5：pad，6：scale
	hd.detectMultiScale(img, bboxes, confs, 0.0, step_size, pad_size, det_scale);

	// 转换置信度
	for (int i = 0; i < bboxes.size(); i++)
		confs[i] = cvt_to_conf(confs[i]);

	// nms 后处理
	nms(bboxes, confs, conf_th, nms_th);
	return;
}
//
//int main()
//{
//	Mat image = imread("test.jpg");
//	// 1. 定义HOG对象
//	HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);//HOG检测器，用来计算HOG描述子的
//	// 2. 设置SVM分类器
//	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());   // 采用已经训练好的行人检测分类器
//	// 3. 在测试图像上检测行人区域
//	vector<cv::Rect> regions;
//	hog.detectMultiScale(image, regions, 0, Size(8, 8), Size(32, 32), 1.05, 1);
//	// 显示
//	for (size_t i = 0; i < regions.size(); i++)
//	{
//		rectangle(image, regions[i], Scalar(0, 0, 255), 2);
//	}
//	imshow("HOG行人检测", image);
//	waitKey();
//
//	return 0;


//float distanceSample(cv::Mat &sample)
//{
//	assert(svm != NULL && svm->isTrained());
//	assert(!sample.empty());
//
//	cv::Mat result;
//	svm->predict(sample, result, cv::ml::StatModel::Flags::RAW_OUTPUT);
//	float dist = result.at<float>(0, 0);
//	return dist;
//}
//...
//
//float dist = distanceSample(yourSample);
//float confidence = (1.0 / (1.0 + exp(-dist)));

void nms(vector<Rect> &bboxes, vector<double> &confs, const double &conf_th, const double &nms_th)
{
	int SIZE = bboxes.size();
	if (SIZE <= 0)
		return;
	vector<bool> keep_idxs(SIZE, true);
	
	// Step1: 降序排序
	vector<int> idxs = argsort_d(confs);
	vector<double> confs_old = confs;
	vector<Rect> bboxes_old = bboxes;
	for (int i = 0; i < SIZE; i++)
	{
		confs[i] = confs_old[idxs[i]];
		bboxes[i] = bboxes_old[idxs[i]];
	}

	// Step2: 去掉置信度较低的
	for (int i = 0; i < SIZE; i++)
	{
		if (confs[i] < conf_th)
			keep_idxs[i] = false;
	}

	// Step3: nms
	for (int i = 0; i < SIZE-1; i++)
	{
		if (!keep_idxs[i])
			continue;
		for (int j = i + 1; j < SIZE; j++)
		{
			if (!keep_idxs[j])
				continue;
			double iou = bbox_iou(bboxes[i], bboxes[j]);
			if (iou > nms_th)
			{
				keep_idxs[j] = false;
			}
		}
			
	}

	// Step 4: 删除所有不满足条件的bbox
	vector<Rect> bboxes_new;
	vector<double> confs_new;
	for (int i = 0; i < SIZE; i++)
	{
		if (keep_idxs[i])
		{
			bboxes_new.emplace_back(bboxes[i]);
			confs_new.emplace_back(confs[i]);
		}
	}
	bboxes.assign(bboxes_new.begin(), bboxes_new.end());
	confs.assign(confs_new.begin(), confs_new.end());
	return;
}


