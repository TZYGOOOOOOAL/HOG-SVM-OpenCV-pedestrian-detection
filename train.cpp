#include "train.h"
#include "eval.h"
#include "test.h"
#include "Config.h"
#include "my_hog.h"
#include "utils.h"

extern Config config;

void train()
{
	Timer timer;

	/* Step 1: 构建特征矩阵 */
	Mat feature_mat, label_mat;

	// 已存在特征数据，直接加载
	if (is_file(config.TRAIN_FEATURE_PATH) && is_file(config.TRAIN_LABEL_PATH)) 
		//is_file(config.VALID_FEATURE_PATH) && is_file(config.VALID_LABEL_PATH))
	{
		cout << "Load Train DATA ... " << endl;
		feature_mat = load_mat_data(config.TRAIN_FEATURE_PATH);
		label_mat = load_mat_data(config.TRAIN_LABEL_PATH);
	}

	// 提取特征
	else
	{
		cout << "Start Extract Train Features ..." << endl;
		// 正样本
		Mat pos_feature_mat, pos_label_mat;
		vector<string> pos_img_paths = get_child_files(config.TRAIN_POS_IMG_DIR);
		create_feature_and_label_matrix(pos_img_paths, pos_feature_mat, pos_label_mat, 1);
		// 负样本
		Mat neg_feature_mat, neg_label_mat;
		vector<string> neg_img_paths = get_child_files(config.TRAIN_NEG_IMG_DIR);
		create_feature_and_label_matrix(neg_img_paths, neg_feature_mat, neg_label_mat, -1);
		// 整合
		cv::vconcat(pos_feature_mat, neg_feature_mat, feature_mat);
		cv::hconcat(pos_label_mat, neg_label_mat, label_mat);
		// 释放资源
		pos_feature_mat.release();
		neg_feature_mat.release();
		pos_label_mat.release();
		neg_label_mat.release();

		save_mat_data(feature_mat, config.TRAIN_FEATURE_PATH);
		save_mat_data(label_mat, config.TRAIN_LABEL_PATH);
		cout << "*** Save Feature Train Data ***" << endl;
	}
	
	timer.get_run_time("load data");

	/* Step 2: SVM 分类器配置 */
	cv::Ptr<ml::SVM> svm = ml::SVM::create();
	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(config.SVM_KERNEL_TYPE);
	svm->setC(1.0);
	svm->setClassWeights(config.CLASS_WEIGHT);

	svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER, config.SVM_TRAIN_ITER, FLT_EPSILON));

	/* Step 3：训练 SVM模型并保存 */
	cout << "Training SVM ..." << endl;
	// 准备数据
	Ptr<ml::TrainData> train_data = ml::TrainData::create(feature_mat, ml::ROW_SAMPLE, label_mat);

	timer.reset();

	// 开始训练
	if (config.TRAIN_AUTO)
	{
		ml::ParamGrid c_grid = ml::SVM::getDefaultGrid(ml::SVM::C);
		//ml::ParamGrid c_grid(0.5, 5.0, 1.2);
		svm->trainAuto(train_data, config.TRAIN_K_FLOD, c_grid);
	}
	else
		svm->train(train_data);
	cout << "Train END" << endl;

	timer.get_run_time("train once");
	
	// 保存结果
	string model_save_path = get_no_repeat_save_path(config.SVM_MODEL_PATH);
	svm->save(model_save_path);
	cout << "SVM model save path: " << model_save_path<< endl;

	/* Step 4 : 评价结果 */
	eval_valid(svm);
	return;
}

void train_hard()
{
	Timer timer;

	Mat feature_mat, label_mat;

	cv::Ptr<ml::SVM> svm = ml::SVM::create();
	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(config.SVM_KERNEL_TYPE);
	svm->setC(1.0);
	svm->setClassWeights(config.CLASS_WEIGHT);

	svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER, config.SVM_TRAIN_ITER, FLT_EPSILON));

	/* Step 5 ：难例查找并训练 */

	if (is_file(config.TRAIN_HARD_FEATURE_PATH) && is_file(config.TRAIN_HARD_LABEL_PATH))
	{
		cout << "Load Hard Features ..." << endl;
		feature_mat = load_mat_data(config.TRAIN_HARD_FEATURE_PATH);
		label_mat = load_mat_data(config.TRAIN_HARD_LABEL_PATH);
	}
	else
	{
		// 找难例
		cout << "Get Hard Examples..." << endl;
		Mat hard_feature_mat, hard_label_mat;
		get_hard_examples(svm, get_child_files(config.TRAIN_NEG_RAW_IMG_DIR), hard_feature_mat, hard_label_mat, false);

		// 保存特征
		cout << "Save Hard Examples..." << endl;
		save_mat_data(feature_mat, config.TRAIN_HARD_FEATURE_PATH);
		save_mat_data(label_mat, config.TRAIN_HARD_LABEL_PATH);

		// 构建数据
		cv::vconcat(feature_mat, hard_feature_mat, feature_mat);
		cv::hconcat(label_mat, hard_label_mat, label_mat);
	}

	timer.get_run_time("load hard data");

	Ptr<ml::TrainData> train_data = ml::TrainData::create(feature_mat, ml::ROW_SAMPLE, label_mat);

	timer.reset();

	// 再次训练
	cout << "Train Hard Examples..." << endl;
	if (config.TRAIN_AUTO)
	{
		//ml::ParamGrid c_grid(0.5, 5.0, 1.2);
		svm->trainAuto(train_data, config.TRAIN_K_FLOD);// , c_grid);
	}
	else
		svm->train(train_data);

	timer.get_run_time("train hard");

	// 保存模型
	string hard_example_model_save_path = get_no_repeat_save_path(config.SVM_HARD_MODEL_PATH);
	svm->save(hard_example_model_save_path);
	cout << "Hard examples SVM model save path: " << hard_example_model_save_path << endl;

	/* Step 6 : 再次评价结果 */
	eval_valid(svm);

	return;
}

void create_feature_and_label_matrix(vector<string> &img_paths, Mat &feature_mat, Mat &label_mat, 
	int label_val)
{
	unsigned int sample_num = img_paths.size();
	unsigned int feature_dim = 0;

	for (int i = 0; i < img_paths.size(); i++)
	{
		string img_path = img_paths[i];
		Mat img = imread(img_path);

		// 保证图像尺寸
		if (img.size() != config.PATCH_SIZE)
			resize(img, img, config.PATCH_SIZE);

		// HOG检测器
		HOGDescriptor hd = get_hog_desriptor(config.PATCH_SIZE, config.HOG_BLOCK_SIZE, config.HOG_BLOCK_STRIDE,
			config.HOG_CELL_SIZE, config.HOG_BINS);

		// 构建特征
		vector<float> descriptors = get_one_patch_hog_features(img, hd, config.WIN_STRIDE);

		if (i == 0)
		{
			feature_dim = descriptors.size();
			feature_mat = Mat::zeros(sample_num, feature_dim, CV_32FC1);
			label_mat = Mat(1, sample_num, CV_32SC1, Scalar::all(label_val));
		}

		for (int j = 0; j < feature_dim; j++)		// 复制->特征矩阵
			feature_mat.at<float>(i, j) = descriptors[j];

	}
}


// 训练难例
void get_hard_examples(Ptr<ml::SVM> &svm, vector<string> &neg_img_paths, 
	Mat &hard_feature_mat, Mat &hard_label_mat, bool save)
{
	HOGDescriptor hd = get_hog_desriptor(config.PATCH_SIZE, config.HOG_BLOCK_SIZE, config.HOG_BLOCK_STRIDE,
		config.HOG_CELL_SIZE, config.HOG_BINS);
	vector<float> hog_detector = convert_svm_detector(svm);
	hd.setSVMDetector(hog_detector);

	int feature_dim;
	hard_feature_mat.release();
	hard_label_mat.release();

	// 遍历每个负样本
	for (int img_idx = 0; img_idx < neg_img_paths.size(); img_idx++)
	{
		vector<Rect> bboxes;
		vector<double> confs;
		string img_path = neg_img_paths[img_idx];
		Mat img = imread(img_path);
		test_one_image_on_cv_method(img, hd, bboxes, confs, config.DETECT_STEP, config.DETECT_PAD, config.DETECT_SCALE,
			config.DETECT_CONF_TH, 0.8);
		
		// 找FP
		Mat one_hard_feature_mat, one_hard_label_mat;

		for (int bboxes_idx = 0; bboxes_idx < bboxes.size(); bboxes_idx++)
		{

			Rect FP_bbox = bboxes[bboxes_idx];
			Mat FP_ROI = img(FP_bbox).clone();
			resize(FP_ROI, FP_ROI, config.PATCH_SIZE);

			// 保存
			if (save)
			{
				string save_path = string("hard") + get_filename(img_path) + "_" + to_string(bboxes_idx) + get_ext(img_path);
				save_path = path_join(config.TRAIN_HARD_IMG_DIR, save_path);
				cout << "Hard example save path : " << save_path << endl;
				imwrite(save_path, FP_ROI);
			}

			// 提特征
			vector<float> descriptors = get_one_patch_hog_features(FP_ROI, hd, config.WIN_STRIDE);
			if (bboxes_idx == 0)
			{
				feature_dim = descriptors.size();
				one_hard_feature_mat = Mat::zeros(bboxes.size(), feature_dim, CV_32FC1);
				one_hard_label_mat = Mat(1, bboxes.size(), CV_32SC1, Scalar::all(-1));

				if (hard_feature_mat.empty() && hard_label_mat.empty())
				{
					hard_feature_mat = Mat(0, feature_dim, CV_32FC1);
					hard_label_mat = Mat(1, 0, CV_32SC1);
				}
			}
			for (int j = 0; j < feature_dim; j++)		// 复制->特征矩阵
				one_hard_feature_mat.at<float>(bboxes_idx, j) = descriptors[j];
			
			// 整合
			cv::vconcat(hard_feature_mat, one_hard_feature_mat, hard_feature_mat);
			cv::hconcat(hard_label_mat, one_hard_label_mat, hard_label_mat);
		}
		
	}

	cout << "\n Total Hard example NUM: " << hard_feature_mat.rows << endl;
	return;
}