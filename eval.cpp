#include "eval.h"
#include "train.h"
#include "Config.h"
#include "my_hog.h"
#include "utils.h"
extern Config config;

void eval_valid(Ptr<ml::SVM> &svm)
{
	/* Step 1: 构建特征矩阵 */

	Mat feature_mat, label_mat;

	// 已存在特征数据，直接加载
	if (is_file(config.VALID_FEATURE_PATH) && is_file(config.VALID_LABEL_PATH))
	{
		cout << "\nStart Load Valid DATA ... " << endl;
		feature_mat = load_mat_data(config.VALID_FEATURE_PATH);
		label_mat = load_mat_data(config.VALID_LABEL_PATH);
	}

	// 提取特征
	else
	{
		cout << "Start Extract Valid Features ..." << endl;
		// 正样本
		Mat pos_feature_mat, pos_label_mat;
		vector<string> pos_img_paths = get_child_files(config.VALID_POS_IMG_DIR);
		create_feature_and_label_matrix(pos_img_paths, pos_feature_mat, pos_label_mat, 1);
		// 负样本
		Mat neg_feature_mat, neg_label_mat;
		vector<string> neg_img_paths = get_child_files(config.VALID_NEG_IMG_DIR);
		create_feature_and_label_matrix(neg_img_paths, neg_feature_mat, neg_label_mat, -1);
		// 整合
		cv::vconcat(pos_feature_mat, neg_feature_mat, feature_mat);
		cv::hconcat(pos_label_mat, neg_label_mat, label_mat);
		// 释放资源
		pos_feature_mat.release();
		neg_feature_mat.release();
		pos_label_mat.release();
		neg_label_mat.release();

		cout << "Save Valid Feature Data ..." << endl;
		save_mat_data(feature_mat, config.VALID_FEATURE_PATH);
		save_mat_data(label_mat, config.VALID_LABEL_PATH);
	}

	/* Step 2： 加载 SVM 分类器模型 */
	if (svm->empty())
	{
		if (!is_file(config.SVM_MODEL_PATH))
		{
			cerr << "MODEL PATH is NOT EXIST !!!";
			return;
		}
		svm = load_svm(config.SVM_MODEL_PATH);
	}
	
	//auto x = convert_svm_detector(svm);
	
	/* Step 3：预测 */
	cout << "Evaling ..." << endl;
	Mat predict_mat;
	svm->predict(feature_mat, predict_mat);
	predict_mat = predict_mat.t();
	eval_classify(predict_mat, label_mat);

	return;
}

void eval_classify(Mat &predicts, Mat &labels)
{
	predicts.convertTo(predicts, CV_32SC1);
	unsigned int TP = 0, TN = 0, FP = 0, FN = 0;
	unsigned int total = predicts.cols;
	float eps = FLT_EPSILON;

	int pred, label;
	for (int i = 0; i < total; i++)
	{
		pred = predicts.at<int>(0, i);
		label = labels.at<int>(0, i);

		if (pred == label)
		{
			if (pred == 1)
				TP++;
			else
				TN++;
		}
		else
		{
			if (pred == 1)
				FP++;
			else
				FN++;
		}
	}
	float acc = 1.0f * (TP + TN) / total;
	float precision = 1.0f * TP / (TP + FP + eps);
	float recall = 1.0f * TP / (TP + FN + eps);
	float f1 = 2 * recall * precision / (recall + precision + eps);
	float FA = 1 - precision;
	float MA = 1 - recall;
	
	cout << "****** EVAL RESULT ******" << endl;
	cout << "Total    : " << total << endl;
	cout << "TP    : " << TP << endl;
	cout << "TN    : " << TN << endl;
	cout << "FP    : " << FP << endl;
	cout << "FN    : " << FN << '\n' << endl;
	cout << "acc      : " << acc       << endl;
	cout << "precision: " << precision << endl;
	cout << "recall   : " << recall    << endl;
	cout << "f1       : " << f1 << endl;
	cout << "FA       : " << FA << endl;
	cout << "MA       : " << MA << endl;
	return;
}


// 检测评价
void eval_detect(const vector<vector<Rect>> &predicts, const vector<vector<Rect>> &labels, const double &iou_th)
{
	assert(predicts.size() == labels.size());

	cout << "Detect Testing ..." << endl;

	int TP = 0;
	int TP_FP = 0;
	int TP_FN = 0;
	double AP = 0.0;
	double eps = 1e-12;

	// 评价每张图像
	for (int i = 0; i < predicts.size(); i++)
	{
		int tp=0;
		double ap=0;
		eval_detect_one_img(predicts[i], labels[i], iou_th, tp, ap);
		TP += tp;
		AP += ap;
		TP_FP += predicts[i].size();
		TP_FN += labels[i].size();
	}

	AP /= predicts.size();

	double precision = 1.0 * TP / (TP_FP + eps);
	double recall = 1.0 * TP / (TP_FN + eps);
	double f1 = 2 * recall * precision / (recall + precision + eps);
	double FA = 1 - precision;
	double MA = 1 - recall;

	cout << "****** DETECT EVAL RESULT ******" << endl;
	cout << "Predict Num  : " << TP_FP << endl;
	cout << "Label Num    : " << TP_FN << endl;
	cout << "TP    : " << TP  << endl;
	cout << "AP    : " << AP << endl;
	cout << "precision: " << precision << endl;
	cout << "recall   : " << recall << endl;
	cout << "f1       : " << f1 << endl;
	cout << "FA       : " << FA << endl;
	cout << "MA       : " << MA << endl;
	return;

}

// 
void eval_detect_one_img(const vector<Rect> &predict_bboxes, const vector<Rect> &label_bboxes, 
	const double &iou_th, int &TP, double &p)
{
	// 记录被访问过的预测bbox
	vector<bool> visited(predict_bboxes.size(), false);
	
	// 遍历label 取Iou最大且大于阈值
	for (int label_idx = 0 ; label_idx < label_bboxes.size(); label_idx++)
	{
		double iou_max = 0;
		int iou_max_idx = -1;
		for (int pred_idx = 0; pred_idx < predict_bboxes.size(); pred_idx++)
		{
			if (visited[pred_idx])
				continue;
			double iou = bbox_iou(label_bboxes[label_idx], predict_bboxes[pred_idx]);
			if (iou > iou_max && iou >= iou_th)
			{
				iou_max = iou;
				iou_max_idx = pred_idx;
			}
		}

		// 找到TP
		if (iou_max_idx >= 0)
		{
			visited[iou_max_idx] = true;
			TP++;
		}
	}

	p = 1.0 * TP / (predict_bboxes.size() + 1e-12);
}