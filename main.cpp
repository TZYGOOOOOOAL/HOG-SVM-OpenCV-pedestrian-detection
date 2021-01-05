#define _CRT_SECURE_NO_WARNINGS

#include "utils.h"
#include "make_data.h"
#include "train.h"
#include "eval.h"
#include "Config.h"
#include "test.h"

Config config;

void main(void)
{
	//make_data();
	//train();
	//train_hard();
	//eval();
	test();

	system("pause");
}