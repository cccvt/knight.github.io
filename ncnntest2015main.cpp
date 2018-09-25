//./https://github.com/honghuCode/mobileFacenet-ncnn/tree/update-mobilefacenet-ncnn

#include<iostream>
#include<ncnn_mobileFace.h>
#include "net.h"
#include<fstream>
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include "mtcnn.h"
using namespace std;
#define _MSC_VER 1900
#define MAXFACEOPEN 1
std::vector<std::string> splitString_1(const std::string &str,
	const char delimiter) {
	std::vector<std::string> splited;
	std::string s(str);
	size_t pos;

	while ((pos = s.find(delimiter)) != std::string::npos) {
		std::string sec = s.substr(0, pos);

		if (!sec.empty()) {
			splited.push_back(s.substr(0, pos));
		}

		s = s.substr(pos + 1);
	}

	splited.push_back(s);

	return splited;
}



float simd_dot_1(const float* x, const float* y, const long& len) {
	float inner_prod = 0.0f;
	__m128 X, Y; // 128-bit values
	__m128 acc = _mm_setzero_ps(); // set to (0, 0, 0, 0)
	float temp[4];

	long i;
	for (i = 0; i + 4 < len; i += 4) {
		X = _mm_loadu_ps(x + i); // load chunk of 4 floats
		Y = _mm_loadu_ps(y + i);
		acc = _mm_add_ps(acc, _mm_mul_ps(X, Y));
	}
	_mm_storeu_ps(&temp[0], acc); // store acc into an array
	inner_prod = temp[0] + temp[1] + temp[2] + temp[3];

	// add the remaining values
	for (; i < len; ++i) {
		inner_prod += x[i] * y[i];
	}
	return inner_prod;
}
float CalcSimilarity_1(const float* fc1,
	const float* fc2,
	long dim) {

	return simd_dot_1(fc1, fc2, dim)
		/ (sqrt(simd_dot_1(fc1, fc1, dim))
			* sqrt(simd_dot_1(fc2, fc2, dim)));
}



int test_picture() {
	char *model_path = "./models";
	MTCNN mtcnn(model_path);

	clock_t start_time = clock();

	cv::Mat image;
	image = cv::imread("./sample.jpg");
	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
	std::vector<Bbox> finalBbox;

#if(MAXFACEOPEN==1)
	mtcnn.detectMaxFace(ncnn_img, finalBbox);
#else
	mtcnn.detect(ncnn_img, finalBbox);
#endif

	const int num_box = finalBbox.size();
	std::vector<cv::Rect> bbox;
	bbox.resize(num_box);
	for (int i = 0; i < num_box; i++) {
		bbox[i] = cv::Rect(finalBbox[i].x1, finalBbox[i].y1, finalBbox[i].x2 - finalBbox[i].x1 + 1, finalBbox[i].y2 - finalBbox[i].y1 + 1);

		for (int j = 0; j<5; j = j + 1)
		{
			cv::circle(image, cvPoint(finalBbox[i].ppoint[j], finalBbox[i].ppoint[j + 5]), 2, CV_RGB(0, 255, 0), CV_FILLED);
		}
	}
	for (vector<cv::Rect>::iterator it = bbox.begin(); it != bbox.end(); it++) {
		rectangle(image, (*it), cv::Scalar(0, 0, 255), 2, 8, 0);
	}

	imshow("face_detection", image);
	clock_t finish_time = clock();
	double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;
	std::cout << "time" << total_time * 1000 << "ms" << std::endl;

	cv::waitKey(0);
	return 1;
}


cv::Mat getsrc_roi(std::vector<cv::Point2f> x0, std::vector<cv::Point2f> dst)
{
	int size = dst.size();
	cv::Mat A = cv::Mat::zeros(size * 2, 4, CV_32FC1);
	cv::Mat B = cv::Mat::zeros(size * 2, 1, CV_32FC1);

	//[ x1 -y1 1 0] [a]       [x_1]
	//[ y1  x1 0 1] [b]   =   [y_1]
	//[ x2 -y2 1 0] [c]       [x_2]
	//[ y2  x2 0 1] [d]       [y_2]	

	for (int i = 0; i < size; i++)
	{
		A.at<float>(i << 1, 0) = x0[i].x;// roi_dst[i].x;
		A.at<float>(i << 1, 1) = -x0[i].y;
		A.at<float>(i << 1, 2) = 1;
		A.at<float>(i << 1, 3) = 0;
		A.at<float>(i << 1 | 1, 0) = x0[i].y;
		A.at<float>(i << 1 | 1, 1) = x0[i].x;
		A.at<float>(i << 1 | 1, 2) = 0;
		A.at<float>(i << 1 | 1, 3) = 1;

		B.at<float>(i << 1) = dst[i].x;
		B.at<float>(i << 1 | 1) = dst[i].y;
	}

	cv::Mat roi = cv::Mat::zeros(2, 3, A.type());
	cv::Mat AT = A.t();
	cv::Mat ATA = A.t() * A;
	cv::Mat R = ATA.inv() * AT * B;

	//roi = [a -b c;b a d ];

	roi.at<float>(0, 0) = R.at<float>(0, 0);
	roi.at<float>(0, 1) = -R.at<float>(1, 0);
	roi.at<float>(0, 2) = R.at<float>(2, 0);
	roi.at<float>(1, 0) = R.at<float>(1, 0);
	roi.at<float>(1, 1) = R.at<float>(0, 0);
	roi.at<float>(1, 2) = R.at<float>(3, 0);
	return roi;

}


cv::Mat faceAlign(cv::Mat image, MTCNN *mtcnn)
{
	double dst_landmark[10] = {
		38.2946, 73.5318, 55.0252, 41.5493, 70.7299,
		51.6963, 51.5014, 71.7366, 92.3655, 92.2041 };
	vector<cv::Point2f>coord5points;
	vector<cv::Point2f>facePointsByMtcnn;
	for (int i = 0; i < 5; i++) {
		coord5points.push_back(cv::Point2f(dst_landmark[i], dst_landmark[i + 5]));
	}
	char *model_path = "./models";
	(model_path);
	clock_t start_time = clock();

	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
	std::vector<Bbox> finalBbox;

#if(MAXFACEOPEN==1)
	mtcnn->detectMaxFace(ncnn_img, finalBbox);
#else
	mtcnn->detect(ncnn_img, finalBbox);
#endif

	const int num_box = finalBbox.size(); //人脸的数量（默认一张脸）
	std::vector<cv::Rect> bbox;
	bbox.resize(num_box);
	for (int i = 0; i < num_box; i++) {
		for (int j = 0; j<5; j = j + 1)
		{
			//cv::rectangle(image, cv::Point(finalBbox[i].x1, finalBbox[i].y1), cv::Point(finalBbox[i].x2, finalBbox[i].y2), CV_RGB(0, 0, 255));
			//cv::circle(image, cvPoint(finalBbox[i].ppoint[j], finalBbox[i].ppoint[j + 5]), 2, CV_RGB(0, 255, 0), CV_FILLED);
			facePointsByMtcnn.push_back(cvPoint(finalBbox[i].ppoint[j], finalBbox[i].ppoint[j + 5]));
		}
	}

	cv::Mat warp_mat = cv::estimateRigidTransform(facePointsByMtcnn, coord5points, false);
	printf("rio,before\n");
	if (warp_mat.empty()) {
		warp_mat = getsrc_roi(facePointsByMtcnn, coord5points);
		printf("rio,ing\n");
	}
	warp_mat.convertTo(warp_mat, CV_32FC1);
	cv::Mat alignFace = cv::Mat::zeros(112, 112, image.type());

	//cv::imshow("image", image);
	//cv::waitKey(0);
	warpAffine(image, alignFace, warp_mat, alignFace.size());
	return alignFace;
}

void mainold()
{
	cv::Mat image = cv::imread("./xuzheng.jpg");
	MTCNN *mtcnn = new MTCNN("./models");
	cv::Mat alignedFace1 = faceAlign(image, mtcnn);

	image = cv::imread("./xuzheng_2.jpg");

	cv::Mat alignedFace2 = faceAlign(image, mtcnn);

	cv::imshow("alignedFace1", alignedFace1);
	cv::waitKey(0);

	cv::imshow("alignedFace2", alignedFace2);
	cv::waitKey(0);

	ncnn::Net squeezenet;
	//98.83
	/*squeezenet.load_param("mobilenet_ncnn.param");
	squeezenet.load_model("mobilenet_ncnn.bin");*/
	//99.4
	squeezenet.load_param("mobilefacenet.param");
	squeezenet.load_model("mobilefacenet.bin");
	ncnn::Extractor ex = squeezenet.create_extractor();
	ex.set_light_mode(true);


	//cout << "lfw-112X112/" + img_L << endl;
	long t1 = clock();
	float* feat1 = getFeatByMobileFaceNetNCNN(ex, alignedFace1);
	float *feat2 = getFeatByMobileFaceNetNCNN(ex, alignedFace2);
	long t2 = clock();


	float sim = CalcSimilarity_1(feat1, feat2, 128);
	fprintf(stderr, "time:%f,sim:%f\n", (t2 - t1) / 2.0, sim);



	//cv::imshow("alignedFace", alignedFace);
	//cv::waitKey(0);
	////人脸对齐
	//
	//ncnn::Net squeezenet;
	////98.83
	///*squeezenet.load_param("mobilenet_ncnn.param");
	//squeezenet.load_model("mobilenet_ncnn.bin");*/
	////99.4
	//squeezenet.load_param("mobilefacenet.param");
	//squeezenet.load_model("mobilefacenet.bin"); 
	//ncnn::Extractor ex = squeezenet.create_extractor();
	//ex.set_light_mode(true);

	//cv::Mat m1 = cv::imread("2_1.jpg", CV_LOAD_IMAGE_COLOR);
	//cv::Mat m2 = cv::imread("2.jpg", CV_LOAD_IMAGE_COLOR);
	//	
	////cout << "lfw-112X112/" + img_L << endl;

	//float* feat1 = getFeatByMobileFaceNetNCNN(ex, m1);
	//float *feat2 = getFeatByMobileFaceNetNCNN(ex, m2);



	//float sim = CalcSimilarity_1(feat1, feat2, 128);
	//fprintf(stderr, "%f\n", sim);
	////LFW: 99.50, CFP_FP: 88.94, AgeDB30: 95.91
	//fstream in("pairs_1.txt");
	//fstream out("rs_lfw_99.50.txt",ios::out);
	//string line;
	//long t1 = clock();
	//int count = 0;
	//while (in >> line)
	//{
	//	//cout <<line<<endl;
	//	std::vector<std::string>  rs = splitString_1(line, ',');

	//	string img_L = rs[0];
	//	string img_R = rs[1];
	//	string flag = rs[2];
	//	//cout <<img_L<<endl;
	//	std::vector<float> cls_scores;
	//	cv::Mat m1 = cv::imread("lfw-112X112/" + img_L, CV_LOAD_IMAGE_COLOR);
	//	cv::Mat m2 = cv::imread("lfw-112X112/" + img_R, CV_LOAD_IMAGE_COLOR);
	//	
	//	//cout << "lfw-112X112/" + img_L << endl;

	//	float* feat1 = getFeatByMobileFaceNetNCNN(ex, m1);
	//	float *feat2 = getFeatByMobileFaceNetNCNN(ex, m2);
	//	float sim = CalcSimilarity_1(feat1, feat2, 128);
	//	fprintf(stderr, "%s,%f\n", flag.c_str(), sim);
	//	out << flag.c_str() << "\t"<<sim << endl;
	//	long t2 = clock();
	//	if (count++ % 10 == 0)
	//	{
	//	
	//		cout << t2 - t1 << "s"<<endl;
	//		t1 = t2;
	//	}
	//}


	//float* getFeatByMobileFaceNetNCNN(ncnn::Extractor ex, cv::Mat img);
	//cout << "ssssss" << endl;
}





cv::Vec3d getTransformationParameters(
	const std::vector<cv::Point2f>& srcImagePoints,
	const std::vector<cv::Point3f>& modelPoints,
	const cv::Mat& srcImage//,
	////glm::mat3& rotationMatrix,
	////glm::vec3& translationVector
)
{
	std::vector<double> rv(3), tv(3);
	cv::Mat rvec(rv), tvec(tv);

	cv::Mat ip(srcImagePoints);
	cv::Mat op = cv::Mat(modelPoints);
	cv::Scalar m = mean(cv::Mat(modelPoints));

	rvec = cv::Mat(rv);
	double _d[9] =
	{
		1, 0, 0,
		0, -1, 0,
		0, 0, -1
	};
	Rodrigues(cv::Mat(3, 3, CV_64FC1, _d), rvec);
	tv[0] = 0; tv[1] = 0; tv[2] = 1;
	tvec = cv::Mat(tv);

	double max_d = MAX(srcImage.rows, srcImage.cols);
	double _cm[9] =
	{
		max_d, 0, (double)srcImage.cols / 2.0,
		0, max_d, (double)srcImage.rows / 2.0,
		0, 0, 1.0
	};
	cv::Mat camMatrix = cv::Mat(3, 3, CV_64FC1, _cm);

	double _dc[] = { 0, 0, 0, 0 };
	solvePnP(op, ip, camMatrix, cv::Mat(1, 4, CV_64FC1, _dc), rvec, tvec, false, CV_EPNP);

	double rot[9] = { 0 };
	cv::Mat rotM(3, 3, CV_64FC1, rot);
	Rodrigues(rvec, rotM);
	double* _r = rotM.ptr<double>();

	// printf("rotation mat: \n %.3f %.3f %.3f\n%.3f %.3f %.3f\n%.3f %.3f %.3f\n", _r[0], _r[1], _r[2], _r[3], _r[4], _r[5], _r[6], _r[7], _r[8]);

	rotM = rotM.t();

	//rotationMatrix = glm::mat3(
	//	(float)rot[0], (float)rot[1], (float)rot[2],
	//	(float)rot[3], (float)rot[4], (float)rot[5],
	//	(float)rot[6], (float)rot[7], (float)rot[8]);

	//// printf("trans vec: \n %.3f %.3f %.3f\n", tv[0], tv[1], tv[2]);	

	//translationVector = glm::vec3(tv[0], tv[1], tv[2]);

	double _pm[12] =
	{
		_r[0], _r[1], _r[2], tv[0],
		_r[3], _r[4], _r[5], tv[1],
		_r[6], _r[7], _r[8], tv[2]
	};

	cv::Mat tmp, tmp1, tmp2, tmp3, tmp4, tmp5;
	/*
	yaw   y
	pitch x
	roll  z
	*/
	cv::Vec3d eav;
	cv::decomposeProjectionMatrix(cv::Mat(3, 4, CV_64FC1, _pm), tmp, tmp1, tmp2, tmp3, tmp4, tmp5, eav);

	printf("Face Rotation Angle:  %.5f %.5f %.5f\n", eav[0], eav[1], eav[2]);
	return eav;
}

std::vector<Bbox> ywlfaceAlign(cv::Mat image, MTCNN *mtcnn)
{
	vector<cv::Point2f>facePointsByMtcnn;

	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
	std::vector<Bbox> finalBbox;

#if(MAXFACEOPEN==1)
	mtcnn->detectMaxFace(ncnn_img, finalBbox);
#else
	mtcnn->detect(ncnn_img, finalBbox);
#endif

	return finalBbox;
	//const int num_box = finalBbox.size(); //人脸的数量（默认一张脸）

	//cv::Mat tmp;
	//cv::Mat alignFace = cv::Mat::zeros(224, 224, image.type());
	//for (int i = 0; i < num_box; i++) 
	//{
	//	image(cv::Rect(cv::Point(finalBbox[i].x1, finalBbox[i].y1), cv::Point(finalBbox[i].x2, finalBbox[i].y2))).copyTo(tmp);

	//	cv::resize(tmp,alignFace, alignFace.size());
	//	//for (int j = 0; j<5; j = j + 1)
	//	//{
	//	//	//cv::rectangle(image, cv::Point(finalBbox[i].x1, finalBbox[i].y1), cv::Point(finalBbox[i].x2, finalBbox[i].y2), CV_RGB(0, 0, 255));
	//	//	//cv::circle(image, cvPoint(finalBbox[i].ppoint[j], finalBbox[i].ppoint[j + 5]), 2, CV_RGB(0, 255, 0), CV_FILLED);
	//	//	facePointsByMtcnn.push_back(cvPoint(finalBbox[i].ppoint[j], finalBbox[i].ppoint[j + 5]));
	//	//}
	//}
	//return alignFace;
}
#include <windows.h>
void main()
{
	ncnn::Net mobilenet;
	//98.83
	/*squeezenet.load_param("mobilenet_ncnn.param");
	squeezenet.load_model("mobilenet_ncnn.bin");*/
	//99.4
	mobilenet.load_param("mobilenet302.param");
	mobilenet.load_model("mobilenet302.bin");
	ncnn::Extractor ex = mobilenet.create_extractor();
	ex.set_light_mode(true);

	cv::Mat image = cv::imread("./rbgaMat/1706.jpg");//subimage/5954.jpg//xuzheng_2.jpg./56_Voter_peoplevoting_56_192.jpg
	cv::imshow("begin", image);
	cv::waitKey(0);
	
	MTCNN *mtcnn = new MTCNN("./models");

	cv::Mat face224=cv::Mat::zeros(224, 224, image.type());
	std::vector<Bbox> finalBbox= ywlfaceAlign(image, mtcnn);

	if(1)
	{
		const int num_box = finalBbox.size(); //人脸的数量（默认一张脸）

		cv::Mat tmp;
		cv::Mat alignFace = cv::Mat::zeros(224, 224, image.type());
		for (int i = 0; i < num_box; i++)
		{
			image(cv::Rect(cv::Point(finalBbox[i].x1, finalBbox[i].y1), cv::Point(finalBbox[i].x2, finalBbox[i].y2))).copyTo(tmp);
			cv::resize(tmp, face224, face224.size());
		}	
	}

	cv::imshow("before", face224);
	cv::waitKey(0);

	ncnn::Mat in = ncnn::Mat::from_pixels_resize(face224.data, ncnn::Mat::PIXEL_RGB2BGR, face224.cols, face224.rows, 224, 224);
	float norm_vals[3] = { 1 / 255.0f, 1 / 255.0f, 1 / 255.0f };
	float zero_vals[3] = { 0.f };
	in.substract_mean_normalize(zero_vals, norm_vals);

	LARGE_INTEGER nFreq, nBeginTime, nEndTime;  // 头文件为#include <windows.h>
	QueryPerformanceFrequency(&nFreq);
	QueryPerformanceCounter(&nBeginTime);
	//long t1 = clock();

	ex.input("data", in);
	//ncnn::Mat out2;
	//ex.extract("dropout0", out2);
	ncnn::Mat out;
	ex.extract("Addmm_1", out);

	//long t2 = clock();
	QueryPerformanceCounter(&nEndTime);
	double time = (double)(nEndTime.QuadPart - nBeginTime.QuadPart) * 1000 / (double)nFreq.QuadPart;

	fprintf(stderr, "time:clock-%f,		query-%f\n", 1.0/*(t2 - t1)*/ ,time);
		


	ncnn::Mat out_flatterned = out.reshape(out.w * out.h * out.c);
	std::vector<cv::Point2f> keypoints;
	for (int j = 0; j<out_flatterned.w; j+=2)
	{
		keypoints.push_back(
			cvPoint(out_flatterned[j] * (finalBbox[0].x2- finalBbox[0].x1)+ finalBbox[0].x1,
			out_flatterned[j+1] * (finalBbox[0].y2 - finalBbox[0].y1) + finalBbox[0].y1)
		);

		cv::circle(face224, cvPoint(out_flatterned[j] * 224, out_flatterned[j+1] * 224), 2, CV_RGB(0, 255, 0), CV_FILLED);
		cv::circle(image, cvPoint(out_flatterned[j] * (finalBbox[0].x2 - finalBbox[0].x1) + finalBbox[0].x1,
			out_flatterned[j + 1] * (finalBbox[0].y2 - finalBbox[0].y1) + finalBbox[0].y1), 2, CV_RGB(0, 255, 0), CV_FILLED);

	}

	cv::imshow("after", face224);
	cv::waitKey(0);

	cv::imshow("after1", image);
	cv::waitKey(0);


	//-----------------------------------------------------------------------------

	std::vector<cv::Point2f> srcImagePoints;
	int i = 0;
	int kpt_old[] = {
		36,39,45,42,	31,35,	48,54,	8,
		37,38, 40,41, 43,44, 46,47,
		0,16,
		5,11,
		4,12,
		27,28,29,
		30,
		//19,24
	};
	int kpt[] = {
		60,64,72,68,	55,59,	76,82,	16,

		61,63, 65,57, 69,71,73,75,
		0,32,
		9,23,
		7,25,
		51,52,53,54,
		//19,24
	};
	for (; i<sizeof(kpt) / sizeof(kpt[0]); i++)
	{
		srcImagePoints.push_back(keypoints[kpt[i]]);
	}


	const cv::Point3f modelPointsArr[] =
	{
		cv::Point3f(-1.030300, -0.41930, -0.38129),//36
		cv::Point3f(-0.493680, -0.38700, -0.55059),//39
		cv::Point3f(+1.030300, -0.41930, -0.38129),//45
		cv::Point3f(+0.493680, -0.38700, -0.55059),//42
		cv::Point3f(-0.363830, +0.52565, -0.79787),//31
		cv::Point3f(+0.363830, +0.52565, -0.79787),//35
		cv::Point3f(-0.599530, +1.10768, -0.71667),//48
		cv::Point3f(+0.599530, +1.10768, -0.71667),//54
		cv::Point3f(-0.000002, +1.99444, -0.94946),//8

		//./ywl add
		cv::Point3f(-0.894518,-0.551335,-0.564469),//37
		cv::Point3f(-0.659822,-0.518812,-0.598789),//38
		cv::Point3f(-0.637257,-0.310396,-0.610667),//40
		cv::Point3f(-0.864146,-0.294665,-0.539839),//41

		cv::Point3f(0.671384,-0.54734,-0.625397),//43
		cv::Point3f(0.91115,-0.551335,-0.54646),//44

		cv::Point3f(0.880778,-0.294665,-0.557848),//46
		cv::Point3f(0.653891,-0.310396,-0.62867),//47

		cv::Point3f(-1.4796,-0.05929,1.17575),//0
		cv::Point3f(1.47959,-0.05929,1.17575),//16
		cv::Point3f(-0.996478,1.63978,-0.012313),//5
		cv::Point3f(0.996478,1.63978,-0.012313),//11

		cv::Point3f(-1.19827,1.47996,0.435455),//4
		cv::Point3f(1.19826,1.47996,0.435455),//12

		cv::Point3f(-0.000002,-0.555344,-0.774465),//27
		cv::Point3f(-0.000002,-0.165599,-0.959329),//28
		cv::Point3f(-0.000002,0.082843,-1.17602),//29
		cv::Point3f(-0.000002,0.340833,-1.38839),//30

												//cv::Point3f(-0.94543,-0.748272,-0.565522),//19
												//cv::Point3f(0.94543,-0.748272,-0.565522)//24

	};

	std::vector<cv::Point3f> modelPoints(modelPointsArr, modelPointsArr + sizeof(modelPointsArr) / sizeof(modelPointsArr[0]));

	cv::Vec3d euler_angle = getTransformationParameters(srcImagePoints, modelPoints, image);

	//glm::mat3 rotationMatrix;
	//glm::vec3 translationVector;
	//getTransformationParameters(srcImagePoints, modelPoints, srcImageCV, rotationMatrix, translationVector);
	//glm::mat4 translationMatrix = glm::translate(translationVector);


	//glm::mat4 rotationMat4 = glm::mat4(
	//	rotationMatrix[0][0], rotationMatrix[0][1], rotationMatrix[0][2], 0,
	//	rotationMatrix[1][0], rotationMatrix[1][1], rotationMatrix[1][2], 0,
	//	rotationMatrix[2][0], rotationMatrix[2][1], rotationMatrix[2][2], 0,
	//	0, 0, 0, 1
	//);



	//-------------------------------------------------------------------------------------



	//cv::imshow("alignedFace", alignedFace);
	//cv::waitKey(0);
	////人脸对齐
	//
	//ncnn::Net squeezenet;
	////98.83
	///*squeezenet.load_param("mobilenet_ncnn.param");
	//squeezenet.load_model("mobilenet_ncnn.bin");*/
	////99.4
	//squeezenet.load_param("mobilefacenet.param");
	//squeezenet.load_model("mobilefacenet.bin"); 
	//ncnn::Extractor ex = squeezenet.create_extractor();
	//ex.set_light_mode(true);

	//cv::Mat m1 = cv::imread("2_1.jpg", CV_LOAD_IMAGE_COLOR);
	//cv::Mat m2 = cv::imread("2.jpg", CV_LOAD_IMAGE_COLOR);
	//	
	////cout << "lfw-112X112/" + img_L << endl;

	//float* feat1 = getFeatByMobileFaceNetNCNN(ex, m1);
	//float *feat2 = getFeatByMobileFaceNetNCNN(ex, m2);



	//float sim = CalcSimilarity_1(feat1, feat2, 128);
	//fprintf(stderr, "%f\n", sim);
	////LFW: 99.50, CFP_FP: 88.94, AgeDB30: 95.91
	//fstream in("pairs_1.txt");
	//fstream out("rs_lfw_99.50.txt",ios::out);
	//string line;
	//long t1 = clock();
	//int count = 0;
	//while (in >> line)
	//{
	//	//cout <<line<<endl;
	//	std::vector<std::string>  rs = splitString_1(line, ',');

	//	string img_L = rs[0];
	//	string img_R = rs[1];
	//	string flag = rs[2];
	//	//cout <<img_L<<endl;
	//	std::vector<float> cls_scores;
	//	cv::Mat m1 = cv::imread("lfw-112X112/" + img_L, CV_LOAD_IMAGE_COLOR);
	//	cv::Mat m2 = cv::imread("lfw-112X112/" + img_R, CV_LOAD_IMAGE_COLOR);
	//	
	//	//cout << "lfw-112X112/" + img_L << endl;

	//	float* feat1 = getFeatByMobileFaceNetNCNN(ex, m1);
	//	float *feat2 = getFeatByMobileFaceNetNCNN(ex, m2);
	//	float sim = CalcSimilarity_1(feat1, feat2, 128);
	//	fprintf(stderr, "%s,%f\n", flag.c_str(), sim);
	//	out << flag.c_str() << "\t"<<sim << endl;
	//	long t2 = clock();
	//	if (count++ % 10 == 0)
	//	{
	//	
	//		cout << t2 - t1 << "s"<<endl;
	//		t1 = t2;
	//	}
	//}


	//float* getFeatByMobileFaceNetNCNN(ncnn::Extractor ex, cv::Mat img);
	//cout << "ssssss" << endl;
}
