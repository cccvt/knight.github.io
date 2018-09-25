// ==================================================
// C\C++ INCLUDES
// ==================================================
#include <iostream>
#include <stdio.h>
#include <math.h>

// ==================================================
// OPENCV INCLUDES
// ==================================================
#include <opencv2\opencv.hpp>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

// ==================================================
// DLIB INCLUDES
// ==================================================
#include <dlib\opencv\cv_image.h>
#include <dlib\image_processing\frontal_face_detector.h>
#include <dlib\image_processing\render_face_detections.h>
#include <dlib\image_processing.h>
#include <dlib\gui_widgets.h>
#include <dlib\image_io.h>

// ==================================================
// GL\SDL\GLM INCLUDES
// ==================================================
#include <GL\glew.h>
#include <SDL2\SDL.h>
#include <glm\glm.hpp>
#include <glm\gtx\transform.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>
#include <glm\gtx\euler_angles.hpp>
#include <glm\gtx\string_cast.hpp>

#include "OBJLoader.h"

#include "shader_m.h"
#include <fstream> 
#include <direct.h>
#include <ctime>
using std::endl;
using std::cout;


const char* TITLE = "3D Face Reconstruction";
 int	WIDTH =  600;//967;// 400;//
 int	HEIGHT = 600;//967;//400;//

const GLchar* VERTEX_SHADER_SOURCE[] = {
	"#version 330 core																				\n"
	"																								\n"
	"layout(location = 0) in vec3 position;															\n"
	"layout(location = 1) in vec3 color;															\n"
	"																								\n"
	"uniform bool calcColor;																		\n"
	"uniform vec2 v2;																				\n"

	"uniform mat4 M;																				\n"
	"uniform mat4 VP;																				\n"
	"																								\n"	
	"out vec2 f_texCoords;																			\n"
	"																								\n"
	"float map(float val, float A, float B, float a, float b)										\n"
	"{																								\n"
	"	return (val - A) * (b - a) / (B - A) + a;													\n"
	"}																								\n"
	"																								\n"
	"void main()																					\n"
	"{																								\n"
	"	vec4 ndc = VP * M * vec4(position, 1);														\n"
	"	gl_Position = ndc;																			\n"
	"	float u = map(ndc.x / ndc.w, -1, 1, 0, 1);													\n"
	"	float v = map(ndc.y / ndc.w, -1, 1, 0, 1);													\n"
	"	f_texCoords = vec2(u+v2.x/10, v-v2.y/20);																	\n"
	"}																								\n"
};
const GLchar* FRAGMENT_SHADER_SOURCE[] = {
	"#version 330 core																				\n"
	"																								\n"
	"in vec2 f_texCoords;																			\n"
	//"uniform vec3 ywlcolor;																			\n"
	"uniform sampler2D sampler;																		\n"
	"																								\n"
	"out vec4 fragColor;																			\n"
	"																								\n"
	"void main()																					\n"
	"{																								\n"
	"	vec4 sampledColor = texture(sampler, f_texCoords);											\n"
	"	if(sampledColor.r == 0)																		\n"
	"		fragColor = vec4(0.0, 0.0, 0.0, 1.0);													\n"//fragColor = vec4(0.0, 0.8, 0.8, 1.0);
	//"		fragColor = vec4(0.0, 0.8, 0., 1.0);													\n"//fragColor = vec4(0.0, 0.8, 0.8, 1.0);
	"	else																						\n"
	"		fragColor = sampledColor;																\n"
	"}																								\n"
};

//feature			index
//Лево око надвор	36
//Лево око внатре	39
//Десно око надвор	45
//Десно око внатре	42
//Нос лево			31
//Нос десно			35
//Уста лево			48
//Уста десно		54
//Брада				8

// MALE
//const cv::Point3f modelPointsArr[] =
//{
//	cv::Point3f(-1.13642, -0.90731, +0.00391),
//	cv::Point3f(-0.48241, -0.83527, -0.05782),
//	cv::Point3f(+1.13642, -0.90731, +0.00391),
//	cv::Point3f(+0.48241, -0.83527, -0.05782),
//	cv::Point3f(-0.44994, +0.07104, -0.46759),
//	cv::Point3f(+0.44994, +0.07104, -0.46759),
//	cv::Point3f(-0.63738, +0.81876, -0.25444),
//	cv::Point3f(+0.63738, +0.81876, -0.25444),
//	cv::Point3f(-0.00005, +1.85261, -0.27220)
//};

// FEMALE
//const cv::Point3f modelPointsArr[] =
//{
//cv::Point3f(-1.030300, -0.41930, -0.38129),//36
//cv::Point3f(-0.493680, -0.38700, -0.55059),//39
//cv::Point3f(+1.030300, -0.41930, -0.38129),//45
//cv::Point3f(+0.493680, -0.38700, -0.55059),//42
//cv::Point3f(-0.363830, +0.52565, -0.79787),//31
//cv::Point3f(+0.363830, +0.52565, -0.79787),//35
//cv::Point3f(-0.599530, +1.10768, -0.71667),//48
//cv::Point3f(+0.599530, +1.10768, -0.71667),//54
//cv::Point3f(-0.000002, +1.99444, -0.94946),//8
//	//./ywl add
//	cv::Point3f(-0.894518,-0.551335,-0.564469),//37
//	cv::Point3f(-0.659822,-0.518812,-0.598789),//38
//	cv::Point3f(0.880778,-0.294665,-0.557848),//46
//	cv::Point3f(0.653891,-0.310396,-0.62867),//47
//
//	cv::Point3f(-1.4796,-0.05929,1.17575),//0
//	cv::Point3f(1.47959,-0.05929,1.17575),//16
//
//	cv::Point3f(-0.000002,0.340833,-1.38839),//30
//	cv::Point3f(-0.94543,-0.748272,-0.565522),//19
//	cv::Point3f(0.94543,-0.748272,-0.565522)//24
//
//
//};

// FEMALE-mini-1
//const cv::Point3f modelPointsArr[] =
//{
//	cv::Point3f(-1.030300, -0.41930, -0.38129),//36
//	cv::Point3f(-0.493680, -0.38700, -0.55059),//39
//	cv::Point3f(+1.030300, -0.41930, -0.38129),//45
//	cv::Point3f(+0.493680, -0.38700, -0.55059),//42
//	 //./ywl add
//	cv::Point3f(-1.4796,-0.05929,1.17575),//0
//	cv::Point3f(1.47959,-0.05929,1.17575),//16
//};

// FEMALE-其他部位的点效果不是很明显
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


// http://www.morethantechnical.com/2012/10/17/head-pose-estimation-with-opencv-opengl-revisited-w-code/
void getTransformationParameters(
	const std::vector<cv::Point2f>& srcImagePoints,
	const std::vector<cv::Point3f>& modelPoints,
	const cv::Mat& srcImage,
	glm::mat3& rotationMatrix,
	glm::vec3& translationVector
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
	
	rotationMatrix = glm::mat3(
		(float)rot[0], (float)rot[1], (float)rot[2],
		(float)rot[3], (float)rot[4], (float)rot[5],
		(float)rot[6], (float)rot[7], (float)rot[8]);
	
	// printf("trans vec: \n %.3f %.3f %.3f\n", tv[0], tv[1], tv[2]);	

	translationVector = glm::vec3(tv[0], tv[1], tv[2]);

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
	
	// printf("Face Rotation Angle:  %.5f %.5f %.5f\n", eav[0], eav[1], eav[2]);	
}

//./ywl已验证，不仅可以保存为bmp格式，也可以保存为jpg格式
void ScreenShot(std::string picpath)
{
	GLint pView[4];
	glGetIntegerv(GL_VIEWPORT, pView);//得到视图矩阵,pView[2]为宽即width,pView[3]为高即height

	GLsizei numComponet = 3;
	GLsizei bufferSize = pView[2] * pView[3] * sizeof(GLfloat)*numComponet;

	GLfloat* _data = new GLfloat[bufferSize];
	unsigned char*  data = new unsigned char[bufferSize];
	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);//设置4字节对齐
	glReadBuffer(GL_FRONT);
	glReadPixels(pView[0], pView[1], pView[2], pView[3], GL_BGR_EXT, GL_FLOAT, _data);//不是GL_RGB的读取方式，而是GL_BGR或者GL_BGR_Ext
	glReadBuffer(GL_BACK);
	for (int i = 0; i <bufferSize; i++)
	{
		data[i] = _data[i] * 255;
	}
	BITMAPFILEHEADER fileHeader;
	BITMAPINFOHEADER infoHeader;
	infoHeader.biSize = 40;
	infoHeader.biWidth = WIDTH;
	infoHeader.biHeight = HEIGHT;
	infoHeader.biPlanes = 1;
	infoHeader.biBitCount = 24;
	infoHeader.biCompression = BI_RGB;
	infoHeader.biSizeImage = pView[2] * pView[3] * 3;
	infoHeader.biXPelsPerMeter = 0;
	infoHeader.biYPelsPerMeter = 0;
	infoHeader.biClrUsed = 0;
	infoHeader.biClrImportant = 0;
	fileHeader.bfType = 0x4D42;
	fileHeader.bfReserved1 = 0;
	fileHeader.bfReserved2 = 0;
	fileHeader.bfOffBits = 54;
	fileHeader.bfSize = (DWORD)(sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + pView[2] * pView[3] * 3);
	FILE *fd;
	if (!(fd = fopen( picpath.c_str(), "wb+")))//filepath为你所保存文件的名字
	{
		//./AfxMessageBox("bmp图片申请出错");
		printf("bmp图片申请出错");
		exit(0);
	}
	else
	{
		fwrite(&fileHeader, 1, sizeof(BITMAPFILEHEADER), fd);
		fwrite(&infoHeader, 1, sizeof(BITMAPINFOHEADER), fd);
		fwrite(data, 1, pView[2] * pView[3] * 3, fd);
		fclose(fd);
	}
	delete[] data;
	delete[] _data;
}

#include "io.h"
using std::string;
using std::vector;
void getFiles(string path, vector<string>& files, vector<string> &ownname)
{
	/*files存储文件的路径及名称(eg.   C:\Users\WUQP\Desktop\test_devided\data1.txt)
	ownname只存储文件的名称(eg.     data1.txt)*/

	//文件句柄  
	long   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
		//	int i = strlen(fileinfo.name);
		//	i = sizeof(fileinfo.name);
		//	cout << i << endl;
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{  
			   if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)
			   getFiles( p.assign(path).append("\\").append(fileinfo.name), files, ownname );

			}
			else //if (fileinfo.name[strlen(fileinfo.name)-1] == 'g')//./
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
				ownname.push_back(fileinfo.name);
			}

		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

//分割字符串
//#include 
cv::Point2f split(const string &str, const string &pattern)
{
	if (str.length()<2)return cv::Point2f(-1,-1);//./
	//const char* convert to char*
	char * strc = new char[strlen(str.c_str()) + 1];
	strcpy(strc, str.c_str());
	vector<float> resultVec;
	char* tmpStr = strtok(strc, pattern.c_str());
	//./if(tmpStr != NULL)resultVec.push_back(atof(tmpStr));

	while (tmpStr != NULL)
	{
		resultVec.push_back(atof(tmpStr));
		tmpStr = strtok(NULL, pattern.c_str());
	}

	delete[] strc;

	cv::Point2f point;
	point.x = resultVec[0];
	point.y = resultVec[1];

	return point;
}
#if 0
// ==================================================
//	MAIN
// ==================================================
// ==================================================
//	MAIN
// ==================================================
int main(int argc, char** argv)
{
	SDL_Init(SDL_INIT_EVERYTHING);
	SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_BUFFER_SIZE, 32);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_Window* window = SDL_CreateWindow(TITLE, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, SDL_WINDOW_OPENGL);
	SDL_GLContext glContext = SDL_GL_CreateContext(window);
	glewInit();
	glViewport(0, 0, WIDTH, HEIGHT);
	glClearColor(0.8, 0.8, 0.8, 1);
	glEnable(GL_DEPTH_TEST);
	glFrontFace(GL_CCW);
	glCullFace(GL_BACK);
	glEnable(GL_CULL_FACE);

	GLuint vertexShader;
	GLuint fragmentShader;
	GLuint program;
	GLint success;
	GLchar infoLog[512];
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, VERTEX_SHADER_SOURCE, NULL);
	glCompileShader(vertexShader);
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		std::cerr << "ERROR: Vertex shader compilation failed.\n" << infoLog << std::endl;
	}
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, FRAGMENT_SHADER_SOURCE, NULL);
	glCompileShader(fragmentShader);
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		std::cerr << "ERROR: Fragment shader compilation failed.\n" << infoLog << std::endl;
	}
	program = glCreateProgram();
	glAttachShader(program, vertexShader);
	glAttachShader(program, fragmentShader);
	glLinkProgram(program);
	glGetProgramiv(program, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(program, 512, NULL, infoLog);
		std::cerr << "ERROR: Program linking failed.\n" << infoLog << std::endl;
	}
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	IndexedModel mesh = OBJModel("./res/female_face.obj").ToIndexedModel();
	mesh.CalcNormals();
	GLuint numVertices = mesh.positions.size();
	GLuint numIndices = mesh.indices.size();
	GLuint faceVAO;
	GLuint faceVBO;
	GLuint faceEBO;
	glGenVertexArrays(1, &faceVAO);
	glGenBuffers(1, &faceVBO);
	glGenBuffers(1, &faceEBO);
	glBindVertexArray(faceVAO);
	glBindBuffer(GL_ARRAY_BUFFER, faceVBO);
	glBufferData(GL_ARRAY_BUFFER, numVertices * sizeof(glm::vec3), &mesh.positions[0], GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, faceEBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, numIndices * sizeof(unsigned int), &mesh.indices[0], GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), (GLvoid*)0);
	glBindVertexArray(0);

	glm::mat4 model = glm::mat4(1.0);
	glm::mat4 view = view = glm::lookAt(glm::vec3(0, 0, 0), glm::vec3(0, 0, 1), glm::vec3(0, -1, 0));
	glm::mat4 projection = glm::perspective(70.0f, (float)WIDTH / (float)HEIGHT, 0.1f, 1000.0f);
	glm::mat4 viewProjection = projection * view;
	GLuint uMLocation = glGetUniformLocation(program, "M");
	GLuint uVPLocation = glGetUniformLocation(program, "VP");

	std::vector<cv::Point3f> modelPoints(modelPointsArr, modelPointsArr + sizeof(modelPointsArr) / sizeof(modelPointsArr[0]));

	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor sp;
	dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> sp;

	cv::Mat srcImageCV;
	dlib::image_window win;

	dlib::array2d<dlib::rgb_pixel> srcImageDLIB;
	std::vector<dlib::rectangle> dets;
	std::vector<dlib::full_object_detection> shapes;
	dlib::full_object_detection shape;

	srcImageCV = cv::imread("./res/wsx2.jpg");
	dlib::assign_image(srcImageDLIB, dlib::cv_image<dlib::bgr_pixel>(srcImageCV));
	dets = detector(srcImageDLIB);
	shape = sp(srcImageDLIB, dets[0]);
	std::vector<cv::Point2f> srcImagePoints;
	int i;
	i = 36; srcImagePoints.push_back(cv::Point2f(shape.part(i).x(), shape.part(i).y()));
	i = 39; srcImagePoints.push_back(cv::Point2f(shape.part(i).x(), shape.part(i).y()));
	i = 45; srcImagePoints.push_back(cv::Point2f(shape.part(i).x(), shape.part(i).y()));
	i = 42; srcImagePoints.push_back(cv::Point2f(shape.part(i).x(), shape.part(i).y()));
	i = 31; srcImagePoints.push_back(cv::Point2f(shape.part(i).x(), shape.part(i).y()));
	i = 35; srcImagePoints.push_back(cv::Point2f(shape.part(i).x(), shape.part(i).y()));
	i = 48; srcImagePoints.push_back(cv::Point2f(shape.part(i).x(), shape.part(i).y()));
	i = 54; srcImagePoints.push_back(cv::Point2f(shape.part(i).x(), shape.part(i).y()));
	i = 8;	srcImagePoints.push_back(cv::Point2f(shape.part(i).x(), shape.part(i).y()));
	shapes.push_back(shape);

	win.clear_overlay();
	win.set_image(srcImageDLIB);
	win.add_overlay(dlib::render_face_detections(shapes));

	glm::mat3 rotationMatrix;
	glm::vec3 translationVector;
	getTransformationParameters(srcImagePoints, modelPoints, srcImageCV, rotationMatrix, translationVector);
	glm::mat4 translationMatrix = glm::translate(translationVector);
	model = translationMatrix *
		glm::mat4(
			rotationMatrix[0][0], rotationMatrix[0][1], rotationMatrix[0][2], 0,
			rotationMatrix[1][0], rotationMatrix[1][1], rotationMatrix[1][2], 0,
			rotationMatrix[2][0], rotationMatrix[2][1], rotationMatrix[2][2], 0,
			0, 0, 0, 1
		);
	// std::cout << glm::to_string(translationVector) << std::endl;

	GLuint srcImageTexture;
	cv::flip(srcImageCV, srcImageCV, 0);
	glGenTextures(1, &srcImageTexture);
	glBindTexture(GL_TEXTURE_2D, srcImageTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, srcImageCV.cols, srcImageCV.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, srcImageCV.ptr());
	glGenerateMipmap(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);

	SDL_Event e;
	boolean run = true;
	float angle = 0.0f;
	while (run)
	{
		SDL_PollEvent(&e);
		if (e.type == SDL_QUIT)
			run = false;

		if (e.type == SDL_KEYDOWN)
		{
			switch (e.key.keysym.sym)
			{
			case SDLK_LEFT:
				angle -= 0.01;
				break;
			case SDLK_RIGHT:
				angle += 0.01;
				break;
			default:
				break;
			}
		}
		glm::mat4 r = glm::rotate(angle, glm::vec3(0, -1, 0));

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glUseProgram(program);
		glProgramUniformMatrix4fv(program, uVPLocation, 1, false, glm::value_ptr(viewProjection));
		glProgramUniformMatrix4fv(program, uMLocation, 1, false, glm::value_ptr(model * r));

		glBindTexture(GL_TEXTURE_2D, 0);
		glBindVertexArray(faceVAO);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);

		glBindTexture(GL_TEXTURE_2D, srcImageTexture);
		glBindVertexArray(faceVAO);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);

		SDL_GL_SwapWindow(window);
	}

	glDeleteBuffers(1, &faceEBO);
	glDeleteBuffers(1, &faceVBO);
	glDeleteVertexArrays(1, &faceVAO);

	SDL_GL_DeleteContext(glContext);
	SDL_DestroyWindow(window);
	SDL_Quit();

	return 0;
}
#else
// ==================================================
//	MAIN
// ==================================================
int main1(vector<string>& ccompath,vector<string >& cfilenames)//, SDL_Window* window=NULL)
{

	// initialization----------------------------------------------------------------------
	SDL_Init(SDL_INIT_EVERYTHING);
	SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_BUFFER_SIZE, 32);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_Window* window = SDL_CreateWindow(TITLE, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, SDL_WINDOW_OPENGL);
	SDL_GLContext glContext = SDL_GL_CreateContext(window);
	glewInit();
	glViewport(0, 0, WIDTH, HEIGHT);
	glClearColor(0.8, 0.8, 0.8, 1);
	glEnable(GL_DEPTH_TEST);
	glFrontFace(GL_CCW);
	//glFrontFace(GL_CW);
	glCullFace(GL_BACK);
	//glCullFace(GL_FRONT_AND_BACK);

	//./glEnable(GL_CULL_FACE);	

	GLuint vertexShader;
	GLuint fragmentShader;
	GLuint program;
	GLint success;
	GLchar infoLog[512];
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, VERTEX_SHADER_SOURCE, NULL);
	glCompileShader(vertexShader);
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		std::cerr << "ERROR: Vertex shader compilation failed.\n" << infoLog << std::endl;
	}
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, FRAGMENT_SHADER_SOURCE, NULL);
	glCompileShader(fragmentShader);
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		std::cerr << "ERROR: Fragment shader compilation failed.\n" << infoLog << std::endl;
	}
	program = glCreateProgram();
	glAttachShader(program, vertexShader);
	glAttachShader(program, fragmentShader);
	glLinkProgram(program);
	glGetProgramiv(program, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(program, 512, NULL, infoLog);
		std::cerr << "ERROR: Program linking failed.\n" << infoLog << std::endl;
	}
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
	//=============================以上为着色器程序的准备阶段======================================================

	Shader lightingShader("2.2.basic_lighting.vs", "2.2.basic_lighting.fs");

	//=============================以上为ywl所建着色器程序的准备阶段======================================================
	//人脸模型
	IndexedModel mesh = OBJModel("./res/female_face_offical.obj").ToIndexedModel();//./male_face_eyes.obj
	mesh.CalcNormals();
	GLuint numVertices = mesh.positions.size();
	GLuint numIndices = mesh.indices.size();
	GLuint faceVAO;
	GLuint faceVBO;
	GLuint faceEBO;
	glGenVertexArrays(1, &faceVAO);
	glGenBuffers(1, &faceVBO);
	glGenBuffers(1, &faceEBO);
	glBindVertexArray(faceVAO);
	glBindBuffer(GL_ARRAY_BUFFER, faceVBO);
	glBufferData(GL_ARRAY_BUFFER, numVertices * sizeof(glm::vec3), &mesh.positions[0], GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, faceEBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, numIndices * sizeof(unsigned int), &mesh.indices[0], GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), (GLvoid*)0);
	glBindVertexArray(0);




	//镜框模型20_12
	#define  gn 6
	IndexedModel mesh1[gn] = {
		OBJModel("./res/glasses/1jingkuang.obj").ToIndexedModel(),
		OBJModel("./res/glasses/2jingkuang.obj").ToIndexedModel(),
		OBJModel("./res/glasses/4jingkuang.obj").ToIndexedModel(),
		OBJModel("./res/glasses/8jingkuang.obj").ToIndexedModel(),
		OBJModel("./res/glasses/11jingkuang.obj").ToIndexedModel(),
		OBJModel("./res/glasses/7jingkuang.obj").ToIndexedModel(),
		//OBJModel("./res/glasses/2jingkuang.obj").ToIndexedModel(),
		//OBJModel("./res/glasses/2jingkuang.obj").ToIndexedModel(),
		//OBJModel("./res/glasses/2jingkuang.obj").ToIndexedModel(),
	};
	GLuint jkVBO[gn * 3], jkVAO[gn]; 
	GLuint jknumVertices[gn] = { 0 };
	GLuint jknumIndices[gn] = {0};

	
	glGenVertexArrays(gn, jkVAO);
	glGenBuffers(3*gn, jkVBO);

	for (int i=0;i<gn;i++)
	{
		mesh1[i].CalcNormals();

		jknumVertices[i] = mesh1[i].positions.size();
		jknumIndices[i] = mesh1[i].indices.size();

		glBindVertexArray(jkVAO[i]);
		glBindBuffer(GL_ARRAY_BUFFER, jkVBO[i*3]);
		glBufferData(GL_ARRAY_BUFFER, jknumVertices[i] * sizeof(glm::vec3), &mesh1[i].positions[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), (GLvoid*)0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ARRAY_BUFFER, jkVBO[i * 3+1]);
		glBufferData(GL_ARRAY_BUFFER, mesh1[i].normals.size() * sizeof(glm::vec3), &mesh1[i].normals[0], GL_STATIC_DRAW);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), (GLvoid*)0);
		glEnableVertexAttribArray(1);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, jkVBO[i * 3+2]);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, jknumIndices[i] * sizeof(unsigned int), &mesh1[i].indices[0], GL_STATIC_DRAW);

		glBindVertexArray(0);
		
	}	

	//镜片模型
	IndexedModel mesh2[gn] = {
		OBJModel("./res/glasses/1jingpian.obj").ToIndexedModel() ,
		OBJModel("./res/glasses/2jingpian.obj").ToIndexedModel(),
		OBJModel("./res/glasses/4jingpian.obj").ToIndexedModel(),
		OBJModel("./res/glasses/8jingpian.obj").ToIndexedModel(),
		OBJModel("./res/glasses/11jingpian.obj").ToIndexedModel(),
		OBJModel("./res/glasses/7jingpian.obj").ToIndexedModel(),
		//OBJModel("./res/glasses/8jingpian.obj").ToIndexedModel(),
		//OBJModel("./res/glasses/2jingpian.obj").ToIndexedModel(),
		//OBJModel("./res/glasses/2jingpian.obj").ToIndexedModel(),
	};
	GLuint jpVBO[gn * 2], jpVAO[gn];
	GLuint jpnumVertices[gn] = { 0 };
	GLuint jpnumIndices[gn] = { 0 };


	glGenVertexArrays(gn, jpVAO);
	glGenBuffers(2 * gn, jpVBO);

	for (int i = 0; i<gn; i++)
	{
		mesh2[i].CalcNormals();

		jpnumVertices[i] = mesh2[i].positions.size();
		jpnumIndices[i] = mesh2[i].indices.size();

		glBindVertexArray(jpVAO[i]);
		glBindBuffer(GL_ARRAY_BUFFER, jpVBO[i *2]);
		glBufferData(GL_ARRAY_BUFFER, jpnumVertices[i] * sizeof(glm::vec3), &mesh2[i].positions[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), (GLvoid*)0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, jpVBO[i * 2 +1]);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, jpnumIndices[i] * sizeof(unsigned int), &mesh2[i].indices[0], GL_STATIC_DRAW);

		glBindVertexArray(0);

	}

	srand((int)time(0));


		
//////////////////////////////////////////

#if 0
	

	IndexedModel mesh1_0 = OBJModel("./res/glasses/20_12.obj").ToIndexedModel();
	mesh1_0.CalcNormals();

	GLuint VBO[3], jingkuanVAO;

	glGenVertexArrays(1, &jingkuanVAO);
	glGenBuffers(3, VBO);
	glBindVertexArray(jingkuanVAO);

	GLuint numVertices1 = mesh1_0.positions.size();
	GLuint numIndices1 = mesh1_0.indices.size();

	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
	glBufferData(GL_ARRAY_BUFFER, numVertices1 * sizeof(glm::vec3), &mesh1_0.positions[0], GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), (GLvoid*)0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
	glBufferData(GL_ARRAY_BUFFER, mesh1_0.normals.size() * sizeof(glm::vec3), &mesh1_0.normals[0], GL_STATIC_DRAW);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), (GLvoid*)0);
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBO[2]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, numIndices1 * sizeof(unsigned int), &mesh1_0.indices[0], GL_STATIC_DRAW);

	glBindVertexArray(0);


	//镜片模型
	IndexedModel mesh2_0 = OBJModel("./res/glasses/20_11.obj").ToIndexedModel();//./male_face_eyes.obj
	mesh2_0.CalcNormals();
	GLuint numVertices2 = mesh2_0.positions.size();
	GLuint numIndices2 = mesh2_0.indices.size();
	GLuint faceVAO2;
	GLuint faceVBO2;
	GLuint faceEBO2;
	glGenVertexArrays(1, &faceVAO2);
	glGenBuffers(1, &faceVBO2);
	glGenBuffers(1, &faceEBO2);
	glBindVertexArray(faceVAO2);
	glBindBuffer(GL_ARRAY_BUFFER, faceVBO2);
	glBufferData(GL_ARRAY_BUFFER, numVertices2 * sizeof(glm::vec3), &mesh2_0.positions[0], GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, faceEBO2);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, numIndices2 * sizeof(unsigned int), &mesh2_0.indices[0], GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), (GLvoid*)0);
	glBindVertexArray(0);
	
	
#endif


	//=====================以上为VAO/VBO/EBO的设置============================================================



	//where loop start
	vector<string > completepath;
	vector<string > filename;

	//./参数形式有外部传入
	if (1)
	{	
		completepath = ccompath;
		filename = cfilenames;

	}
	else
	{
		string picspath = "G:/Downloads/vggface2_test/test/300";// "C:/Users/123/Desktop/bugpics";// "res/vggface2train/3";// "res/n000259";
		getFiles(picspath, completepath, filename);//"res/testprn"//smallscale//"res/test"//imm_face_db./路径末尾可以不用分隔符

	}

	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor sp;
	dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> sp;

	//dlib::array2d<dlib::rgb_pixel> srcImageDLIB;
	//std::vector<dlib::rectangle> dets;

	std::vector<dlib::full_object_detection> shapes;
	dlib::full_object_detection shape;

	std::vector<cv::Point3f> modelPoints(modelPointsArr, modelPointsArr + sizeof(modelPointsArr) / sizeof(modelPointsArr[0]));

	for (int index = 0; index < completepath.size(); index++)
	{
		//-----------------------------------------------以下为人脸特征点检测---------------------

	//dlib::image_window win;

	dlib::array2d<dlib::rgb_pixel> srcImageDLIB;
	std::vector<dlib::rectangle> dets;
		//face1.jpg//二丫
		//"4500_1864_18.jpg";//长脸树下长发女
		//"4500_2198_5.jpg";//老八字脸男
		//"4500_809_13.jpg";//白衣女


		std::string picpath = filename[index]; //./"4500_2357_10.jpg";//argv[1];// "4500_1864_18.jpg";// "4500_809_13.jpg";//"4500_2198_5.jpg";// 
		cv::Mat srcImageCV = cv::imread(completepath[index]);//org
		//./cv::Mat srcImageCV0=srcImageCV.clone();
		//srcImageCV = cv::imread("./res/wsx3.jpg");	
try{
		if (0)
		{
			dlib::assign_image(srcImageDLIB, dlib::cv_image<dlib::bgr_pixel>(srcImageCV));
			dets = detector(srcImageDLIB);

			cv::Point center = cv::Point((dets[0].left() + dets[0].right()) / 2, (dets[0].top() + dets[0].bottom()) / 2);
			int width = srcImageCV.cols - center.x;
			int height = srcImageCV.rows - center.y;
			int wh = std::min(std::min(width, height), 4000);
			wh = std::min(std::min(center.x, center.y), wh);

			//cv::Mat srcroi = srcImageCV(cv::Rect(center.x-wh,center.y-wh,2*wh,2*wh));
			cv::Mat srcroi = srcImageCV(cv::Rect(center.x - wh, center.y - wh, 2 * wh, 2 * wh)).clone();
			//cv::imwrite("ywl/srcroi_1.jpg", srcroi);

			dets.clear();
			srcImageDLIB.clear();
			dlib::assign_image(srcImageDLIB, dlib::cv_image<dlib::bgr_pixel>(srcroi));
			dets = detector(srcImageDLIB);

			srcImageCV = srcroi.clone();
		}

		//if (0) {

		//	int hw = std::min(srcImageCV.rows, srcImageCV.cols);

		//	cv::Mat srcroi = srcImageCV(cv::Rect(0, 0, hw, hw)).clone();
		//	cv::Mat dst;
		//	cv::resize(srcroi, dst, cv::Size(WIDTH, HEIGHT), (0, 0), (0, 0), cv::INTER_LINEAR);
		//	cv::imwrite("ywl/dst_2.jpg", dst);

		//	dets.clear();
		//	srcImageDLIB.clear();
		//	dlib::assign_image(srcImageDLIB, dlib::cv_image<dlib::bgr_pixel>(dst));
		//	dets = detector(srcImageDLIB);

		//	srcImageCV = dst.clone();
		//}

		int savecols = 0;
		int saverows = 0;
		if (01)
		{
			int hw = std::min(srcImageCV.rows, srcImageCV.cols);

			if (srcImageCV.rows > srcImageCV.cols)
			{
				savecols = HEIGHT* ((float)srcImageCV.cols / srcImageCV.rows);
				saverows = HEIGHT;

			}
			else {
				saverows = WIDTH* ((float)srcImageCV.rows / srcImageCV.cols);
				savecols = WIDTH;
			}
			//cv::Mat srcroi = srcImageCV.clone();
			cv::Mat dst;
			//WIDTH = srcImageCV.cols ;
			//HEIGHT = srcImageCV.rows;
			cv::resize(srcImageCV, dst, cv::Size(WIDTH, HEIGHT), 0, 0, cv::INTER_LINEAR);
			//cv::imwrite("./prnet/" + filename[index], dst);
			dets.clear();
			srcImageDLIB.clear();
			dlib::assign_image(srcImageDLIB, dlib::cv_image<dlib::bgr_pixel>(dst));
			dets = detector(srcImageDLIB);

			srcImageCV = dst.clone();
		}
		//./shape.clear();
		
		shape = sp(srcImageDLIB, dets[0]);


		std::vector<cv::Point2f> srcImagePoints;
		int i=0;
		int kpt[] = {
			36,39,45,42,	31,35,	48,54,	8,
			37,38, 40,41, 43,44, 46,47,
			0,16,
			5,11,
			4,12,
			27,28,29,
			30,
			//19,24
		};
		for (;i<sizeof(kpt)/sizeof(kpt[0]);i++)
		{
			srcImagePoints.push_back(cv::Point2f(shape.part(kpt[i]).x(), shape.part(kpt[i]).y()));
		}
		//i = 36; srcImagePoints.push_back(cv::Point2f(shape.part(i).x(), shape.part(i).y()));
		//i = 39; srcImagePoints.push_back(cv::Point2f(shape.part(i).x(), shape.part(i).y()));
		//i = 45; srcImagePoints.push_back(cv::Point2f(shape.part(i).x(), shape.part(i).y()));
		//i = 42; srcImagePoints.push_back(cv::Point2f(shape.part(i).x(), shape.part(i).y()));
		//i = 31; srcImagePoints.push_back(cv::Point2f(shape.part(i).x(), shape.part(i).y()));
		//i = 35; srcImagePoints.push_back(cv::Point2f(shape.part(i).x(), shape.part(i).y()));
		//i = 48; srcImagePoints.push_back(cv::Point2f(shape.part(i).x(), shape.part(i).y()));
		//i = 54; srcImagePoints.push_back(cv::Point2f(shape.part(i).x(), shape.part(i).y()));
		shapes.clear();
		shapes.push_back(shape);


		if (0)//./从prnet生成的txt中读取关键点位置
		{

			vector<cv::Point2f > ivec;
			string s;
			string FILE_NAME = /*"prnet/"*/"G:/Downloads/PRNet-master/TestImages/results/scale/" + filename[index].substr(0, filename[index].length() - 4) + "_kpt.txt";

			std::ifstream input_file;
			input_file.open(FILE_NAME, std::ifstream::in);
			if (!input_file)
			{
				std::cout << "open file fail" << filename[index] << endl;
				continue;
			}
			cv::Point2f keypt;
			while (!input_file.eof())
			{
				getline(input_file, s);
				keypt = split(s, " ");
				ivec.push_back(keypt);
				//.std::cout << kpt << std::endl;

			}
			input_file.close();

			//重新装载关键点
			srcImagePoints.clear();
			int i = 0;

			for (; i<sizeof(kpt) / sizeof(kpt[0]); i++)
			{
				srcImagePoints.push_back(ivec[kpt[i]]);
			}
			/*
			i = 36; srcImagePoints.push_back(ivec[i]);
			i = 39; srcImagePoints.push_back(ivec[i]);
			i = 45; srcImagePoints.push_back(ivec[i]);
			i = 42; srcImagePoints.push_back(ivec[i]);
			i = 31; srcImagePoints.push_back(ivec[i]);
			i = 35; srcImagePoints.push_back(ivec[i]);
			i = 48; srcImagePoints.push_back(ivec[i]);
			i = 54; srcImagePoints.push_back(ivec[i]);
			i = 8;	srcImagePoints.push_back(ivec[i]);
*/
		}
		if (01)//./在图片中显示关键点位置//显示了11个人脸的关键点后关闭开关为0
		{
			cv::Mat tmp = srcImageCV.clone();
			for (int i = 0; i < srcImagePoints.size(); i++)
			{
				cv::circle(tmp, srcImagePoints[i], 4, cv::Scalar(1, 0, 0), 3);

			}
			imwrite("9dian/" + filename[index], tmp);//fromprn

		}
		if (0)//./窗口显示关键点连线
		{
			dlib::image_window win;

			win.clear_overlay();
			win.set_image(srcImageDLIB);
			win.add_overlay(dlib::render_face_detections(shapes));

			//./getchar();
		}

		//-----------------------------------------------以上为人脸特征点检测---------------------

	//./此处移走了关键点检测

		glm::mat3 rotationMatrix;
		glm::vec3 translationVector;
		getTransformationParameters(srcImagePoints, modelPoints, srcImageCV, rotationMatrix, translationVector);
		glm::mat4 translationMatrix = glm::translate(translationVector);

		glm::mat4 model = glm::mat4(1.0);
		//glm::mat4 view = view = glm::lookAt(glm::vec3(2, 0, 2), glm::vec3(0, 0, 1), glm::vec3(0, -1, 0));
		glm::mat4 view = view = glm::lookAt(glm::vec3(0, 0, 0), glm::vec3(0, 0, 1), glm::vec3(0, -1, 0));


		glm::mat4 projection = glm::perspective(70.0f, (float)WIDTH / (float)HEIGHT, 0.1f, 1000.0f);
		glm::mat4 viewProjection = projection * view;
		GLuint uMLocation = glGetUniformLocation(program, "M");
		GLuint uVPLocation = glGetUniformLocation(program, "VP");
		GLuint uv2Location = glGetUniformLocation(program, "v2");

		//GLuint uywlcolorLocation = glGetUniformLocation(program, "ywlcolor");

		if (0)//./从prn生成的旋转矩阵中直接读数
		{
			rotationMatrix = glm::mat3(
				9.048252183258692627e-01, 8.865341376608569524e-02, 4.164515536159785203e-01,
				-1.108722903287271688e-01, 9.849315704571137697e-01, 1.327295624725171874e-01,
				-3.984093539014230134e-01, -1.662699929028527424e-01, 9.010201303046054822e-01
			);
		}

		glm::mat4 rotationMat4 = glm::mat4(
			rotationMatrix[0][0], rotationMatrix[0][1], rotationMatrix[0][2], 0,
			rotationMatrix[1][0], rotationMatrix[1][1], rotationMatrix[1][2], 0,
			rotationMatrix[2][0], rotationMatrix[2][1], rotationMatrix[2][2], 0,
			0, 0, 0, 1
		);


		model = translationMatrix *
			glm::mat4(
				rotationMatrix[0][0], rotationMatrix[0][1], rotationMatrix[0][2], 0,
				rotationMatrix[1][0], rotationMatrix[1][1], rotationMatrix[1][2], 0,
				rotationMatrix[2][0], rotationMatrix[2][1], rotationMatrix[2][2], 0,
				0, 0, 0, 1
			);
		//glm::mat4 bgmodel = glm::translate(translationMatrix, glm::vec3(0.5, -2.5f, 6.f));
		glm::mat4 bgmodel = glm::translate(translationMatrix, glm::vec3(0., 0., 0.));
		//./bgmodel= bgmodel *rotationMat4;
		bgmodel = glm::rotate(bgmodel, glm::radians(180.0f), glm::vec3(0, -1, 0));

		if (0)//./输出model矩阵和关键点
		{
			std::ofstream out("rotate.txt", std::ios::out | std::ios::app);

			out << filename[index] << std::endl << glm::to_string(translationVector) << std::endl;
			out << glm::to_string(rotationMat4) << std::endl << std::endl;
			for (int i = 0; i < 68; i++)
			{
				if (0 == i % 3) out << std::endl;
				out << cv::Point2f(shape.part(i).x(), shape.part(i).y());
			}
			out.close();

		}
		//std::cout << glm::to_string(translationVector) << std::endl;
		glm::vec4 offset = rotationMat4 * glm::vec4(0, -0.65, 0.65, 1.);

		glm::vec4 normv = rotationMat4 * glm::vec4(0, -0., -1., 1.);//./face normal vector

		glm::mat4 model2 = glm::mat4(translationMatrix);
		if (01)
		{
			model2[3][0] += offset[0];
			model2[3][1] += offset[1];
			model2[3][2] += offset[2];
		}
		//model2 = glm::translate(model2, glm::vec3(0, 0.5f, -1.5f));
		model2 = model2*rotationMat4;
		model2 = glm::rotate(model2, glm::radians(88.0f), glm::vec3(0.0f, 1.0f, 0.0f));

		//./model2 = glm::scale(model2, glm::vec3(0.025, 0.025, 0.025));

		//glm::mat4 model0 = glm::mat4(1.0);
		//model0 = glm::scale(model0, glm::vec3(0.05, 0.05, 0.05));
		//glm::mat4 model01 = glm::mat4(1.0);
		//model01 = glm::rotate(model0, glm::radians(89.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		//model01 = glm::scale(model01, glm::vec3(0.02, 0.02, 0.02));
		//model01 = glm::translate(model01, glm::vec3(-100.4, -30.4f, 6.f));


		GLuint srcImageTexture;
		cv::flip(srcImageCV, srcImageCV, 0);
		glGenTextures(1, &srcImageTexture);
		glBindTexture(GL_TEXTURE_2D, srcImageTexture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, srcImageCV.cols, srcImageCV.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, srcImageCV.ptr());
		glGenerateMipmap(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, 0);


		boolean run = true;
		float angle = 0.0f; //./getchar();

		//SDL_Event e;
		//./for (int i=0;i<completepath.size();i++)

		//}
		//while (run)
		//{
		//	SDL_PollEvent(&e);
		//	if (e.type == SDL_QUIT)
		//		run = false;

		//	if (e.type == SDL_KEYDOWN)
		//	{
		//		switch (e.key.keysym.sym)
		//		{
		//		case SDLK_LEFT:
		//			angle -= 0.001;
		//			break;
		//		case SDLK_RIGHT:
		//			angle += 0.001;
		//			break;
		//		default:
		//			break;
		//		}
		//	}
		glm::mat4 r = glm::rotate(angle, glm::vec3(0, -1, 0));
		//./glm::mat4 r = glm::rotate(angle, translationVector);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


		glUseProgram(program);
		//glProgramUniform3f(program, uywlcolorLocation, 0.0f, 0.0f, 0.0f);
		glProgramUniformMatrix4fv(program, uVPLocation, 1, false, glm::value_ptr(viewProjection));
		glProgramUniform2f(program, uv2Location, 0., 0.);

		if (01)//./产生背景,大人脸
		{
			glProgramUniformMatrix4fv(program, uMLocation, 1, false, glm::value_ptr(glm::scale(bgmodel*r, glm::vec3(3.0))));

			//人脸网格模型
			//glBindTexture(GL_TEXTURE_2D, 0);
			//glBindVertexArray(faceVAO);
			//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			//glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, 0);
			//glBindVertexArray(0);

			//人脸纹理
			glBindTexture(GL_TEXTURE_2D, srcImageTexture);
			glBindVertexArray(faceVAO);
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, 0);
			glBindVertexArray(0);


			//./旋转90度，补上眼睛
			//glProgramUniformMatrix4fv(program, uVPLocation, 1, false, glm::value_ptr(viewProjection));
			//glProgramUniformMatrix4fv(program, uMLocation, 1, false, glm::value_ptr(glm::scale(glm::rotate(bgmodel, glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f))*r, glm::vec3(5.0))));

			////人脸网格模型
			////glBindTexture(GL_TEXTURE_2D, 0);
			////glBindVertexArray(faceVAO);
			////glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			////glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, 0);
			////glBindVertexArray(0);

			////人脸纹理
			//glBindTexture(GL_TEXTURE_2D, srcImageTexture);
			//glBindVertexArray(faceVAO);
			//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			//glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, 0);
			//glBindVertexArray(0);
		}
#define verification false
		if (01)//./小人脸
		{
			//glm::mat4 viewProjectionface = glm::mat4(1);

			//glm::mat4 modelface = glm::mat4(
			//	1.270731617527997681e+00, 1.245043723281646919e-01, 5.848622978563926544e-01, 2.061771794007457288e+02,
			//	-1.636940275333602923e-01, 1.454172319655854695e+00, 1.959645335137231159e-01, -8.369296135180371721e+01,
			//	-5.459203668503185281e-01, -1.960814510647252984e-01, 1.273380064882728702e+00, 1.017921309329269661e+02,
			//	0,0,0,1
			//);
			//glUseProgram(program);
			//glProgramUniform3f(program, uywlcolorLocation, 0.0f, 0.0f, 0.0f);
			//glProgramUniformMatrix4fv(program, uVPLocation, 1, false, glm::value_ptr(viewProjectionface));
			//glProgramUniformMatrix4fv(program, uMLocation, 1, false, glm::value_ptr(modelface));

			glProgramUniformMatrix4fv(program, uMLocation, 1, false, glm::value_ptr(model * r));


			//人脸网格模型
			if(verification)
			{
			glBindTexture(GL_TEXTURE_2D, 0);
			glBindVertexArray(faceVAO);
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, 0);
			glBindVertexArray(0);
			 }
			
			//人脸纹理
			glBindTexture(GL_TEXTURE_2D, srcImageTexture);
			glBindVertexArray(faceVAO);
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, 0);
			glBindVertexArray(0);

			//人脸纹理45度
			//glProgramUniformMatrix4fv(program, uMLocation, 1, false, glm::value_ptr(glm::rotate(glm::translate(translationMatrix, glm::vec3(0., 0., 0.))*rotationMat4, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)) * r));
			//glBindTexture(GL_TEXTURE_2D, srcImageTexture);
			//glBindVertexArray(faceVAO);
			//glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, 0);
			//glBindVertexArray(0);
		}

		//坐标轴三角形
		//glUseProgram(program);
		//glProgramUniform3f(program, uywlcolorLocation, 1.0f, 0.0f, 0.0f);
		//glProgramUniformMatrix4fv(program, uVPLocation, 1, false, glm::value_ptr(glm::vec4(1.0f)));
		//glProgramUniformMatrix4fv(program, uMLocation, 1, false, glm::value_ptr(glm::vec4(1.0f)));
		//glBindVertexArray(faceVAO3);
		//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		//glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);//./glDrawArrays(GL_TRIANGLES, 0, 6);
		//glBindVertexArray(0);

		int random=rand() % gn;
#define thresh 0.2

		//眼镜片
		if (!verification)
		{
			//float fx = normv.x < -0.3 ? normv.x*0.5 - 0.15 : normv.x>0.3 ? normv.x*0.5 + 0.15 : normv.x;
			float fx = normv.x < -thresh ? -thresh : normv.x>thresh ? thresh : normv.x;

			glUseProgram(program);
			//glProgramUniform2f(program, uv2Location, normv.x, normv.y);
			glProgramUniform2f(program, uv2Location, fx, normv.y);
			
			glProgramUniformMatrix4fv(program, uMLocation, 1, false, glm::value_ptr(model));
			//glProgramUniformMatrix4fv(program, uMLocation, 1, false, glm::value_ptr(glm::scale(model2 * r, glm::vec3(0.02))));
			glBindTexture(GL_TEXTURE_2D, srcImageTexture);
			glBindVertexArray(jpVAO[random]);
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			glDrawElements(GL_TRIANGLES, jpnumIndices[random], GL_UNSIGNED_INT, 0);
			glBindVertexArray(0);
			//glBindVertexArray(faceVAO2);
			//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			//glDrawElements(GL_TRIANGLES, numIndices1, GL_UNSIGNED_INT, 0);
			//glBindVertexArray(0);
		}

		//镜框
		if (!verification)
		{
		lightingShader.use();
		//./lightingShader.setVec3("objectColor", 0.1, 0.1, 0.1);//1.0f, 0.5f, 0.31f

		//./lightingShader.setVec3("lightColor", 1.0f, 1.0f, 1.0f);
		//./lightingShader.setVec3("viewPos", glm::vec3(0, 0, 0));
		lightingShader.setVec3("lightPos", glm::vec3(0, 0.5, 0.5));

		lightingShader.setMat4("pv", viewProjection);
		//lightingShader.setMat4("view", view);

		// world transformation
		//glm::mat4 model;
		lightingShader.setMat4("model", model/*glm::scale(model2 * r, glm::vec3(0.02))*/);

		// render the cube
		
		glBindVertexArray(jkVAO[random]);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glDrawElements(GL_TRIANGLES, jknumIndices[random], GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);	
		
		/*glBindVertexArray(jingkuanVAO);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glDrawElements(GL_TRIANGLES, numIndices1, GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);*/
		}


		SDL_GL_SwapWindow(window);

		ScreenShot("tmp/" + picpath);

		cv::Mat dst;
		cv::Mat srct = cv::imread("tmp/" + picpath);
		//WIDTH = srcImageCV.cols ;
		//HEIGHT = srcImageCV.rows;
		cv::resize(srct, dst, cv::Size(savecols, saverows), 0, 0, cv::INTER_LINEAR);
		//./cv::resize(srcroi, dst, cv::Size(300,400), (0, 0), (0, 0), cv::INTER_LINEAR);
		//string savepath = completepath[index].substr(0, strlen(completepath[index].c_str()) - strlen(filename[index].c_str()) -1)+"_result";
		
		if (0)//old
		{
		
		string savepath = completepath[index].substr(0, completepath[index].rfind('/')) + "/result";
		_mkdir(savepath.c_str());
		savepath=savepath+completepath[index].substr(completepath[index].find('\\'), completepath[index].rfind('\\')- completepath[index].find('\\'));
		_mkdir(savepath.c_str());
		savepath = savepath + "/"+filename[index];
		//savepath = savepath + "/" + filename[index];
		cv::imwrite(savepath, dst);
		//cv::imwrite(picspath + "_result/" + picpath, dst);
		}
		else//g盘满了空间，转了盘
		{
			string savepath = "E:/vggface2_train_xa2/result" + completepath[index].substr(completepath[index].find('\\'), completepath[index].rfind('\\') - completepath[index].find('\\'));
			_mkdir(savepath.c_str());
			savepath = savepath + "/" + filename[index];
			cv::imwrite(savepath, dst);
		}
		

		srcImageCV.release();
		srct.release();
		dst.release();

	}

	//__except (EXCEPTION_EXECUTE_HANDLER)  	//
	catch (...)
	{
		cout << completepath[index] << endl;//baocunfilename
		srcImageCV.release();
		dets.clear();
		srcImageDLIB.clear();
		std::ofstream out("trainf3000.txt", std::ios::out | std::ios::app);

		out << completepath[index] << std::endl;

		out.close();
	}
}
	glDeleteBuffers(1, &faceEBO);
	glDeleteBuffers(1, &faceVBO);
	glDeleteVertexArrays(1, &faceVAO);

	glDeleteBuffers(gn * 3, jkVBO);
	glDeleteVertexArrays(gn , jkVAO);

	glDeleteBuffers(gn * 2, jpVBO);
	glDeleteVertexArrays(gn , jpVAO);
#if 0
	glDeleteBuffers(1, &VBO[2]);

	glDeleteBuffers(1, &faceEBO2);
	glDeleteBuffers(1, &faceVBO2);
	glDeleteVertexArrays(1, &faceVAO2);

	glDeleteBuffers(2, VBO);
	glDeleteVertexArrays(1, &jingkuanVAO);
#endif


	SDL_GL_DeleteContext(glContext);
	SDL_DestroyWindow(window);
	SDL_Quit();

	return 0;
}

int main(int argc, char** argv)
{
	//// initialization----------------------------------------------------------------------
	//SDL_Init(SDL_INIT_EVERYTHING);
	//SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
	//SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
	//SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
	//SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);
	//SDL_GL_SetAttribute(SDL_GL_BUFFER_SIZE, 32);
	//SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16);
	//SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	//SDL_Window* window = SDL_CreateWindow(TITLE, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, SDL_WINDOW_OPENGL);
	//SDL_GLContext glContext = SDL_GL_CreateContext(window);

	const int minbatch = 5000;// 2500;
	vector<string > completepath;
	vector<string > filename;

	
	string picspath;
	if (1==argc)
	{	
		picspath = "C:/Users/123/Desktop/1/org";//"G:/Downloads/PRNet-master/scale";// "G:/Downloads/vggface2_train/train";// "C:/Users/123/Desktop/bugpics";// "res/vggface2train/3";// "res/n000259";

	}
	else
	{
		picspath = argv[1];
	}

	getFiles(picspath, completepath, filename);//"res/testprn"//smallscale//"res/test"//imm_face_db./路径末尾可以不用分隔符
	int i = 0;
	for (;minbatch * i + minbatch<completepath.size();i++)
	{
		vector<string > completepath_minbatch(completepath.begin()+ minbatch*i, completepath.begin() + minbatch * i+ minbatch);
		vector<string > filename_minbatch(filename.begin() + minbatch*i, filename.begin() + minbatch * i + minbatch);
		
		cout << "begin:	"<<i<<completepath[minbatch*i] << endl;
		main1(completepath_minbatch, filename_minbatch);//, window);

	}

	cout << "end:	" <<i<< completepath[minbatch*i] << endl;

	if(0==i)
	{ 
		vector<string > completepath_minbatch(completepath.begin(), completepath.end());
		vector<string > filename_minbatch(filename.begin(), filename.end());
		main1(completepath_minbatch, filename_minbatch);//, window);
	
	}
	else
	{
		vector<string > completepath_minbatch(completepath.end() - minbatch, completepath.end());
		vector<string > filename_minbatch(filename.end() - minbatch, filename.end());
		main1(completepath_minbatch, filename_minbatch);//, window);
	}
	//SDL_GL_DeleteContext(glContext);
	//SDL_DestroyWindow(window);
	//SDL_Quit();

	return 0;
}

//直接简单的函数调用就行，每次调用结束，内存都会释放一大波出来。所以可以不必使用system调用方式
int main_callsystem(int argc, char** argv)
{
	//// initialization----------------------------------------------------------------------
	//SDL_Init(SDL_INIT_EVERYTHING);
	//SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
	//SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
	//SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
	//SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);
	//SDL_GL_SetAttribute(SDL_GL_BUFFER_SIZE, 32);
	//SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16);
	//SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	//SDL_Window* window = SDL_CreateWindow(TITLE, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, SDL_WINDOW_OPENGL);
	//SDL_GLContext glContext = SDL_GL_CreateContext(window);

	const int minbatch = 2000;
	vector<string > completepath;
	vector<string > filename;

	string picspath = "G:/Downloads/vggface2_test/test/300";// "C:/Users/123/Desktop/bugpics";// "res/vggface2train/3";// "res/n000259";
	getFiles(picspath, completepath, filename);//"res/testprn"//smallscale//"res/test"//imm_face_db./路径末尾可以不用分隔符
	int i = 0;
	for (; minbatch * i + minbatch<completepath.size(); i++)
	{
		vector<string > completepath_minbatch(completepath.begin() + minbatch*i, completepath.begin() + minbatch * i + minbatch);
		vector<string > filename_minbatch(filename.begin() + minbatch*i, filename.begin() + minbatch * i + minbatch);

		std::ofstream out("30identities.txt", std::ios::out);// | std::ios::app);
		for (int index=0; index<completepath_minbatch.size(); index++)
		{
			out << completepath_minbatch[index] << std::endl;

		}
		out.close();

		cout << "begin:	" << i << completepath[minbatch*i] << endl;
		main1(completepath_minbatch, filename_minbatch);//, window);
		
		system("");
	}

	cout << "end:	" << i << completepath[minbatch*i] << endl;

	vector<string > completepath_minbatch(completepath.end() - minbatch, completepath.end());
	vector<string > filename_minbatch(filename.end() - minbatch, filename.end());

	main1(completepath_minbatch, filename_minbatch);//, window);

													//SDL_GL_DeleteContext(glContext);
													//SDL_DestroyWindow(window);
													//SDL_Quit();

	return 0;
}
#endif