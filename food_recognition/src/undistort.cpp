#include<opencv2/opencv.hpp>
#include<iostream>

int calib()
{
	cv::Mat image, img_gray;
	int BOARDSIZE[2]{6, 9};

	std::vector<std::vector<cv::Point3f>> objpoints_img;//保存棋盘格上角点的三维坐标
	std::vector<cv::Point3f> obj_world_pts;
	std::vector<std::vector<cv::Point2f>> images_points;//保存所有角点
	std::vector<cv::Point2f> img_corner_points;//保存每张图检测到的角点
	std::vector<cv::String> images_path;

	std::string image_path = "./distImgs/*.jpg";
	cv::glob(image_path, images_path);

	for (int i = 0; i < BOARDSIZE[1]; i++)
	{
		for (int j = 0; j < BOARDSIZE[0]; j++)
		{
			obj_world_pts.push_back(cv::Point3f(j, i, 0));
		}
	}

	for (int i = 0; i < images_path.size(); i++)
	{
		image = cv::imread(images_path[i]);
		cv::cvtColor(image, img_gray, cv::COLOR_BGR2GRAY);
		//检测角点
		bool found_success = cv::findChessboardCorners(img_gray, cv::Size(BOARDSIZE[0], BOARDSIZE[1]),
			img_corner_points,
			cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

		if (found_success)
		{
			//迭代终止条件
			cv::TermCriteria criteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.001);

			//进一步提取亚像素角点
			cv::cornerSubPix(img_gray, img_corner_points, cv::Size(11, 11),
				cv::Size(-1, -1), criteria);

			cv::drawChessboardCorners(image, cv::Size(BOARDSIZE[0], BOARDSIZE[1]), img_corner_points,
				found_success);

			objpoints_img.push_back(obj_world_pts);//从世界坐标系到相机坐标系
			images_points.push_back(img_corner_points);
		}
		//char *output = "image";
		char text[] = "image";
		char *output = text;
		cv::imshow(output, image);
		cv::waitKey();
	}

	/*
	计算内参和畸变系数等
	*/
	cv::Mat cameraMatrix, distCoeffs, R, T;//内参矩阵，畸变系数，旋转量，偏移量
	cv::calibrateCamera(objpoints_img, images_points, img_gray.size(),
		cameraMatrix, distCoeffs, R, T);

	std::cout << "cameraMatrix:" << std::endl;
	std::cout << cameraMatrix << std::endl;

	std::cout << "*****************************" << std::endl;
	std::cout << "distCoeffs:" << std::endl;
	std::cout << distCoeffs << std::endl;
	std::cout << "*****************************" << std::endl;

	std::cout << "Rotation vector:" << std::endl;
	std::cout << R << std::endl;

	std::cout << "*****************************" << std::endl;
	std::cout << "Translation vector:" << std::endl;
	std::cout << T << std::endl;

	///*
	//畸变图像校准
	//*/
	cv::Mat src, dst;
	src = cv::imread("./Works/C++/2.jpg");
	cv::undistort(src, dst, cameraMatrix, distCoeffs);

	char texts[] = "image_dst";
	char *dst_output = texts;
	//char *dst_output = "image_dst";
	cv::imshow(dst_output, dst);
	cv::waitKey();
	// cv::imwrite("./Work/image/3.jpg", dst);
	cv::destroyAllWindows();
	system("pause");
	return 0;
}


