#include <detection/detector.h>
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace hanshang_algo::cnn_hanshang;
#include <unordered_map>

#include <fstream>
#include <numeric>
#include "search_file.hpp"
/*
   Color     Red      Green      Blue      值     
   黑色   Black    0   0    0    0     
   白色   White    255    255    255    16777215     
   灰色   Gray    192    192    192    12632256     
   深灰色    Dark    Grey    128    128    128    8421504     
   红色    Red    255    0    0    255     
   深红色    Dark    Red    128    0    0    128     
   绿色    Green    0    255    0    65280     
   深绿色    Dark    Green    0    128    0    32768     
   蓝色    Blue    0    0    255    16711680     
   深蓝色    Dark    Blue    0    0    128    8388608     
   紫红色    Magenta    255    0    255    16711935     
   深紫红    Dark    Magenta    128    0    128    8388736     
   紫色    Cyan    0    255    255    16776960     
   深紫    Dark    Cyan    0    128    128    8421376     
   黄色    Yellow    255    255    0    65535     
   棕色    Brown    128    128    0    32896     
*/   

std::vector<cv::Scalar> _color_map { cv::Scalar(0, 0, 0),
 cv::Scalar(0, 255, 255),
 cv::Scalar(192, 192, 192),
 cv::Scalar(255, 255, 0),
 cv::Scalar(255, 0, 0),
 cv::Scalar(0, 255, 0),
 cv::Scalar(0, 0, 255),
 cv::Scalar(255, 0, 255),
 cv::Scalar(0, 128, 128)};

cv::Mat draw_det_res(const cv::Mat image, std::vector<Bbox> det_res, std::string class_name[], int class_num)
{
    cv::Mat show_img = image.clone();
    int bbox_thick = 3;

    std::vector<float> score_list(class_num);
    for (int i = 0; i < det_res.size(); ++i)
    {
        Bbox box = det_res[i];
        if (box.class_id >= _color_map.size())
        {
            std::cout<<"box.class_id >= _color_map.size(): "<<box.class_id <<" "<<
             _color_map.size()<<std::endl;
            getchar();
        }
        score_list[box.class_id] += box.score;
    }
    float max_score = -1;
    int max_score_idx = -1;
    for (int i = 0; i < score_list.size(); ++i)
    {
        if (score_list[i] > 0)
        {
            if (max_score < score_list[i])
            {
                max_score = score_list[i];
                max_score_idx = i;
            }
        }
    }
    int fontScale = 1;
    if (max_score_idx >= 0)
        cv::putText(show_img, class_name[max_score_idx],
         cv::Point(100, 100), cv::FONT_HERSHEY_SIMPLEX,
         fontScale, _color_map[3], bbox_thick);
    return show_img;
}

void inference_step()
{
    std::unique_ptr<detection> uq_detection = nullptr;
    detector_config detector_conf;
    detector_conf.enable_multi_thread = false;
    detector_conf.input_width = 416;
    detector_conf.input_height = 416;
    detector_conf.input_depth = 3;
    detector_conf.frozen_net_path = "/data/seg_RK/1.8997-409-detecor-cplus.tflite";
    detector_conf.class_num = 4;
    detector_conf.enable_gpu_inference = true;
    uq_detection = std::make_unique<detection>(detector_conf);

    cv::Mat frame = get_camera_image();//自定义函数:获取相机图像
    int64_t input_timestamp = get_timestamps();//自定义函数：获取图像的时间戳
    std::vector<Bbox> det_boxes;
    det_boxes = (*uq_detection)(frame, input_timestamp);

    cv::Mat output_res = draw_det_res(frame, det_boxes, detector_conf.class_name, detector_conf.class_num);

    draw_result_on_screen(output_res);//自定义函数：显示预测结果到屏幕
}