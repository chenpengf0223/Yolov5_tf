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

template<typename T> std::string to_string_custom(const T& t){
    std::ostringstream oss;
    oss << t;
    return oss.str();  
}

cv::Mat draw_det_res(const cv::Mat image, std::vector<Bbox> det_res, std::string class_name[], int class_num)
{
    cv::Mat show_img = image.clone();
    int bbox_thick = 3;

    std::vector<float> score_list(class_num);
    std::cout<<"score l "<< score_list.size()<<std::endl;
    for (int i = 0; i < score_list.size(); ++i)
        std::cout<<"score "<<score_list[i] <<std::endl;
    getchar();

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
        cv::putText(show_img, to_string_custom(max_score_idx),
         cv::Point(100, 100), cv::FONT_HERSHEY_SIMPLEX,
         fontScale, _color_map[max_score_idx], bbox_thick);
    return show_img;
}

int main()
{
    std::unique_ptr<sidewalk_perception> uq_sidewalk_perception = nullptr;
	segmentor_config sidewalk_config;
    sidewalk_config.enable_multi_thread = false;
    sidewalk_config.input_width = 416;
    sidewalk_config.input_height = 416;
    sidewalk_config.input_depth = 3;
    sidewalk_config.frozen_net_path = "/data/seg_RK/1.8997-409-detecor.tflite";//yolov4-416-fp32-tf1.tflite
    sidewalk_config.class_num = 4;
    sidewalk_config.enable_gpu_inference = false;
    uq_sidewalk_perception = std::make_unique<sidewalk_perception>(sidewalk_config);
    getchar();

//#define TEST_TIME__
#ifdef TEST_TIME__
    cv::Mat frame;
    frame = cv::imread("/data/seg_RK/1572585668141557.jpg");
    if(frame.empty())
    {
        std::cout<<"frame.empty(). "<<std::endl;
        return 1;
    }
    float time_cal = 0;
    int test_iteration = 150;
    int ignore_iteration = 10;
    for(int k = 0; k < test_iteration; k ++)
    {
        int64_t input_timestamp = k;

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<Bbox> det_res = (*uq_sidewalk_perception)(frame, input_timestamp);
        auto end = std::chrono::high_resolution_clock::now();
        if (k >= ignore_iteration)
        {
            std::chrono::duration<double, std::milli> elapsed = end - start;
            time_cal += elapsed.count();
        }
    }
    std::cout << " -------inference time ------ " << time_cal / (test_iteration - ignore_iteration) << " ms " <<std::endl;
    return 0;
#else
    cv::Mat frame;
    //char img_path[100];
    std::string img_folder = "/data/seg_RK/qixing-test";//test-z /data/seg_RK/qixing-test

    std::vector< std::string> img_list;
    std::string end_str = ".jpg";
    searchImgsIterativelyInADir(img_folder, img_list, end_str);

    std::string res_folder("/data/seg_RK/res");
    int64_t start_idx = 0; //1572585668141557; //1567572258539; //500
    //int64_t end_idx = 1575548529734023;
    std::vector<cv::Point> birdview_to_fishview_map;
    int ignore_iteration = 10;
    int test_time_iteration = 110;
    std::vector<double> time_cal;

    for(int64_t img_idx = start_idx; img_idx < img_list.size(); img_idx += 1)
    {
        //sprintf(img_path, "/%ld.jpg", img_idx); //%.5ld.png
        //sprintf(img_path, "/%ldfinsheye.png", img_idx);
        std::string cur_img_path = img_list[img_idx];
        std::cout<<"img "<< cur_img_path <<std::endl;
        frame = cv::imread(cur_img_path);
        if(frame.empty())
        {
            std::cout<<"frame.empty(). "<<std::endl;
            continue;
            break;
        }
        int64_t input_timestamp = img_idx;
        std::vector<Bbox> det_boxes;
        auto start = std::chrono::high_resolution_clock::now();
        det_boxes = (*uq_sidewalk_perception)(frame, input_timestamp);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;

        std::cout << " operater fun time  " << elapsed.count() << " ms... " <<std::endl;

        if (img_idx - start_idx >= ignore_iteration && img_idx - start_idx < test_time_iteration)
        {
            time_cal.push_back(elapsed.count());
        }
        else if (img_idx - start_idx == test_time_iteration)
        {
            double sum = std::accumulate(time_cal.begin(), time_cal.end(), 0);
            std::cout << " -------operater average time ------ " << sum / time_cal.size() << " ms... " <<std::endl;

            getchar();
        }

        cv::Mat output_res = draw_det_res(frame, det_boxes, sidewalk_config.class_name, sidewalk_config.class_num);
        std::size_t idx_tt = img_list[img_idx].find_last_of("/");
        std::cout<< "writedone: " << img_list[img_idx] << std::endl;
        std::size_t len = img_list[img_idx].size();
        cv::imwrite(res_folder + "/" + img_list[img_idx].substr(idx_tt+1, len) + "-res.jpg", output_res);
        std::cout<< "writedone: " << res_folder + "/" + img_list[img_idx].substr(idx_tt+1, len) + "-res.jpg" << std::endl;
    }
    return 0;
#endif
}