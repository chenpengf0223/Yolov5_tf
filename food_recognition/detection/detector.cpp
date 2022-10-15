#include <android/log.h>

#include "detector.h"

#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

//#include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

//#include "ninebot_log.h"
#include <iostream>
// using std::vector;
using namespace hanshang_algo;

#define _USE_TF_FLOAT_MODEL

// #define CALCULATE_COST_TIME_

#define SP_LOG_TAG "detector_so_log"
#define SP_LOG(...) __android_log_print(ANDROID_LOG_DEBUG, SP_LOG_TAG, __VA_ARGS__)

namespace hanshang_algo{ namespace cnn_hanshang {
	detection::detection(const detector_config &cfg)
	{
		cfg_ = cfg;
		model_ = tflite::FlatBufferModel::BuildFromFile(cfg_.frozen_net_path.c_str());
        if (!model_)
            return;

		tflite::ops::builtin::BuiltinOpResolver resolver;
		//tflite::InterpreterBuilder builder(*model_.get(), resolver);
        //builder(&interpreter_);

        tflite::InterpreterBuilder(*model_, resolver)(&interpreter_);
		
        if(cfg_.enable_multi_thread)
            interpreter_->SetNumThreads(2);
		interpreter_->AllocateTensors();

        // Prepare GPU delegate.
        if(cfg_.enable_gpu_inference){
            // const TfLiteGpuDelegateOptions options = {
            //       .metadata = NULL,
            //       .compile_options = {
            //       //.precision_loss_allowed = 1,  // FP16
            //       .preferred_gl_object_type = TFLITE_GL_OBJECT_TYPE_FASTEST,
            //       .dynamic_batch_enabled = 0,   // Not fully functional yet
            //      },
            //     };

            //const TfLiteGpuDelegateOptions options = TfLiteGpuDelegateOptionsDefault();
            //const TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();

            const TfLiteGpuDelegateOptionsV2 options = {
                // .max_delegated_partitions = 2,
                .is_precision_loss_allowed = 1,
                .inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER,
                //.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED,
    #ifdef _USE_TF_FLOAT_MODEL
    #else
                .experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT,
    #endif
            };
            delegate_ = TfLiteGpuDelegateV2Create(&options);
            //delegate_ = TfLiteGpuDelegateCreate(&options);
            
            if (interpreter_->ModifyGraphWithDelegate(delegate_) != kTfLiteOk) {
              std::cout << "ModifyGraphWithDelegate return false....." << std::endl;
              return;
            }
        }

        SP_LOG("Detection library initialization is over.");
	}

    detection::~detection(){
        if(cfg_.enable_gpu_inference){
            //Clean up.
            TfLiteGpuDelegateV2Delete(delegate_);
            //TfLiteGpuDelegateDelete(delegate_);

            SP_LOG("Detection library TfLiteGpuDelegateV2Delete is over.");
        }
        SP_LOG("Detection library destruction is over.");
    }

	std::vector<Bbox> detection::operator() (const cv::Mat &src_img,
     int64_t timestamp){
		std::lock_guard<std::mutex> guard(operator_mutex_);
        return run_quantization(src_img, timestamp);
	}
    
    float detection::sigmoid_fun(float x) const
    {
	    return 1. / (1. + exp(-x));
    }

    std::vector<detection_res> detection::change_det_res_order(TfLiteTensor* p_tflts_output,
     int class_num, int stride, std::vector<std::vector<float> > anchors) const
    {
        auto dims = p_tflts_output->dims;
        // auto scale = p_tflts_output->params.scale;
        // auto zero_point = p_tflts_output->params.zero_point;
        // std::cout<<"dims->data "<<" "<<dims->data[3]<<" "<<dims->data[2]<<" "<<dims->data[1]<<" "<<dims->data[0]<<std::endl;
        std::vector<detection_res> result;
        #ifdef _USE_TF_FLOAT_MODEL
            //float* p_output_data = interpreter_->typed_output_tensor<float>(0);
            float* p_output_data = (float*)p_tflts_output->data.uint8;
            // memcpy(result.data(), p_output_data, det_count* sizeof(detection_res));
            int c_num = dims->data[3];
            int anchor_decode_num = c_num / cfg_.anchor_num_per_scale;
            for (int x = 0; x < dims->data[2]; ++x)
            {
                for(int y = 0; y < dims->data[1]; ++y){
                    // int idxbase = i * last_dim;
                    // if (i==0){
                    //     for(int df=0; df<9; ++df)
                    //         std::cout<<"dims->data "<< p_output_data[df + idxbase]<<std::endl;
                    //         getchar();
                    // }
                    int xy_base = dims->data[2] * y + x;
                    for (int c = 0; c < cfg_.anchor_num_per_scale; c++)
                    {
                        detection_res tmp_res;

                        int idxbase = c_num * xy_base + c * anchor_decode_num;
                        tmp_res.bbox[0] = (sigmoid_fun(p_output_data[idxbase]) + x) * stride;
                        tmp_res.bbox[1] = (sigmoid_fun(p_output_data[idxbase+1]) + y) * stride;
                        tmp_res.bbox[2] = exp(p_output_data[idxbase+2]) * stride * anchors[c][0];
                        tmp_res.bbox[3] = exp(p_output_data[idxbase+3]) * stride * anchors[c][1];

                        int xmin = tmp_res.bbox[0] - 0.5*tmp_res.bbox[2];
                        int xmax = tmp_res.bbox[0] + 0.5*tmp_res.bbox[2];
                        int ymin = tmp_res.bbox[1] - 0.5*tmp_res.bbox[3];
                        int ymax = tmp_res.bbox[1] + 0.5*tmp_res.bbox[3];
                        if (xmin >= xmax || ymin >= ymax){
                            continue;
                        }
                        float score = sigmoid_fun(p_output_data[idxbase+5]);
                 
                        tmp_res.score_list.push_back(score);
                        int class_idx = 0;
                        for (int j = 6; j < class_num + 5; j++)
                        {
                            if (score < sigmoid_fun(p_output_data[idxbase+j]))
                            {
                                class_idx = j - 5;
                                score = sigmoid_fun(p_output_data[idxbase+j]);
                            }
                            tmp_res.score_list.push_back(sigmoid_fun(p_output_data[idxbase+j]));
                        }
                        if (score < cfg_.score_thresh)
                        {
                            continue;
                        }
                        tmp_res.class_id = class_idx;
                        tmp_res.prob = sigmoid_fun(p_output_data[idxbase+4]) * sigmoid_fun(p_output_data[class_idx+5+idxbase]);
                        if (tmp_res.prob < cfg_.score_thresh)
                        {
                            continue;
                        }
                        // if (tmp_res.class_id>0)
                        // {
                        //     std::cout<<"tmp_res.class_id "<< tmp_res.class_id<<std::endl;
                        //     getchar();
                        // }
                        result.push_back(tmp_res);
                    }
                }
            }
        #else
            auto p_output_data = p_tflts_output->data.uint8;
            for (int i = 0; i < data_size; ++i)
            {
                int j = i * 2;
                if(scale > 0){
                    if(p_output_data[j] < p_output_data[j + 1])
                    {
                        data_ptr[i] = 255;
                    }
                }
                else if(scale < 0)
                {
                    if(p_output_data[j] > p_output_data[j + 1])
                    {
                        data_ptr[i] = 255;
                    }
                }
                else
                {
                    SP_LOG("Detection result parsing, scale is 0, quantization is wrong.");
                }
            }
        #endif
        return result;
    }

    std::vector<Bbox> detection::det_res_parse(TfLiteTensor* p_tflts_output0, TfLiteTensor* p_tflts_output1,
    TfLiteTensor* p_tflts_output2, cv::Size img_size) const
    {
        std::vector<detection_res> result0 = change_det_res_order(p_tflts_output0, cfg_.class_num, 8, cfg_.anchors_8);
        std::vector<detection_res> result1 = change_det_res_order(p_tflts_output1, cfg_.class_num, 16, cfg_.anchors_16);
        std::vector<detection_res> result2 = change_det_res_order(p_tflts_output2, cfg_.class_num, 32, cfg_.anchors_32);
        // std::cout<<"result0 "<<result0.size()<<" "<<result1.size()<<" "<<result2.size()<<std::endl;
        result0.insert(result0.end(), result1.begin(), result1.end());
        result0.insert(result0.end(), result2.begin(), result2.end());
        // std::cout<<"result0 aft "<<result0.size()<<std::endl;


        std::vector<Bbox> bboxes = output_postprocess(img_size, result0, cfg_.class_num);
        //std::vector<Bbox> bboxes = output_postprocess(img_size, result1, cfg_.class_num);
        return bboxes;
    }

    std::vector<Bbox> detection::run_quantization(const cv::Mat &src_img,
     int64_t timestamp)
    {
        std::vector<Bbox> det_boxes;
        cv::Size img_size = cv::Size(src_img.cols, src_img.rows);

#ifdef CALCULATE_COST_TIME_
        auto start = std::chrono::high_resolution_clock::now();
#endif

        cv::Mat resized_img = input_preprocess(src_img);
        SP_LOG("Detection library input_preprocess is over.");

        // cv::Mat resized_img(416, 416, CV_32FC3, cv::Scalar(0.2,0.6,0.5));
        // for (int j = 0; j < resized_img.rows*resized_img.cols; ++j)
        // {
        //     std::cout<<"te "<<resized_img.at<cv::Vec3f>(j/416, j%416)[0] <<" "<<
        //     resized_img.at<cv::Vec3f>(j/416, j%416)[1]<<" "<<resized_img.at<cv::Vec3f>(j/416, j%416)[2]<< std::endl;
        //     // getchar();
        // }

        // std::cout<<"te "<<resized_img.at<cv::Vec3f>(0, 0)[0] <<" "<<
        //     resized_img.at<cv::Vec3f>(0, 0)[1]<<" "<<
        //     resized_img.at<cv::Vec3f>(0, 0)[2]<< std::endl;
        // std::cout<<"te "<<resized_img.at<cv::Vec3f>(200, 200)[0] <<" "<<
        //     resized_img.at<cv::Vec3f>(200, 200)[1]<<" "<<
        //     resized_img.at<cv::Vec3f>(200, 200)[2]<< std::endl;
            // getchar();
#ifdef _USE_TF_FLOAT_MODEL
        auto input = interpreter_->typed_input_tensor<float>(0);
        memcpy(input, resized_img.data, resized_img.rows*resized_img.cols*3*sizeof(float));
        SP_LOG("Detection library memcpy is over.");
#else
        auto input_node_index = interpreter_->inputs()[0];
        auto input = interpreter_->typed_tensor<std::uint8_t>(input_node_index);//float
        //for (int j = 0; j < resized_img.rows*resized_img.cols*3; ++j)
        //    input[j] = resized_img.ptr<std::uint8_t>()[j];//float
        memcpy(input, resized_img.data, resized_img.rows*resized_img.cols*3);
        //input = resized_img.data;//resized_img.ptr<std::uint8_t>();
        // auto input = interpreter_->typed_input_tensor<std::uint8_t>(0);
        // memcpy(input, resized_img.data, resized_img.rows*resized_img.cols*3);
#endif
        // std::cout << " data input...  " << std::endl;
        
#ifdef CALCULATE_COST_TIME_
        auto end0 = std::chrono::high_resolution_clock::now();
#endif

        // std::cout << " start invoking...  " << std::endl;
        // int ty = 0;
        // while(ty++ < 100)
        // {
        if(interpreter_->Invoke() != kTfLiteOk){
            std::cout << " interpreter_->Invoke()!= kTfLiteOk  " << std::endl;
            return det_boxes;
        }
        // }
        SP_LOG("Detection library Invoke is over.");

#ifdef CALCULATE_COST_TIME_
        auto end1 = std::chrono::high_resolution_clock::now();
#endif

        auto det_node_index0 = interpreter_->outputs()[0];
        TfLiteTensor* p_tflts_output0 = interpreter_->tensor(det_node_index0);
        auto det_node_index1 = interpreter_->outputs()[1];
        TfLiteTensor* p_tflts_output1 = interpreter_->tensor(det_node_index1);
        auto det_node_index2 = interpreter_->outputs()[2];
        TfLiteTensor* p_tflts_output2 = interpreter_->tensor(det_node_index2);
        det_boxes = det_res_parse(p_tflts_output0, p_tflts_output1, p_tflts_output2, img_size);
        SP_LOG("Detection library det_res_parse is over.");

#ifdef CALCULATE_COST_TIME_
        auto end2 = std::chrono::high_resolution_clock::now();
#endif

#ifdef CALCULATE_COST_TIME_
        std::chrono::duration<double, std::milli> elapsed0 = end0 - start;
        std::chrono::duration<double, std::milli> elapsed1 = end1 - end0;
        std::chrono::duration<double, std::milli> elapsed2 = end2 - end1;
        std::cout<<"preprocess  "<< elapsed0.count() << std::endl;
        std::cout<<"invoke 7-6  "<< elapsed1.count() << std::endl;
        std::cout<<"det_res_parse "<< elapsed2.count() << std::endl;
#endif
        return det_boxes;
    }

    cv::Mat detection::input_preprocess(cv::Mat img)
    {
        if (img.type() != CV_8UC3){
            std::cout << "img.type() != CV_8UC3, convert..." << std::endl;
            img.convertTo(img, CV_8UC3);
            getchar();
        }
        int c = cfg_.input_depth;
        int h = cfg_.input_height;
        int w = cfg_.input_width;

        float scale = cv::min(float(w)/img.cols, float(h)/img.rows);
        auto scale_size = cv::Size(img.cols * scale, img.rows * scale);
        // std::cout << "scale_size "<< scale_size << std::endl;
        cv::Mat rgb_img;
        cv::cvtColor(img, rgb_img, CV_BGR2RGB);
        cv::Mat rgb_img_f;
        rgb_img.convertTo(rgb_img_f, CV_32FC3);

        cv::Mat resized_img;
        cv::resize(rgb_img_f, resized_img, scale_size); //, 0, 0, CV_INTER_LINEAR);

        cv::Mat padding_img(h, w, CV_32FC3, cv::Scalar(128.0f, 128.0f, 128.0f));
        cv::Rect rect((w-scale_size.width)/2, (h-scale_size.height)/2, scale_size.width, scale_size.height);
        resized_img.copyTo(padding_img(rect));

        return padding_img/255.0;
    }

    void detection::nms(std::vector<detection_res>& detections, int classes, float nms_thresh) const
    {
        auto t_start = std::chrono::high_resolution_clock::now();

        std::vector<std::vector<detection_res>> class_res;
        class_res.resize(classes);

        for (const auto& item : detections)
            class_res[item.class_id].push_back(item);

        auto compute_iou = [](float * lbox, float* rbox)
        {
            float inter_box[] = {
                    cv::max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
                    cv::min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
                    cv::max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
                    cv::min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
            };

            if(inter_box[2] > inter_box[3] || inter_box[0] > inter_box[1])
                return 0.0f;

            float inter_boxs = (inter_box[1]-inter_box[0]) * (inter_box[3]-inter_box[2]);
            return inter_boxs / (lbox[2]*lbox[3] + rbox[2]*rbox[3] - inter_boxs);
        };

        std::vector<detection_res> result;
        for (int i = 0; i < classes; ++i)
        {
            auto& dets = class_res[i];
            if(dets.size() == 0)
                continue;

            sort(dets.begin(),dets.end(),[=](const detection_res& left, const detection_res& right){
                return left.prob > right.prob;
            });

            for (unsigned int m = 0; m < dets.size() ; ++m)
            {
                auto& item = dets[m];
                result.push_back(item);
                for(unsigned int n = m + 1; n < dets.size() ; ++n)
                {
                    if (compute_iou(item.bbox, dets[n].bbox) > nms_thresh)
                    {
                        dets.erase(dets.begin()+n);
                        --n;
                    }
                }
            }
        }
        // if (detections.size()>0)
        //     std::cout<<"item "<<detections[0].bbox[0]<< std::endl;
        //swap(detections,result);
        detections = move(result);
        // if (detections.size()>0)
        //     std::cout<<"aft move item "<<detections[0].bbox[0]<< std::endl;
        auto t_end = std::chrono::high_resolution_clock::now();
        float total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
        // std::cout << "Time taken for nms is " << total << " ms." << std::endl;
    }

    std::vector<Bbox> detection::output_postprocess(cv::Size img_size, std::vector<detection_res>& detections,
     int classes) const
    {
        int h = cfg_.input_height;
        int w = cfg_.input_width;

        //scale bbox to img
        int width = img_size.width;
        int height = img_size.height;
        float scale = cv::min(float(w)/width, float(h)/height);
        float scale_size[] = {width * scale, height * scale};

        //correct box
        for (auto& item : detections)
        {
            auto& bbox = item.bbox;
            bbox[0] = (bbox[0] - (w - scale_size[0])/2.f) / scale_size[0];
            bbox[1] = (bbox[1] - (h - scale_size[1])/2.f) / scale_size[1];
            bbox[2] /= scale_size[0];
            bbox[3] /= scale_size[1];
        }

        //nms
        float nms_thresh = cfg_.nms_thresh;
        if(nms_thresh > 0)
            nms(detections, classes, nms_thresh);

        std::vector<Bbox> boxes;
        for(const auto& item : detections)
        {
            auto& b = item.bbox;
            Bbox bbox =
                {
                    item.class_id,   //class_id
                    cv::max(int((b[0]-b[2]/2.)*width),0), //left
                    cv::min(int((b[0]+b[2]/2.)*width),width-1), //right
                    cv::max(int((b[1]-b[3]/2.)*height),0), //top
                    cv::min(int((b[1]+b[3]/2.)*height),height-1), //bot
                    item.prob       //score
                };

            for(int tp = 0; tp < item.score_list.size(); ++tp)
            {
                bbox.score_list.push_back(item.score_list[tp]);
            }

            boxes.push_back(bbox);
        }

        return boxes;
    }
}}
