#pragma once

#include <thread>
#include <mutex>
#include <tensorflow/lite/context.h>
//#include <tensorflow/contrib/lite/context.h>

#include <opencv2/opencv.hpp>
#include <unordered_map>

namespace tflite {
	//namespace impl{
	class Interpreter;
	//}
	class FlatBufferModel;
}

namespace hanshang_algo { namespace cnn_hanshang {
        //struct alignas(float) detection_res{
	struct detection_res{
        //cneterx centery w h
        float bbox[4];
        //float objectness;
        int class_id;
        float prob;

        std::vector<float> score_list;
    };
	struct Bbox
    {
        int class_id;
        int left;
        int right;
        int top;
        int bot;
        float score;

        std::vector<float> score_list;
    };
	struct detector_config {
		int input_width = 416;
		int input_height = 416;
        int input_depth = 3;
		std::string frozen_net_path;
        bool enable_multi_thread = false;
		bool enable_gpu_inference = true;
		float score_thresh = 0.6;
		float nms_thresh = 0.3;
		//int class_num = 4;
		// int class_num = 6;
		int class_num = 28;
		int anchor_num_per_scale = 3;
		std::vector<std::vector<float> > anchors_8 = {{1.25f, 1.625f}, {2.0f, 3.75f}, {4.125f, 2.875f}};
        std::vector<std::vector<float> > anchors_16 = {{1.875f, 3.8125f}, {3.875f, 2.8125f}, {3.6875f, 7.4375f}};
        std::vector<std::vector<float> > anchors_32 = {{3.625f, 2.8125f}, {4.875f, 6.1875f}, {11.65625f, 10.1875f}};
                /*
		std::string class_name[4] = {"zhibeidangao",
			"qifeng",
			"tusi",
			"quqi"};
                std::string class_name[6] = {"zhibeidangao",
                        "qifeng",
                        "tusi",
                        "quqi",
                        "zaocanbao",
			"dangaojuan"};
		*/

		/*std::string class_name[16] = {"zhibeidangao",
				        "qifeng",
				        "tusi",
				        "quqi",
				        "zaocanbao",
								"dangaojuan",
					"danta",
					"jichi",
					"jichigen",
					"jixiongrou",
					"jimihua",
					"manyuemeibinggan",
					"peigen",
					"niupai",
					"shutiao",
					"oubao"};*/

		std::string class_name[28] = {"danta",
					     "digua",
					    "huage",
					    "jichi",
					    "jinzhengu",
					    "jiucai",
					    "luyu",
					    "manyuemeibinggan",
					    "mianbaopian",
					    "niupai",
					    "paigu",
					    "peigen",
					    "pisa",
					    "qiezi",
					    "qifeng",
					    "quqi",
					    "rouchuan",
					    "zaocanbao",
					   "zhibeidangao",
					    "jichigen",
					     "jimihua",
					    "kaochang",
					    "magelitebiaobinggan",
					    "paofu",
					    "pipatui",
					    "shutiao",
					    "tusi",
					    "xia"};
        };

	class detection {
	public:
		detection(const detector_config &cfg);
		~detection();
        std::vector<Bbox> operator() (const cv::Mat &src_img,
		  int64_t timestamp);

	private:
		std::unique_ptr<tflite::Interpreter> interpreter_;
		std::unique_ptr<tflite::FlatBufferModel> model_;
		TfLiteDelegate* delegate_;
		detector_config cfg_;
		mutable std::mutex operator_mutex_;

		std::vector<Bbox> det_res_parse(TfLiteTensor* p_tflts_output0, TfLiteTensor* p_tflts_output1,
		 TfLiteTensor* p_tflts_output2, cv::Size img_size) const;

		std::vector<Bbox> run_quantization(const cv::Mat &src_img, int64_t timestamp);
		
		std::vector<detection_res> change_det_res_order(TfLiteTensor* p_tflts_output, int class_num, int stride,
		 std::vector<std::vector<float> > anchors) const;
		cv::Mat input_preprocess(cv::Mat img);
		void nms(std::vector<detection_res>& detections, int classes, float nms_thresh) const;
		std::vector<Bbox> output_postprocess(cv::Size img_size, std::vector<detection_res>& detections, int classes) const;
        float sigmoid_fun(float x) const;
	};
} }
