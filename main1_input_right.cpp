#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/examples/label_image/get_top_n.h"
#include "tensorflow/lite/model.h"

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

int main(int argc, char **argv)
{
    printf("+++++++++++++FIRST STATEMENT+++++++++++++\n");

    // Get Model label and input image
    if (argc != 4)
    {
        fprintf(stderr, "TfliteClassification.exe modelfile labels image\n");
        exit(-1);
    }

    printf("++++++++++++++SECOND STATEMENT++++++++++++++\n");
    const char *modelFileName = argv[1];
    const char *labelFile = argv[2];
    const char *imageFile = argv[3];
    std::chrono::time_point<std::chrono::system_clock> start, end, first, last;
    first = std::chrono::system_clock::now();
    // Load Model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(modelFileName);
    if (model == nullptr)
    {
        fprintf(stderr, "failed to load model\n");
        exit(-1);
    }
    
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);
    if (interpreter == nullptr)
    {
        fprintf(stderr, "Failed to initiate the interpreter\n");
        exit(-1);
    }
    printf("INTERPRETER LOADED SUCCESSFULLY\n");
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        fprintf(stderr, "Failed to allocate tensor\n");
        exit(-1);
    }
    
    // Get Input Tensor Dimensions
    int input = interpreter->inputs()[0];
    // std::cout << "inputs value: " << input << std::endl;
    auto height = interpreter->tensor(input)->dims->data[1];
    auto width = interpreter->tensor(input)->dims->data[2];
    auto channels = interpreter->tensor(input)->dims->data[3];

    printf("INPUT DIMENSIONS FROM MODEL LOADED \n");
    printf("%d %d %d\n", height, width, channels);

    // Load Input Image
    cv::Mat image;
    cv::Mat frame = cv::imread(imageFile, cv::IMREAD_COLOR);
    int h_orig = frame.rows;
    int w_orig = frame.cols;
    int c_orig = frame.channels();

    frame.convertTo(frame, CV_32F, 1.0/255);

    if (frame.empty())
    {
        fprintf(stderr, "Failed to load iamge\n");
        exit(-1);
    }
   
    cv::resize(frame, image, cv::Size(height, width), cv::INTER_LINEAR);
    // std::cout << "frame dimensions: " << frame.rows << " " << frame.cols << " " << frame.channels() << std::endl;
    // std::cout << "here: " << image.rows << " "  << image.cols << " " << image.channels() << std::endl;
    
    float *input_data_ptr = interpreter->typed_tensor<float>(input);
    for (int i = 0; i < 160; i++) {
        for (int j = 0; j < 160; j++) {
            for (int k = 0; k < 3; k++){
                *(input_data_ptr) = (float)image.at<cv::Vec3f>(i,j)[k];
                input_data_ptr++;
            }
        }
    }
    
    start = std::chrono::system_clock::now();
    interpreter->Invoke();
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Time taken for inference: " << elapsed_seconds.count() << std::endl;
    int output_idx = interpreter->outputs()[0];
    float *output = interpreter->typed_tensor<float>(output_idx);
    // std::cout << "OUTPUT: " << *output << " " << *(output + 1) << " " << *(output + 2) << std::endl;
    cv::Mat final_img(160, 160, CV_32FC3, cv::Scalar(0));
    
    for (int i = 0; i < 160; i++) {
        for (int j = 0; j < 160; j++) {
            for (int k = 0; k < 3; k++){
                final_img.at<cv::Vec3f>(i, j)[k] = *output;
                output+=1;
            }
        }
    }

    // std::cout << final_img.rows << " " << final_img.at<cv::Vec3f>(0, 0)[0];
    final_img.convertTo(final_img, CV_32F, 255.0, 0);
    // cv::imwrite("final_img.png", final_img);
    
    int output_mapp_idx = interpreter->outputs()[1];
    float *output_mapp = interpreter->typed_tensor<float>(output_mapp_idx);
    // std::cout << "Mapp values check: " << *output_mapp << " " << *(output_mapp + 1) << std::endl;
    
    cv::Mat mapp(160, 160, CV_32FC3, cv::Scalar(0.0));
    for (int i = 0; i < 160; i++) {
        for (int j = 0; j < 160; j++) {
            for (int k = 0; k < 3; k++){
                mapp.at<cv::Vec3f>(i, j)[k] = *output_mapp;
                output_mapp++;
            }
        }
    }

    // std::cout << "mapp mat 0th index: " << mapp.at<cv::Vec3f>(0, 0)[0] << std::endl;

    //post processing
    int iteration = 6;
    cv::Mat mapp_bigger(h_orig, w_orig, CV_32FC3, cv::Scalar(0));
    // std::cout << "Just declared: " << mapp_bigger.rows << mapp_bigger.cols << std::endl;
    cv::resize(mapp, mapp_bigger, cv::Size(w_orig, h_orig), cv::INTER_CUBIC);
    // std::cout << "frame dimensions: " << frame.rows << " " << frame.cols << " " << frame.channels() << std::endl;
    // original_image = original_image + (a_maps)*(tf.square(original_image) - original_image)
    // std::cout << type2str(mapp.type()) << " " << type2str(mapp_bigger.type()) << type2str(frame.type()) << std::endl;
    // std::cout << "Mapp dimensions: " << mapp_bigger.rows << " " << mapp_bigger.cols << mapp_bigger.channels() << std::endl;
    // std::cout << "frame dimensions: " << frame.rows << " " << frame.cols << " " << frame.channels() << std::endl;
    for (int iter = 0; iter < iteration; iter++) {
        for (int i = 0; i < frame.rows; i++) {
            for (int j = 0; j < frame.cols; j++) {
                for (int k = 0; k < 3; k++) {
                    // std::cout << "value printed above: " << frame.at<cv::Vec3f>(i, j)[k] << i, j, k << std::endl;
                    frame.at<cv::Vec3f>(i, j)[k] = frame.at<cv::Vec3f>(i, j)[k] + 
                                                    mapp_bigger.at<cv::Vec3f>(i, j)[k] * (pow(frame.at<cv::Vec3f>(i, j)[k], 2)
                                                    - frame.at<cv::Vec3f>(i, j)[k]);
                    // std::cout << "value printed below: " << frame.at<cv::Vec3f>(i, j)[k] << std::endl;
                }
            }
        }
    }
    // std::cout << "HELLO" << std::endl;

    frame.convertTo(frame, CV_8UC3, 255);
    for (int i = 0; i < frame.rows; i++) {
            for (int j = 0; j < frame.cols; j++) {
                for (int k = 0; k < 3; k++) {
                    int a = frame.at<cv::Vec3b>(i, j)[k];
                    if (a < 1){
                        frame.at<cv::Vec3b>(i, j)[k] = 0;
                    }
                    else if(a > 254){
                        frame.at<cv::Vec3b>(i, j)[k] = 255;
                    }
                }
            }
    }
    last = std::chrono::system_clock::now();
    elapsed_seconds = last - start;
    // cv::imwrite("final_img_orig.png", frame);
    std::cout << "Time taken by complete pipeline: " << elapsed_seconds.count() << std::endl;
    return 0;
}