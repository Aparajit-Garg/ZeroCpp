'''
MIT License

Copyright (c) 2021 Tauhid Khan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

'''RUNNING COMMAND: 
python3.6 single_image_enhance_tflite.py --tflite_path zero_dce_lite_160x160_iter8_30.tflite --img_path 1.png --plot 0 --save_result 1
'''

import os
import cv2
import numpy as np
import argparse
import tracemalloc
from time import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
# from tensorflow.keras.utils import save_img

from utils import plot_image_enhanced_image_amaps, read_image, post_enhance_iteration

def tflite_run_inference(tflite_path:str, img_path:str, iteration:int = 6, plot:bool = False, save_result:bool = True):
    '''
    Run inference on a single resized image.
    args:
        tflite_path: path to tflite model
        img_path: path to image file
        iteration: number of Post Ehnancing iterations
        plot: plot enhanced image. 0: no plot, 1: plot
        save_result: save enhanced image. 0: no save, 1: save
    return: None
    '''
    assert plot in [False, True] , 'plot must be either 0 or 1'
    assert save_result in [False, True] , 'save_results must be either 0 or 1'
    
    _results_dir = 'small_subset_results/'
    if not os.path.exists(_results_dir):
        os.mkdir(_results_dir)
    
    # Get image name from path
    image_name = img_path#(img_path.split('/')[-1]).split('.')[0]
    
    # Get model name from model path
    model_name = (tflite_path.split('/')[-1]).split('.')[0]
    
    # Load model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)#, num_threads=4)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("INPUT DETAILS: ", input_details)
    print("OUTPUT DETAILS: ", output_details)
    # Read image
    img_h = int(input_details[0]['shape'][1])
    img_w = int(input_details[0]['shape'][2])
    print("Required image dimension for inference: ", img_h, img_w)
    # images = os.listdir(img_path)
    count = 0
    total_time = 0
#    for image_path in images:
    # print(image_path)
    image_path = img_path
    i = cv2.imread(image_path)
    print(i[0][0])
    resize_image, original_image = read_image(image_path, img_h=img_h, img_w=img_w)
    
    # Run inference
    tf.print("[info] running inference....")
    interpreter.set_tensor(input_details[0]['index'], resize_image.numpy())
    start_time = time()
    tracemalloc.start()
    interpreter.invoke()
    print("Memory taken: ", tracemalloc.get_traced_memory())
    end_time = time()
    count += 1
    total_time = total_time + end_time - start_time
    print("Time taken per frame inference: ", end_time - start_time)
    # print("Line 100: ", interpreter.get_tensor(output_details))
    # print("Line 100: ", len(interpreter.get_tensor(output_details[1]['index'])[0]))
    a_maps = tf.cast(interpreter.get_tensor(output_details[1]['index']), tf.float32)
    print("Mapp values: ", type(a_maps[0]), a_maps[0][0][0])
    enhanced_img = tf.cast(interpreter.get_tensor(output_details[0]['index']), tf.float32)
    print(type(enhanced_img), enhanced_img[0][0][0][0], enhanced_img[0][0][0][1], enhanced_img[0][0][0][2])
    # print(type(enhanced_img))
    tf.print(f'Time taken to run inference: {(end_time - start_time)*1000} ms')
    # cv2.imshow("enhanced_img", enhanced_img)
    # cv2.waitKey(0)
    if plot:
        a_maps_shifted_mean = (a_maps + 1)/2
        plot_image_enhanced_image_amaps(resize_image, enhanced_img, a_maps_shifted_mean)
    
    if save_result:
        # print(type(enhanced_img), type(np.array(enhanced_img)))
        enhanced_original_image = post_enhance_iteration(original_image, a_maps, iteration)
        # save_img(
        #     os.path.join(_results_dir, f'00{count}_{model_name}_enhanced_tflite_{image_name}.jpg'),
        #         enhanced_original_image
        #     )
    print("Total time taken: ", total_time)
    print("Avg time taken: ", total_time/count)
    print("Number of images: ", count)

def main():
    parser = argparse.ArgumentParser(description='Run inference on a single image.')
    parser.add_argument('--tflite_path', type=str, help='path to tflite model')
    parser.add_argument('--img_path', type=str, help='path to image file')
    parser.add_argument('--iteration', type=int, default=6, help='number of Post Ehnancing iterations')
    parser.add_argument('--plot', type=int, default=0, help='plot enhanced image. 0: no plot, 1: plot')
    parser.add_argument('--save_result', type=int, default=0, help='save enhanced image. 0: no save, 1: save')
    args = parser.parse_args()
    s = time()
    tflite_run_inference(
        args.tflite_path,
        args.img_path,
        args.iteration,
        bool(args.plot),
        bool(args.save_result)
    )
    print("Complete pipeline time: ", time() - s)

if __name__ == "__main__":
    main()
