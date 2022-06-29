"""
@file   : step2_ocr_recognize.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-04-01
"""
import os
import time
import cv2
import paddlehub as hub
# pip install paddlepaddle==2.2.2
# pip install paddlehub==2.0.0
# pip install shapely=='1.8.1.post1'
# pip install pyclipper=='1.3.0.post2'
 


if __name__ == '__main__':
    # 服务端可以加载大模型，效果更好
    ocr = hub.Module(name="chinese_ocr_db_crnn_server")

    # 加载视频的所有帧
    files = os.listdir('./frame')
    files.sort()
    test_img_path = ['./frame/' + i for i in files]
    print(test_img_path)

    # 读取测试文件夹test.txt中的照片路径
    np_images = [cv2.imread(image_path) for image_path in test_img_path]

    s = time.time()
    results = ocr.recognize_text(
        images=np_images,  # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
        use_gpu=False,  # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
        output_dir='ocr_result',  # 图片的保存路径，默认设为 ocr_result；
        visualization=False,  # 是否将识别结果保存为图片文件；
        box_thresh=0.5,  # 检测文本框置信度的阈值；
        text_thresh=0.5)  # 识别中文文本置信度的阈值；
    e = time.time()
    final_text = []
    for res in results:
        try:
            final_text.append(res['data'][-1]['text'])
        except:
            continue
    print(final_text)
    final_text = list({}.fromkeys(final_text).keys())
    final_text = '，'.join(final_text)
    print(final_text)
    print('总共花费时间:', e - s)




