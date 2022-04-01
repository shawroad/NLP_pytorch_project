"""
@file   : step1_extract_frame.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-04-01
"""
import cv2

if __name__ == '__main__':
    video_path = './data/video.mp4'

    # 1. 加载视频
    videoCapture = cv2.VideoCapture(video_path)

    # 2. 读取每一帧
    success, frame = videoCapture.read()

    # 3. 视频的总帧数
    total_frame = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
    print(total_frame)

    # 4. 计算一下当前视频的帧率
    fps = int(videoCapture.get(cv2.CAP_PROP_FPS))  # 每秒这么多帧
    print(fps)

    i, j = 0, 0
    save_path = 'frame'
    while success:
        if i % fps == 0:  # 保存   这个相当于每秒取一帧
            j = j + 1
            h, w, c = frame.shape
            frame = frame[(h//4)*3:h, :, :]   # 保留视频下面四分之一的图片
            image_save_path = '{}/image_{}.jpg'.format(save_path, '0' * (3-len(str(j))) + str(j))
            cv2.imwrite(image_save_path, frame)
            print('save images:', i)
        success, frame = videoCapture.read()
        i = i + 1