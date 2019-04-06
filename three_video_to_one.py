import numpy as np
import cv2

human_video_cap = cv2.VideoCapture("human_video.mp4")
cat_video_cap =cv2.VideoCapture("cat_video2.mp4")
human_mouth_cat_cap = cv2.VideoCapture("cat_with_humanmouth.avi")

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video_writer = cv2.VideoWriter('three_video_to_one_cat_with_humanmouth.avi', fourcc, 29.43, (720,1280))

index=0
while(True):
    print(index)
    index+=1
    if(index>400):
        break
    human_ret,human_frame = human_video_cap.read()
    human_frame = cv2.resize(human_frame,dsize=(240,426))
    cat_ret,cat_frame = cat_video_cap.read()
    cat_frame = cv2.resize(cat_frame,dsize=(240,426))
    human_mouth_cat_ret,human_mouth_cat_frame = human_mouth_cat_cap.read()
    human_mouth_cat_frame[:426,:240,:]=human_frame
    human_mouth_cat_frame[:426,480:,:]=cat_frame
    video_writer.write(human_mouth_cat_frame)
video_writer.release()