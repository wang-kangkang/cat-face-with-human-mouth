import cv2
import numpy as np
import dlib
#人脸检测
detector = dlib.get_frontal_face_detector()
#人脸landmark
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
#猫脸检测
cat_path = ".\haarcascade_frontalcatface.xml"
facecascade = cv2.CascadeClassifier(cat_path)
ret = facecascade.load(cat_path)
#猫脸landmark，这个没有搜到源码，暂时不做
old_cat_face_loc = np.array([-1,-1])
def human_mouth_paste_to_cat(human,cat):
    global old_cat_face_loc
    human_gray = cv2.cvtColor(human,cv2.COLOR_BGR2GRAY)
    points_keys = []
    #人脸检测
    rects = detector(human_gray,1)
    #人脸landmark检测
    landmarks = np.matrix([[p.x,p.y] for p in predictor(human_gray,rects[0]).parts()])
    landmarks = np.array(landmarks)
    #mouth的landmark
    mouth_landmark = landmarks[48:,:]
    #扩个边
    border=8
    mouth = human[np.min(mouth_landmark[:,1])-border:np.max(mouth_landmark[:,1]+border),np.min(mouth_landmark[:,0])-border:np.max(mouth_landmark[:,0])+border,:]
    mouth_landmark[:,0] -= (np.min(mouth_landmark[:,0])-border)
    mouth_landmark[:,1] -= (np.min(mouth_landmark[:,1])-border)

    #制作用于泊松融合的mask
    mask=np.zeros((mouth.shape[0],mouth.shape[1],3)).astype(np.float32)
    for i in range(mouth_landmark.shape[0]):#先画线
        cv2.line(mask,(mouth_landmark[i,0],mouth_landmark[i,1]),(mouth_landmark[(i+1)%mouth_landmark.shape[0],0],mouth_landmark[(i+1)%mouth_landmark.shape[0],1]),(255,255,255),10)
    mask_tmp=mask.copy()
    for i in range(6,mask.shape[0]-6):#将线内部的范围都算作mask=255
        for j in range(6,mask.shape[1]-6):
            if(np.max(mask_tmp[:i,:j,:])==0 or np.max(mask_tmp[i:,:j,:])==0 or np.max(mask_tmp[:i,j:,:])==0 or np.max(mask_tmp[i:,j:,:])==0):
                mask[i,j,:]=0
            else:
                mask[i,j,:]=255
    #猫脸检测
    width, height, channels = cat.shape
    cat_gray = cv2.cvtColor(cat,cv2.COLOR_BGR2GRAY)
    cat_face_loc= facecascade.detectMultiScale(cat_gray,scaleFactor = 1.1,minNeighbors=3,minSize=(100,100),flags=cv2.CASCADE_SCALE_IMAGE)
    cat_face_loc = cat_face_loc[0]
    if(old_cat_face_loc[0] != -1):#因为猫脸检测抖动太厉害，所以此处用历史坐标缓冲一下
        cat_face_loc = 0.9*old_cat_face_loc + 0.1*cat_face_loc
    old_cat_face_loc = cat_face_loc
    center = (int(cat_face_loc[0]+cat_face_loc[2]/2), int(cat_face_loc[1]+cat_face_loc[3]*0.8))#0.8为手动设定的猫嘴位置，因为没找到猫脸landmark

    normal_clone = cv2.seamlessClone(mouth, cat, mask.astype(mouth.dtype), center, cv2.NORMAL_CLONE)
    #mixed_clone = cv2.seamlessClone(mouth, cat, mask.astype(mouth.dtype), center, cv2.MIXED_CLONE)
    #cv2.imwrite("opencv-normal-clone-example.jpg", normal_clone)
    #cv2.imwrite("opencv-mixed-clone-example.jpg", mixed_clone)
    return normal_clone

#human = cv2.imread('bigmouth.jpg')
#cat = cv2.imread('cat.jpg')
#cat_with_human_mouth = human_mouth_paste_to_cat(human,cat)

human_video_cap = cv2.VideoCapture("human_video.mp4")
cat_video_cap =cv2.VideoCapture("cat_video2.mp4")

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video_writer = cv2.VideoWriter('cat_with_humanmouth.avi', fourcc, 29.43, (720,1280))

index=0
while(True):
    print(index)
    index+=1
    if(index>400):
        break
    human_ret,human_frame = human_video_cap.read()
    human_frame = cv2.resize(human_frame,dsize=None,fx=2,fy=2)
    cat_ret,cat_frame = cat_video_cap.read()
    if(human_ret == True and cat_ret == True):
        cat_with_human_mouth = human_mouth_paste_to_cat(human_frame,cat_frame)
        video_writer.write(cat_with_human_mouth.astype(np.uint8))

video_writer.release()