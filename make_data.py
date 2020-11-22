import csv
import numpy as np
import cv2
import math
from numpy import dot
from numpy.linalg import norm

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                "LEye": 15, "REar" :16, "Lear" :17 }

POSE_PAIRS = [ ["Nose", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["RShoulder", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["LShoulder", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]

POSE_PAIRS = [ ["RShoulder", "LShoulder"],["LShoulder", "LHip"], ["LHip", "LAnkle"],
                ["RHip", "RKnee"], ["RKnee", "RAnkle"]]

COLOR_TABLE =[(255,0,0), (255,0,0), 
            (0,255,0),(0,255,0),(0,255,0),
            (0,0,255),(0,0,255),(0,0,255),
            (255,255,0),(255,255,0),(255,255,0),
            (0,255,255),(0,255,255),(0,255,255),
            (255,0,255),(255,0,255),(255,0,255),(255,0,255)]

def GetCosineSimilarity(A, B):

    a0, a1 = A[0] + 0.001 , A[1] + 0.001
    b0, b1 = B[0] + 0.001, B[1] + 0.001 

    a = tuple([a0, a1])
    b = tuple([b0, b1])

    csim = dot(a, b)/(norm(a)*norm(b))
    if csim > 1 : csim = 1


    return csim

def similarity_score(input_points, gt_points):

    # csim_sum = 0    
    csim_sum = np.zeros([len(POSE_PAIRS), ])
    xaxis = 0
    for i, pair in enumerate(POSE_PAIRS):
        # print(pair, end=" : ")
        xaxis += 0.5
        partA = pair[0]             # Head
        partA = BODY_PARTS[partA]   # 0
        partB = pair[1]             # Neck
        partB = BODY_PARTS[partB]   # 1

        a = np.array(input_points[partB]) - np.array(input_points[partA])
        b = np.array(gt_points[partB]) - np.array(gt_points[partA])
        # print(a,b)
        
        csim = abs(GetCosineSimilarity(a, b))
        # degree = math.degrees(math.acos(csim))
        
        scaled_csim = (2 * csim) -1
        scaled_csim = (10 * csim) -9
        
        csim_sum[i] += scaled_csim
    

    return csim_sum
    
# def similarity_score(input_points, gt_points):

#     csim_sum = 0
#     distance_sum = 0
    
#     # print(len(POSE_PAIRS))#5
#     csim_sum = np.zeros([len(POSE_PAIRS), ])

#     for i, pair in enumerate(POSE_PAIRS):
#         # print(pair, end=" : ")
#         partA = pair[0]             # Head
#         partA = BODY_PARTS[partA]   # 0
#         partB = pair[1]             # Neck
#         partB = BODY_PARTS[partB]   # 1

#         a = np.array(input_points[partB]) - np.array(input_points[partA])
#         b = np.array(gt_points[partB]) - np.array(gt_points[partA])
        
#         # print(a,b)
        
#         csim = abs(GetCosineSimilarity(a, b))
#         # print(int(math.degrees(math.acos(csim)) ))
#         scaled_csim = (2 * csim) -1
#         scaled_csim = (10 * csim) -9
#         # print(math.degrees(math.acos(GetCosineSimilarity(a, b))), scaled_csim)
#         csim_sum[i] += scaled_csim
    
    
#     return csim_sum

            
def visualize_data(data):

    num_frame, num_kpt, xy = data.shape
    # print(data.shape)
    img = np.zeros([600,600,3])
    org_img = img.copy()

    for i in range(num_frame):

        for j in range(num_kpt):

            h = data[i][j][0]
            w = data[i][j][1]
            cv2.circle(img, (int(h), int(w)), 3, COLOR_TABLE[j], thickness=-1, lineType=cv2.FILLED)
        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        img = np.zeros([600,600,3])
        key = cv2.waitKey(33)
        
        if key == 27:  # esc
            return  

def visualize_comparing(data, gt):

    gt_num_frame, num_kpt, xy = gt.shape
    num_frame,  _, _ = data.shape

    gt_img = np.zeros([600,600,3])
    img = np.zeros([600,600,3])
    # total_img = np.zeros([600,1200,3])

    ratio  = num_frame / gt_num_frame

    for i in range(gt_num_frame):

        for j in range(num_kpt):

            h_gt = gt[i][j][0]
            w_gt = gt[i][j][1]

            # data_hidx = int(ratio *i)
            # data_widx = int(ratio *j)
            h = data[int(ratio *i)][j][0]
            w = data[int(ratio *i)][j][1]

            # print(h_gt, w_gt, h, w)
            
            
            cv2.circle(gt_img, (int(h_gt), int(w_gt)), 3, COLOR_TABLE[j], thickness=-1, lineType=cv2.FILLED)
            cv2.circle(img, (int(h), int(w)), 3, COLOR_TABLE[j], thickness=-1, lineType=cv2.FILLED)

        # print(numpy_vertical.shape)
        numpy_vertical = np.hstack((img, gt_img))
        cv2.imshow('Lightweight Human Pose Estimation Python Demo', numpy_vertical)
        total_img = np.zeros([600,1200,3])
        gt_img = np.zeros([600,600,3])
        img = np.zeros([600,600,3])

        key = cv2.waitKey(33)
        
        if key == 27:  # esc
            return  

def test(data, gt):


    data = np.array(data)
    gt_num_frame, num_kpt, xy = gt.shape
    num_frame,  _, _ = data.shape
    ratio  = num_frame / gt_num_frame
        
    idx = [i*ratio for i in range(gt_num_frame)]
    total_score = 0
    count = 0
    for i in range(gt_num_frame):
        time_scaled_data = data[int(idx[i])]
        score = similarity_score(time_scaled_data, gt[i])
        total_score += score
        count +=1
        # print(count, "==> ", score)
    
    return total_score/gt_num_frame

def test_per_frame(data, gt, fps=10):


    data = np.array(data)
    gt_num_frame, num_kpt, xy = gt.shape
    num_frame,  _, _ = data.shape
    ratio  = num_frame / gt_num_frame
        
    idx = [int(i*ratio) for i in range(gt_num_frame)]
    total_score = 0
    count = 0

    # 0-92 : 10 frame의 median value?
    test_frame = int(gt_num_frame/fps) #18

    for i in range(test_frame):
        
        time_scaled_data = np.median(data[idx[i*fps : i*fps+fps]], axis=0)
        gt_data= np.median(gt[i*fps : i*fps+fps], axis=0)
    
        score = similarity_score(time_scaled_data, gt_data)
        total_score += score
        count +=1

    return total_score/test_frame*100

def visualize_particular_frame(data, gt, target, fps=10):

    gt_num_frame, num_kpt, xy = gt.shape
    num_frame,  _, _ = data.shape

    gt_img = np.zeros([600,600,3])
    img = np.zeros([600,600,3])

    ratio  = num_frame / gt_num_frame
    idx = [int(i*ratio) for i in range(gt_num_frame)]
    test_frame = int(gt_num_frame/fps) #18

    #Target frame
    time_scaled_data = np.median(data[idx[target*fps : target*fps+fps]], axis=0)
    gt_data= np.median(gt[target*fps : target*fps+fps], axis=0)

    for j in range(num_kpt):

        h_gt = gt_data[j][0]
        w_gt = gt_data[j][1]

        h = time_scaled_data[j][0]
        w = time_scaled_data[j][1]

        # print(h_gt, w_gt, h, w)
        
        
        cv2.circle(gt_img, (int(h_gt), int(w_gt)), 3, COLOR_TABLE[j], thickness=-1, lineType=cv2.FILLED)
        cv2.circle(img, (int(h), int(w)), 3, COLOR_TABLE[j], thickness=-1, lineType=cv2.FILLED)

    # print(numpy_vertical.shape)
    numpy_vertical = np.hstack((img, gt_img))
    cv2.imshow('Lightweight Human Pose Estimation Python Demo', numpy_vertical)
    total_img = np.zeros([600,1200,3])
    gt_img = np.zeros([600,600,3])
    img = np.zeros([600,600,3])

    key = cv2.waitKey(0)
    
    if key == 27:  # esc
        return  



if __name__ == '__main__':

    data1 = np.load('./result/gt_1.npy')
    gt = np.load('./result/test.npy')
    
    video_path = '/home/eunbyeol/reflective/result/orange_wrong_1111.npy'
    video_path = '/home/eunbyeol/reflective/result/orange_correct_1111.npy'
    video_path = '/home/eunbyeol/reflective/result/lift_oneleg_wrong1.npy'

    example = np.load(video_path)
    
    # print(gt.shape) #93,18,2
    visualize_particular_frame(example, gt, 6)

    


