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