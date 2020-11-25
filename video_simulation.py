import argparse
import cv2
import numpy as np
import torch
import math
import os
import matplotlib.pyplot as plt

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, propagate_ids
from val import normalize, pad_width

#Add
from inference import infer_fast, VideoReader
from liftoneleg import LiftOneLeg, get_angle
from metric import test_per_frame

#Global varaible
# LABEL = np.load('./result/test.npy')
LABEL = np.load('./result/groundtruth.npy')



def run_demo(net, image_provider, height_size=256, cpu=False, track_ids=False):

    net = net.eval()
    if not cpu:
        net = net.cuda()
        print("use cuda")

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts # 18
    
    #Initialize
    previous_pose_kpts = []
    graph_x, graph_y = [], []
    result = [-1, -1, -1, -1, -1]

    count = 0
    start_frame, end_frame = 1000000, -1
    completed_half = False
    total_len_frame = 0
    one_cycle_kpts =[]

    for i, img in enumerate(image_provider):
        
        img = cv2.resize(img, (600,600))
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)
       
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)
        #total_keypoints_num = 18

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            ####
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)
            pose.draw(img) 

        
        #Select joints
        pose_keypoints = np.concatenate((pose_keypoints[2],pose_keypoints[5],pose_keypoints[8],
                                        pose_keypoints[10],pose_keypoints[11],pose_keypoints[13])).reshape(-1, 2)
        #Analyze posture
        previous_pose_kpts.append(pose_keypoints)
        liftoneleg = LiftOneLeg(previous_pose_kpts)#Wrong
        angle, leg_status = liftoneleg.check_leg_up_down() 
        
        #Update status and count
        leg_status, completed_half, count_update, start_frame_update, end_frame_update= \
                    liftoneleg.count_repetition(angle, leg_status, completed_half,  count, i, start_frame, end_frame)
        if (count_update == count +1):
            print("count : %d" %count)

            one_cycle_kpts.append(previous_pose_kpts[start_frame:])
            
            result = test_per_frame(previous_pose_kpts[start_frame-total_len_frame:end_frame-total_len_frame], LABEL)
            total_len_frame += len(previous_pose_kpts)
            previous_pose_kpts = []
            
            

        count, start_frame, end_frame = count_update, start_frame_update, end_frame_update

        #To plot angle graph
        if int(angle) != 90 :
            graph_x.append(i)
            graph_y.append(angle)

        
        #Put text on the screen
        cv2.putText(img, 'count : {}'.format(count),
                    (10,520), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        cv2.putText(img, "Rsho-Lsho :%3.2f" %(result[0]),
                    (10,550), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0),2)
        cv2.putText(img, "Lsho-Lhip :%3.2f, Lhip-Lank :%3.2f" %(result[1], result[2]),
                    (10,570), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0),2)
        cv2.putText(img, "Rhip-Rank :%3.2f" %(result[3]),
                    (10,590), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0),2)
        cv2.putText(img, '3 align :{}'.format(liftoneleg.check_if_3points_are_aligned()),
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255),2)
        cv2.putText(img, 'shoulder :{}'.format(liftoneleg.check_if_shoulders_are_aligned()),
                    (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255),2)
    

        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        key = cv2.waitKey(33)
        if key == 27:  # esc
            return

    return graph_x, graph_y
    # return one_cycle_kpts


    
if __name__ == '__main__':


    checkpoint_path = './checkpoint_iter_370000.pth'
    # video_path = '../openpose/rehab_data/ligt_oneleg_correct.mp4'
    # video_path = '/home/eunbyeol/openpose/rehab_data/lift_oneleg_wrong1.mp4'
    # video_path = '/home/eunbyeol/openpose/rehab_data/liftoneleg.mp4'

    video_path = '/home/eunbyeol/openpose/rehab_data/wrong_1111.mp4'
    # video_path = '/home/eunbyeol/openpose/rehab_data/correct_1111.mp4'
    # video_path = '/home/eunbyeol/openpose/rehab_data/3set_video.mp4'

    
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)
    
    correct_video_frame_provider = VideoReader(video_path)
    graph_x, graph_y =run_demo(net, correct_video_frame_provider)
    # one_cycle_kpts = run_demo(net, correct_video_frame_provider)
    # np.save('./result/groundtruth.npy', one_cycle_kpts[1])

    
    # plt.plot(graph_x, graph_y, color="red",  marker=".",)
    # plt.xlabel('frame #')
    # plt.ylabel('angle')
    # plt.show()

