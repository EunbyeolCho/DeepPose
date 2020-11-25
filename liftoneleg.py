import numpy as np
import math


def get_angle(pt_pair1, pt_pair2):

    ax,ay = pt_pair1[1][0] - pt_pair1[0][0], pt_pair1[1][1] - pt_pair1[0][1]
    bx,by = pt_pair2[1][0] - pt_pair2[0][0], pt_pair2[1][1] - pt_pair2[0][1]
    mod_a = math.sqrt(ax**2 + ay**2)
    mod_b = math.sqrt(bx**2 + by**2)
    angle = math.degrees(math.acos((ax*bx + ay*by) / ((mod_a*mod_b) or 1) ))
    if(angle==math.nan):
        angle=-1 
    return angle


class LiftOneLeg():

    def __init__(self, points):

        self.num_frames = len(points)
        self.kpt_list = points
        self.kpt = points[-1]

    # def check_if_hand_on_chest(self):
    #     """
    #     Right wrists is upper than right elbow
    #     """
    #     if self.kpt[3][1] > self.kpt[4][1]:
    #         return True
        
    #     return False

    def check_if_3points_are_aligned(self):
        """
        Angle between Left shoulder, hip, ankle
        """
        alinged_angle = get_angle((self.kpt[1], self.kpt[4]), (self.kpt[4], self.kpt[5]))
        if(alinged_angle < 5):
            return True
        return False
    
    def check_if_shoulders_are_aligned(self):
        """
        Anlge between Right shoulder, Left shoulder
        """
        kpt1 = np.int_([0,0])
        kpt2 = np.int_([1,0])
        angle = get_angle((self.kpt[0], self.kpt[1]), (kpt1, kpt2))
        if(angle < 5):
            return True
        return False

    def check_leg_up_down(self):

        leg_up, leg_down, leg_stop = False, False, False
        avg_per_frame = 5
        angle = -1
        
        if (self.num_frames > avg_per_frame*3):

            frame1 = -avg_per_frame
            frame2 = -avg_per_frame*2
            frame3 = -avg_per_frame*3
 
            # kpt_30ago = np.mean(self.kpt_list[frame3:frame2], axis=0)
            # kpt_15ago = np.mean(self.kpt_list[frame2:frame1], axis=0)
            # kpt_5ago = np.mean(self.kpt_list[frame1:], axis=0)

            kpt_30ago = np.median(self.kpt_list[frame3:frame2], axis=0)
            kpt_15ago = np.median(self.kpt_list[frame2:frame1], axis=0)
            kpt_5ago = np.median(self.kpt_list[frame1:], axis=0)


            kpt1 = np.int_([0,0])
            kpt2 = np.int_([0,1])

            
            angle_more_prev = get_angle((kpt_30ago[2], kpt_30ago[3]), (kpt1, kpt2))
            angle_prev = get_angle((kpt_15ago[2], kpt_15ago[3]), (kpt1, kpt2))

            if self.kpt[2][0] > self.kpt[4][0]:
                self.kpt[2] = self.kpt[4]
            if self.kpt[3][0] > self.kpt[5][0]:
                self.kpt[3] = self.kpt[5]
                
            angle = get_angle((self.kpt[2], self.kpt[3]), (kpt1, kpt2))

            #Up
            if( angle >= angle_prev and angle_prev >= angle_more_prev and angle >5):
                leg_up = True
            #Down
            elif( angle < angle_prev and angle_prev < angle_more_prev and angle < 10):
                leg_down = True
            #Stop
            else :
                leg_stop =True

        leg_status = [leg_up, leg_down, leg_stop]
        return angle, leg_status

    def count_repetition(self, angle, leg_status, completed_half,  count, num_frame, start_frame, end_frame) :
        
        if leg_status == [True, False, False]: 
            completed_half = True
            if num_frame < start_frame:
                start_frame  = num_frame

        if leg_status == [False, False, True]:
            if (end_frame !=  -1 and angle < 5):
                
                #초기화
                start_frame, end_frame = 1000000, -1
                leg_status = [False, False, False]
                completed_half = False
                count +=1
        
        if (leg_status == [False, True, False] and completed_half):
            if num_frame > end_frame:
                end_frame  = num_frame
        
        return leg_status, completed_half, count, start_frame, end_frame
