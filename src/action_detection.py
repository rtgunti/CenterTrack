from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
class ActionDetection():
    def __init__(self, opt):
        self.opt = opt
        self.opt.inp_fr = 30
        self.kti = {
            "nose":0,
            "left_eye":1,
            "right_eye":2,
            "left_ear":3,
            "right_ear":4,
            "left_shoulder":5, 
            "right_shoulder":6,
            "left_elbow":7,
            "right_elbow":8,
            "left_wrist":9,
            "right_wrist":10,
            "left_hip":11,
            "right_hip":12,
            "left_knee":13,
            "right_knee":14,
            "left_ankle":15,
            "right_ankle":16
            }

    def detect_action(self, cnt, results, det_hist):
        # top_crease_x = [510, 804]     #Precise points
        # bottom_crease_x = [483, 846]
        # top_crease_y = [304]
        # bottom_crease_y = [488]
        top_crease_x = [500, 810]       #Add buffer
        bottom_crease_x = [470, 860]
        top_crease_y = [290, 310]
        bottom_crease_y = [483, 492]
        std_thresh = 12
        bowling_df_frame = []
        for ind, res in enumerate(results):
            if results[ind]['score'] > self.opt.vis_thresh:
                if 'active' in results[ind] and results[ind]['active'] == 0  and len(det_hist[res['tracking_id']]) < 30:
                    continue
                hist_pts = np.array(det_hist[res['tracking_id']])
                std = np.std(hist_pts[-30:], axis=0)
                action_text = "id : " + str(results[ind]['tracking_id'])
                hps = np.array(results[ind]['hps'], dtype=np.int32).reshape(-1, 2)
                # abs_ls_le = abs(hps[self.kti["left_ear"], 1] - hps[self.kti["left_shoulder"], 1])
                # abs_rs_re = abs(hps[self.kti["right_ear"], 1] - hps[self.kti["right_shoulder"], 1])
                foot_point = [(res['bbox'][0] + res['bbox'][2])/2, res['bbox'][3]]
                action_lb = hps[self.kti["left_wrist"], 1] < hps[self.kti["left_elbow"], 1] < hps[self.kti["left_shoulder"], 1]
                action_rb = hps[self.kti["right_wrist"], 1] < hps[self.kti["right_elbow"], 1] < hps[self.kti["right_shoulder"], 1]
                # if ( res['tracking_id'] == 3):
                #     print(cnt, hps[self.kti["left_wrist"], 1], hps[self.kti["left_elbow"], 1], hps[self.kti["left_shoulder"], 1], " abs dist : ", abs_ls_le)
                # if hps[self.kti["left_wrist"], 1] < hps[self.kti["left_elbow"], 1] < hps[self.kti["left_shoulder"], 1] and stat_std > 10 and abs_ls_le < 5:
                #     out_strings.append(action_text + " Left arm Bowling detected" + str(stat_std) + " abs dis: " + str(abs_ls_le))
                # if hps[self.kti["right_wrist"], 1] < hps[self.kti["right_elbow"], 1] < hps[self.kti["right_shoulder"], 1] and stat_std > 10 and abs_rs_re < 5:
                #     out_strings.append(action_text + " Right arm Bowling detected" + str(stat_std) + " abs dist : " + str(abs_rs_re))
                # if res['tracking_id'] in [3]:
                #     print(res['tracking_id'], foot_point[1], "std", std, "bbox", res['bbox'])
                #     print(hps[self.kti["left_wrist"], 1], hps[self.kti["left_elbow"], 1], hps[self.kti["left_shoulder"], 1], action_lb)
                #     print(hps[self.kti["right_wrist"], 1], hps[self.kti["right_elbow"], 1], hps[self.kti["right_shoulder"], 1], action_rb)
                if 290 < foot_point[1] < 310 and top_crease_x[0] < foot_point[0] < top_crease_x[1] and std[1] > std_thresh and (action_lb or action_rb): # 304
                    results[ind].update({'action': 'bowl_top'})
                    bowling_df_frame = [cnt, str(datetime.timedelta(seconds = int(cnt/30))), res['bbox'], res['hps'], res['tracking_id'], 'top']
                elif 483 < foot_point[1] < 492 and bottom_crease_x[0] < foot_point[0] < bottom_crease_x[1] and std[1] > std_thresh and (action_lb or action_rb): # 488
                    results[ind].update({'action': 'bowl_bottom'})
                    bowling_df_frame = [cnt, str(datetime.timedelta(seconds = int(cnt/30))), res['bbox'], res['hps'], res['tracking_id'], 'bottom']
                else:
                    results[ind].update({'action': 'idle'})

        return bowling_df_frame, results
