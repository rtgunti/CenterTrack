from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
class ActionDetection():
    def __init__(self, opt):
        self.opt = opt
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
        out_strings = []
        out_strings.append("Frame : " + str(cnt))
        for ind, res in enumerate(results):
            if results[ind]['score'] > self.opt.vis_thresh:
                if 'active' in results[ind] and results[ind]['active'] == 0:
                    continue
                hist_pts = np.array(det_hist[res['tracking_id']])
                stat_std = np.std(hist_pts[:-30], axis=0)[1]
                action_text = "id : " + str(results[ind]['tracking_id'])
                hps = np.array(results[ind]['hps'], dtype=np.int32).reshape(-1, 2)
                abs_ls_le = abs(hps[self.kti["left_ear"], 1] - hps[self.kti["left_shoulder"], 1])
                abs_rs_re = abs(hps[self.kti["right_ear"], 1] - hps[self.kti["right_shoulder"], 1])
                if ( res['tracking_id'] == 3):
                    print(cnt, hps[self.kti["left_wrist"], 1], hps[self.kti["left_elbow"], 1], hps[self.kti["left_shoulder"], 1], " abs dist : ", abs_ls_le)
                if hps[self.kti["left_wrist"], 1] < hps[self.kti["left_elbow"], 1] < hps[self.kti["left_shoulder"], 1] and stat_std > 10 and abs_ls_le < 5:
                    out_strings.append(action_text + " Left arm Bowling detected" + str(stat_std) + " abs dis: " + str(abs_ls_le))
                if hps[self.kti["right_wrist"], 1] < hps[self.kti["right_elbow"], 1] < hps[self.kti["right_shoulder"], 1] and stat_std > 10 and abs_rs_re < 5:
                    out_strings.append(action_text + " Right arm Bowling detected" + str(stat_std) + " abs dist : " + str(abs_rs_re)) 
                else:
                    out_strings.append(action_text + " idle")

        print(out_strings)
        return out_strings