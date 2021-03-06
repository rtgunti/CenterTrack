from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import sys
import cv2
import json
import copy
import numpy as np
import pandas as pd
from opts import opts
from action_detection import ActionDetection
from detector import Detector
from vidgear.gears import CamGear
from frame_time import timecode_to_frames, frames_to_timecode
import time


image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'display']
action_col_names = ['frame_cnt', 'timestamp', 'bbox', 'hpe', 'tracking_id', 'direction']  #hpe to be normalized using bbox


def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  detector = Detector(opt)

  org = (50, 100) 
  font = cv2.FONT_HERSHEY_SIMPLEX
  fontScale = 1
  color = (255, 0, 0) 
  thickness = 2

  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    is_video = True
    # demo on video stream
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
  elif 'http' in opt.demo:
    is_video = True
    cam = CamGear(source=opt.demo, stream_mode = True, logging=True).start() 
  else:
    is_video = False
    # Demo on images sequences
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]

  # Initialize output video
  out = None
  out_name = opt.demo[opt.demo.rfind('/') + 1:]
  print('out_name', out_name)
  if not os.path.exists('../results'):
    os.mkdir('../results')
  if opt.save_video:
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter('../results/{}.avi'.format(
      opt.exp_id + '_' + out_name[:-4]),fourcc, opt.save_framerate, (
        opt.video_w, opt.video_h))
  
  if opt.debug < 5:
    detector.pause = False
  cnt = 0
  results = {}
  det_hist = {}

  ad = ActionDetection(opt)
  bowling_df = pd.DataFrame(columns = action_col_names)
  
  last_det_cnt = {}
  flag_dup_det = {}
  if opt.start_time and not opt.start_frame:
    opt.start_frame = timecode_to_frames(opt.start_time)
    opt.end_frame = timecode_to_frames(opt.end_time)
  
  cam.set(cv2.CAP_PROP_POS_FRAMES, opt.start_frame)
  cnt = opt.start_frame
  if opt.end_frame == -1:
    opt.end_frame = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
  print("Frame range", opt.start_frame, opt.end_frame)
  ex_start_time = time.time()
  while True:
      if is_video:
        if 'http' in opt.demo:
          img = cam.read()
        else:
          _, img = cam.read()
        if img is None or cnt > opt.end_frame:
          print("time taken to process ", opt.end_frame - opt.start_frame, "frames", time.time() - ex_start_time)
          bowling_df.to_pickle('/content/drive/MyDrive/cric_actions/results/results_df_' + str(opt.start_frame) + '_' + str(opt.end_frame) + '.df')
          save_and_exit(opt, out, results, out_name)
      else:
        if cnt < len(image_names):
          img = cv2.imread(image_names[cnt])
        else:
          save_and_exit(opt, out, results, out_name)
      cnt += 1

      # resize the original video for saving video results
      if opt.resize_video:
        img = cv2.resize(img, (opt.video_w, opt.video_h))

      # skip the first X frames of the video
      if cnt < opt.skip_first:
        continue
      
      # cv2.imshow('input', img)

      # track or detect the image.
      ret = detector.run(img)

      # log run time
      time_str = 'frame {} |'.format(cnt)
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      # print(time_str)

      # results[cnt] is a list of dicts:
      #  [{'bbox': [x1, y1, x2, y2], 'tracking_id': id, 'category_id': c, ...}]
      results[cnt] = ret['results']

      for res in ret['results']:
        if res['score'] > opt.vis_thresh:
            if 'active' in res and res['active'] == 0:
                continue
            if res['tracking_id'] in det_hist:
              det_hist[res['tracking_id']] = np.append(det_hist[res['tracking_id']], res['ct'].reshape(1,2), axis = 0)
            else:
              det_hist[res['tracking_id']] = np.array(res['ct'].reshape(1, 2))

      bowling_df_frame, results[cnt] = ad.detect_action(cnt, ret['results'], det_hist)
      print("[Frame : "+ str(cnt) + "]")
      for ind, res in enumerate(results[cnt]):
        ret['generic'] = cv2.putText(ret['generic'], str(res['tracking_id']) + " : " + res['action'], (org[0], org[1] + (ind+1)*50), font,  
                   fontScale, color, thickness, cv2.LINE_AA)      
      ret['generic'] = cv2.putText(ret['generic'],  "Frame : " + str(cnt), (org[0], org[1]), font,  
                  fontScale, color, thickness, cv2.LINE_AA)   

      if bowling_df_frame:
        tracking_id = bowling_df_frame[-2]
        if not tracking_id in last_det_cnt:
          last_det_cnt.update({tracking_id:cnt})
        if not tracking_id in flag_dup_det:
          flag_dup_det.update({tracking_id:False})

        if cnt > last_det_cnt[tracking_id] + 50:
          flag_dup_det[tracking_id] = False

        if not flag_dup_det[tracking_id]:
          print("bowling detected by ", tracking_id)
          cv2.imwrite('/content/drive/MyDrive/cric_actions/results/demo{}.jpg'.format(cnt), ret['generic'])
          bowling_df.loc[len(bowling_df)] = bowling_df_frame
          last_det_cnt.update({tracking_id:cnt})
          flag_dup_det[tracking_id] = True
        else:
          [results[cnt][ind].update({'action':'idle'}) for ind, res in enumerate(results[cnt]) if res['tracking_id'] == tracking_id]

      # save debug image to video
      if opt.save_video:
        out.write(ret['generic'])
        if not is_video:
          cv2.imwrite('../results/demo{}.jpg'.format(cnt), ret['generic'])

      if cnt%1000 == 0:
        save_and_exit(opt, out, results, "inter" + out_name, False)
        bowling_df.to_pickle('/content/drive/MyDrive/cric_actions/results/results.df')

      # esc to quit and finish saving video
      if cv2.waitKey(1) == 27:
        save_and_exit(opt, out, results, out_name)
        return 
  
  save_and_exit(opt, out, results)


def save_and_exit(opt, out=None, results=None, out_name='', exit = True):
  if opt.save_results and (results is not None):
    save_dir =  '/content/drive/MyDrive/cric_actions/results/{}_results_{}_{}.json'.format(opt.exp_id + '_' + out_name[:-4], opt.start_frame, opt.end_frame)
    print('saving results to', save_dir)
    json.dump(_to_list(copy.deepcopy(results)), 
              open(save_dir, 'w'))
  if opt.save_video and out is not None:
    out.release()
  if exit:
    sys.exit(0)

def _to_list(results):
  for img_id in results:
    for t in range(len(results[img_id])):
      for k in results[img_id][t]:
        if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
          results[img_id][t][k] = results[img_id][t][k].tolist()
  return results

if __name__ == '__main__':
  opt = opts().init()
  opt.video_h = 720
  opt.video_w = 1280
  opt.max_age = 5
  opt.save_framerate = 10
  opt.start_time = "05:31:10"
  opt.end_time = "07:11:43"
  # opt.start_frame = 1
  # opt.end_frame = 10
  print("Frame range", opt.start_frame, opt.end_frame)
  demo(opt)
