import os
from google.protobuf import text_format

import caffe, caffe.draw
from caffe.proto import caffe_pb2
import glob
import leveldb
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import ntpath
from sklearn.neighbors import NearestNeighbors

import sys

def GetKeyList(key_list_file_name):
  key_list = {}
  id_to_key_list = []
  key_list_file = open(key_list_file_name)
  ctr = 0
  for line in key_list_file:
    line_splits = line.split(':')
    video_name = line_splits[0]
    shot_id = int(line_splits[1])
    if video_name not in key_list:
      key_list[video_name] = []
    else:
      key_list[video_name].append([shot_id, ctr])
    id_to_key_list.append([video_name, shot_id])
    ctr = ctr + 1
  key_list_file.close()

  for key in key_list:
    key_list[key] = sorted(key_list[key], key = lambda x: x[0])

  return key_list, id_to_key_list

def GetFrameNames(vid_name, shot, ev_dir_map):
  if vid_name not in ev_dir_map:
    return None
  
  frame_dir = ev_dir_map[vid_name]
  frame_dir = frame_dir + '/shot%05d/'%(shot)
  image_list = glob.glob(frame_dir + '/*.jpeg')
  if len(image_list) > 1:
    return image_list[1]
  if len(image_list) == 1:
    return image_list[0]
  else:
    return None

def main(argv):

  print('Entering main')

  shot_frames_dev_dir = '/afs/cs.stanford.edu/u/vigneshr/scratch2/ICCV2015/data/MED_dev_set/shot_keyframe_fc7_features/shot_frames/'
  shot_frames_event_dir = '/afs/cs.stanford.edu/u/vigneshr/scratch2/ICCV2015/data/MED_event_kits/shot_keyframe_fc7_features/shot_frames/'
  dev_frames_dir = glob.glob(shot_frames_dev_dir + '/HVC*')
  ev_frames_dir = glob.glob(shot_frames_event_dir + '/*/HVC*')
  ev_dir_map = {ntpath.basename(e) : e for e in ev_frames_dir}
  dev_dir_map = {ntpath.basename(e) : e for e in dev_frames_dir}
  dir_map = dict(ev_dir_map.items() + dev_dir_map.items())

	#wordvec_table = '/afs/cs.stanford.edu/u/vigneshr/scratch2/ICCV2015/data/_temp/wordvec_med_eculid_64'
  wordvec_table = '/afs/cs.stanford.edu/u/vigneshr/scratch2/ICCV2015/data/_temp/wordvec_med_normalized_eculid_64'

  if len(argv) >= 2:
    wordvec_table = argv[1]

  key_list_file_name = '/afs/cs.stanford.edu/u/vigneshr/scratch2/ICCV2015/data/med_dev_event_kit_shots_only_leveldb/key_list.txt'
  if len(argv) >= 3:
    key_list_file_name = argv[2]

  key_list, id_to_key_list = GetKeyList(key_list_file_name)
  wordvec_db = leveldb.LevelDB(wordvec_table)
  print('Opened leveldb')
  all_wordvecs = []
  for id in range(len(id_to_key_list)):
    datum = caffe_pb2.Datum()
    datum.ParseFromString(wordvec_db.Get('%d'%id))
    wordvec = np.array(datum.float_data)
    wordvec = np.reshape(wordvec, [wordvec.shape[0], 1])
    all_wordvecs.append(wordvec)
    if id % 1000 == 0:
      print('Done reading %d/%d values'%(id, len(id_to_key_list)))
    #if id >= 1000:
    #  break
  wordvec_matrix = np.concatenate(all_wordvecs, axis=1)
  print wordvec_matrix.shape
  kdt = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(wordvec_matrix.T)

  #input_video_name = 'HVC757596'
  #input_video_name = ''
  vis_id = 3
  input_video_name = raw_input('Enter the video to look for: ')
  while not input_video_name == 'done':
    if input_video_name in key_list:
      shots = key_list[input_video_name]

      for shot_id, ctr in shots:
        datum = caffe_pb2.Datum()
        datum.ParseFromString(wordvec_db.Get('%d'%ctr))
        wordvec = np.array(datum.float_data)
        wordvec = np.reshape(wordvec, [1, wordvec.shape[0]])
        print ('shot-id: %d ------'%(shot_id))
        
        query_shot_frame = GetFrameNames(input_video_name, shot_id, dir_map)
        print ('Query: %s'%(query_shot_frame))
        if query_shot_frame:
          query_image = misc.imread(query_shot_frame)
          query_image = misc.imresize(query_image, [100, 100])
          plt.subplot(len(shots), 6, ((shot_id-1)*6)+1)
          fig = plt.imshow(query_image)
          fig.axes.get_xaxis().set_visible(False)
          fig.axes.get_yaxis().set_visible(False)


        nnbrs = kdt.kneighbors(wordvec, return_distance=False)
        nearest_shots = [id_to_key_list[nn] for nn in nnbrs[0]]
        nid = 0
        for vid_name, shot in nearest_shots:
          nn_shot_frame = GetFrameNames(vid_name, shot, dir_map)
          nid = nid + 1
          if nn_shot_frame:
            nn_image = misc.imread(nn_shot_frame)
            nn_image = misc.imresize(nn_image, [100, 100])
            plt.subplot(len(shots), 6, ((shot_id-1)*6) + 1 + nid)
            fig = plt.imshow(nn_image)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
          print('NN: %s'%(nn_shot_frame))
      plt.savefig('vis_dir/vis_image_%s.png'%(input_video_name))
      vis_id = vis_id + 1
    else:
      print('Could not find %s video in list'%(input_video_name))
    input_video_name = raw_input('Enter the video to look for: ')

    #break

if __name__ == '__main__':
  main(sys.argv)
