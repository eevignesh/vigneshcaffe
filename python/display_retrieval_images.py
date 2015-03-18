import os
import glob
import matplotlib.pyplot as plt
from scipy import misc
import sys
import numpy as np

#stat_file_1='/data2/vigneshr/vigneshcaffe/misc/mednet_pair_relu_1_results_test_40k.txt'
stat_file_1='/data2/vigneshr/vigneshcaffe/misc/mednet_pair_fc7_results_test_event.txt'
stat_file_2='/data2/vigneshr/vigneshcaffe/misc/mednet_cont_2_skpgram_test_22k.txt'

image_dir='/afs/cs.stanford.edu/u/vigneshr/scratch2/ICCV2015/data/MED_event_kits/shot_keyframe_fc7_features/shot_frames/'
video_list_file='/scail/scratch/u/vigneshr/ICCV2015/code/med_svm_models/test_event_kit_list.txt'
id_to_video_shot_list='/data2/vigneshr/vigneshcaffe/misc/test_event_kid_item_id_to_video_shot_ids.txt'
output_directory = '/afs/cs.stanford.edu/group/vision/www/vigneshr_data/retrieval_images_full_cont_12k_fc7/'


def parse_stat_file(stat_file):
# Parse the stat file first
  fid = open(stat_file, 'r')
  top_5_images = []
  top_5_classes = []
  acc_at_5 = []
  ap = []
  class_ids = []
  line_ctr = 0
  for line in fid:
# skip first line
    if line_ctr == 0:
      line_ctr = 1
      continue
    line_splits = line.split(',')
    line_nums = [float(line_splits[i]) for i in range(len(line_splits))]
    top_5_images.append((line_nums[5], line_nums[6], line_nums[7], line_nums[8], line_nums[9]))
    top_5_classes.append((line_nums[10], line_nums[11], line_nums[12], line_nums[13], line_nums[14]))
    acc_at_5.append(line_nums[4])
    class_ids.append(line_nums[1])
    ap.append(line_nums[2])
  fid.close()
  return top_5_images, top_5_classes, acc_at_5, class_ids, ap

if __name__ == "__main__":

  top_5_images, top_5_classes, acc_at_5, class_ids, ap = parse_stat_file(stat_file_1)
  top_5_images_2, top_5_classes_2, acc_at_5_2, class_ids_2, ap_2 = parse_stat_file(stat_file_2)

  mean_ap_1 = {}
  mean_ap_2 = {}
  num_shots_1 = {}
  num_shots_2 = {}

  for i in range(len(ap)):
    if class_ids[i] not in mean_ap_1:
      mean_ap_1[class_ids[i]] = 0
      num_shots_1[class_ids[i]] = 0.0
    if class_ids_2[i] not in mean_ap_2:
      mean_ap_2[class_ids_2[i]] = 0
      num_shots_2[class_ids_2[i]] = 0.0

    mean_ap_1[class_ids[i]] += ap[i]
    mean_ap_2[class_ids_2[i]] += ap_2[i]
    num_shots_1[class_ids[i]] += 1.0
    num_shots_2[class_ids_2[i]] += 1.0

  for c in mean_ap_1:
    mean_ap_1[c] /= num_shots_1[c]
    mean_ap_2[c] /= num_shots_2[c]

  print mean_ap_1
  print mean_ap_2
  print num_shots_1
  print num_shots_2
  print '%f, %f'%(np.mean(np.array(ap)), np.mean(np.array(ap_2)))

  sys.exit(0)

  ap_diff = [-(acc_at_5[i] - acc_at_5_2[i]) for i in range(len(ap))]
  sorted_ids = sorted(range(len(ap)), key=lambda x:ap_diff[x])
# Parse the video list
  fid = open(video_list_file, 'r')
  video_list = []
  for line in fid:
    video_list.append(line[:-1])
  fid.close()

# Parse id to video list
  fid = open(id_to_video_shot_list)
  video_shot_ids = []
  for line in fid:
    line_splits = line.split(':')
    video_shot_ids.append((int(line_splits[1]), int(line_splits[2])))
  fid.close()

# Actually get the images
  for id in range(101, 500): #len(class_ids)):
    i = sorted_ids[id]
    target_frame_dir = '%s/E%03d/%s/shot%05d'%(image_dir, class_ids[i],
        video_list[video_shot_ids[i][0]], video_shot_ids[i][1])
    target_images = glob.glob('%s/*.jpeg'%target_frame_dir)
    print 'Query ---->> %d:%s'%(id,target_images[0])
    src_image = misc.imread(target_images[0])

    output_image_name = '%s/%05d_%s_%05d_E%03d.jpeg'%(output_directory, id, video_list[video_shot_ids[i][0]], video_shot_ids[i][1], class_ids[i])

    fig = plt.figure()
    
    fig1 = fig.add_subplot(1, 6, 1);
    fig1.imshow(src_image)
    fig1.axes.get_xaxis().set_visible(False)
    fig1.axes.get_yaxis().set_visible(False)
    fig1.set_title('E%d: %.2f'%(class_ids[i], ap_diff[i]))
    #fig1.set_title('query: E%d'%(class_ids[i]))

    for k in range(5):

# Original ret images
      ret_image_dir = '%s/E%03d/%s/shot%05d'%(image_dir, int(top_5_classes[i][k]),
        video_list[video_shot_ids[int(top_5_images[i][k])][0]], video_shot_ids[int(top_5_images[i][k])][1])
      #print 'Ret(fc7) ==================> : %s\n'%ret_image_dir

      ret_images = glob.glob('%s/*.jpeg'%ret_image_dir)
      ret_image = misc.imread(ret_images[0])
      fig_sub = fig.add_subplot(1, 6, 2+k);
      fig_sub.imshow(ret_image)
      fig_sub.axes.get_xaxis().set_visible(False)
      fig_sub.axes.get_yaxis().set_visible(False)
      fig_sub.set_title('fc7: E%d'%top_5_classes[i][k])
# Original ret images
      ret_image_dir = '%s/E%03d/%s/shot%05d'%(image_dir, int(top_5_classes_2[i][k]),
        video_list[video_shot_ids[int(top_5_images_2[i][k])][0]], video_shot_ids[int(top_5_images_2[i][k])][1])
      #print 'Ret ==================> : %s\n'%ret_image_dir
      ret_images = glob.glob('%s/*.jpeg'%ret_image_dir)
      ret_image = misc.imread(ret_images[0])
      fig_sub = fig.add_subplot(2, 6, 2+k);
      fig_sub.imshow(ret_image)
      fig_sub.axes.get_xaxis().set_visible(False)
      fig_sub.axes.get_yaxis().set_visible(False)
      fig_sub.set_title('our: E%d'%top_5_classes_2[i][k])
    plt.savefig(output_image_name, bbox_inches='tight')
    #break
