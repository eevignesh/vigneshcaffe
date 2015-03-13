import os

import caffe
from caffe.proto import caffe_pb2
from caffe.proto import video_shot_sentences_pb2
import lmdb
import gflags
import random
import sys

FLAGS = gflags.FLAGS

#gflags.DEFINE_string('input_db', '/data2/vigneshr/ICCV2015/data/med_event_kit_test_shot_c3_lmdb_revised_fc5', 'input db path from which we will sample');
#gflags.DEFINE_string('output_db', '/data2/vigneshr/ICCV2015/data/med_event_kit_test_shot_c3_lmdb_revised_validation_fc5', 'output db path to which we will write');
gflags.DEFINE_string('input_db', '/data2/vigneshr/ICCV2015/data/med_only_new_test_shot_c3_lmdb_fc6', 'input db path from which we will sample');
gflags.DEFINE_string('output_db', '/data2/vigneshr/ICCV2015/data/med_only_new_test_shot_c3_lmdb_fc6_validation', 'output db path to which we will write');

gflags.DEFINE_integer('max_num_videos', 7324, 'maximum number of videos');
gflags.DEFINE_integer('num_videos', 4000, 'number of videos to sample');
gflags.DEFINE_integer('max_shots', 3, 'maximum number of shots per video');

def main(argv):
  try:
    argv = FLAGS(argv)  # parse flags
  except gflags.FlagsError, e:
    print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS)
    sys.exit(1)
  env_read  = lmdb.open(FLAGS.input_db, create=False)
  env_write = lmdb.open(FLAGS.output_db, map_size=504857600, create=True)
  num_added = 0
  with env_read.begin(write=False) as txn, env_write.begin(write=True) as txn_write:
    video_id_samples = random.sample(range(FLAGS.max_num_videos), FLAGS.num_videos)
    for video_id in video_id_samples:
      shot_id = 0
      while txn.get('%d:%d'%(video_id, shot_id)):
        shot_id = shot_id + 1
      if shot_id < FLAGS.max_shots:
        sample_shots = range(shot_id)
      else:
        sample_shots = random.sample(range(shot_id), FLAGS.max_shots)
      for shot_id in sample_shots:
        b = txn.get('%d:%d'%(video_id, shot_id))
        v = video_shot_sentences_pb2.VideoShotWindow()
        v.ParseFromString(b)
        txn_write.put('%d:%d'%(video_id, shot_id), b, overwrite=False)
        print ('Adding(%d): %d:%d'%(num_added, video_id, shot_id))
        num_added = num_added + 1
        #shot_id = shot_id + 1
        #print '----------------> %d: Added %d shots'%(v.video_id, shot_id)

if __name__ == '__main__':
  main(sys.argv)
