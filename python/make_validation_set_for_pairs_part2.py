import os

#import caffe
from caffe.proto import caffe_pb2
from caffe.proto import video_shot_sentences_pb2
import lmdb
import gflags
import random
import sys

FLAGS = gflags.FLAGS

gflags.DEFINE_string('input_db', '/data2/vigneshr/ICCV2015/data/med_event_kit_test_shot_pairs_lmdb_revised_fc6', 'input db path from which we will sample');
gflags.DEFINE_string('output_db', '/data2/vigneshr/ICCV2015/data/med_event_kit_test_shot_pairs_lmdb_revised_fc6_sampled_dist_5', 'output db path to which we will write');
gflags.DEFINE_integer('num_videos', 600, 'number of videos to sample');
gflags.DEFINE_integer('max_num_videos', 650, 'maximum number of videos in dataset');
gflags.DEFINE_integer('min_shot_dist', 5, 'minimum shot distance between a pair of shots');
gflags.DEFINE_integer('max_shot_pairs', 10, 'maximum number of shot pairs per video');
gflags.DEFINE_boolean('only_one_pair_per_shot', False, 'only include one pair per shot from a video');
def main(argv):
  try:
    argv = FLAGS(argv)  # parse flags
  except gflags.FlagsError, e:
    print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS)
    sys.exit(1)
  env_read  = lmdb.open(FLAGS.input_db, create=False)
  env_write = lmdb.open(FLAGS.output_db, map_size=504857600, create=True)
  num_added = 0

  video_id_samples = random.sample(range(FLAGS.max_num_videos), FLAGS.num_videos)

  with env_read.begin(write=False) as txn, env_write.begin(write=True) as txn_write:
    for video_id in video_id_samples:
      shot_id = 1
      if not txn.get('%d:0:1'%(video_id)):
        print 'Could not get: %d'%(video_id)
      while txn.get('%d:0:%d'%(video_id, shot_id)):
        shot_id = shot_id + 1
      if shot_id < FLAGS.min_shot_dist:
        continue
      shot_id_pairs = []
      for s in range(shot_id):
        for r in range(s+FLAGS.min_shot_dist, shot_id):
          shot_id_pairs.append((s,r))

      if len(shot_id_pairs) < FLAGS.max_shot_pairs:
        sample_shot_pairs = shot_id_pairs 
      else:
        sample_shot_pairs = random.sample(shot_id_pairs, FLAGS.max_shot_pairs)

      for shot_id_1, shot_id_2 in sample_shot_pairs:
        if shot_id_1 == shot_id_2:
          continue
        b = txn.get('%d:%d:%d'%(video_id, shot_id_1, shot_id_2))
        if not b:
          print ('Could not find: %d:%d:%d'%(video_id, shot_id_1, shot_id_2))
          continue
        v = video_shot_sentences_pb2.VideoShotWindow()
        v.ParseFromString(b)
        txn_write.put('%d:%d:%d'%(video_id, shot_id_1, shot_id_2), b, overwrite=False)
        print ('Adding(%d): %d:%d'%(num_added, video_id, shot_id_1))
        num_added = num_added + 1
        if FLAGS.only_one_pair_per_shot:
          break
        #shot_id = shot_id + 1
        #print '----------------> %d: Added %d shots'%(v.video_id, shot_id)

if __name__ == '__main__':
  main(sys.argv)
