import os

import caffe
from caffe.proto import caffe_pb2
from caffe.proto import video_shot_sentences_pb2
import lmdb
import gflags

import sys

FLAGS = gflags.FLAGS

gflags.DEFINE_string('input_db', '/data2/vigneshr/ICCV2015/data/med_event_kit_train_dev_shot_pairs_lmdb_revised', 'input db path from which we will sample');
gflags.DEFINE_string('output_db', '/data2/vigneshr/ICCV2015/data/med_event_kit_train_dev_shot_pairs_lmdb_revised_validation', 'output db path to which we will write');
gflags.DEFINE_integer('num_videos', 30, 'number of videos to sample');


def main(argv):
  try:
    argv = FLAGS(argv)  # parse flags
  except gflags.FlagsError, e:
    print '%s\\nUsage: %s ARGS\\n%s' % (e, sys.argv[0], FLAGS)
    sys.exit(1)
  env_read  = lmdb.open(FLAGS.input_db, create=False)
  env_write = lmdb.open(FLAGS.output_db, map_size=50485760, create=True)
  
  with env_read.begin(write=False) as txn, env_write.begin(write=True) as txn_write:
    for video_id in range(FLAGS.num_videos):
      shot_id = 0
      while txn.get('%d:%d'%(video_id, shot_id)):
        b = txn.get('%d:%d'%(video_id, shot_id))
        v = video_shot_sentences_pb2.VideoShotWindow()
        v.ParseFromString(b)
        txn_write.put('%d:%d'%(video_id, shot_id), b, dupdata=True, overwrite=False)
        print ('Adding: %d:%d'%(video_id, shot_id))
        shot_id = shot_id + 1
        print '----------------> %d: Added %d shots'%(v.video_id, shot_id)

if __name__ == '__main__':
  main(sys.argv)
