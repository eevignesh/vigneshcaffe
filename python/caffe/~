
import leveldb
import unicodedata

l = leveldb.LevelDB('/afs/cs.stanford.edu/u/vigneshr/scratch2/ICCV2015/data/_temp/HVC060530_leveldb')
b = l.Get('-1:0')
from proto import video_shot_sentences_pb2
v = video_shot_sentences_pb2.VideoShotWindow()

v.ParseFromString(b)
print(v.video_id)
print(v.video_name)
print (len(v.target_shot_word.float_data))
print (len(v.context_shot_words))
print (len(v.context_shot_words[0].float_data))
