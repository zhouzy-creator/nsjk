import random
import io
import global_config
from utils import *
import time
import sys
import xml.etree.ElementTree as ET
sys.path.append("/project/train/src_repo/tf-models/research")
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import dataset_util

def class_text_to_int(row_label):
    if row_label == 'lifevest_person':  # 标注文件里面的标签名称
        return 1
    elif row_label == 'no_lifevest_person':
        return 2
    else:
        return 0
    
    


IMG_LIST=get_IMG_PATHS(global_config.DATA_PATH)
found_data_list = get_crop_item(IMG_LIST,3,3)
random.shuffle(found_data_list)
train_data_count = int(len(found_data_list) * 4 / 5)
print("train_data_count",train_data_count,len(found_data_list))
train_data_list = found_data_list[0:train_data_count]
valid_data_list = found_data_list[train_data_count:]

print(train_data_list[0].imgname,   train_data_list[0].cropbox,
    train_data_list[0].boxlist,
    train_data_list[0].classlist)



train_file="train.record"
valid_file="valid.record"
write_record(os.path.join(global_config.TFRECORD_PATH,train_file),train_data_list)
write_record(os.path.join(global_config.TFRECORD_PATH,valid_file),valid_data_list)
with open(os.path.join(global_config.TFRECORD_PATH, 'label_map.pbtxt'), 'w') as f:
  label_map = """
  item {
  id: 1
  name: "lifevest_person"
  },
  item {
  id: 2
  name: "no_lifevest_person"
  }
  """
  f.write(label_map)
