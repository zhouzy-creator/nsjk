import random
import io
import global_config
from utils import *
import time
import sys
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element,tostring
# sys.path.append("/project/train/src_repo/tf-models/research")
# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as viz_utils
# from object_detection.utils import dataset_util

def class_text_to_int(row_label):
    if row_label == 'lifevest_person':  # 标注文件里面的标签名称
        return 1
    elif row_label == 'no_lifevest_person':
        return 2
    else:
        return 0
    
    

#IMG_LIST=["/project/train/NJSK/NJSK_lifevest_none_none_none_train_none_day_20210415_126.xml"]
IMG_LIST=get_IMG_PATHS(global_config.DATA_PATH)
found_data_list = get_crop_item(IMG_LIST,3,3)

labels={"lifevest":1,"lifevest_person":2,"no_lifevest_person":3,"no_lifevest":4}

# random.shuffle(found_data_list)
# train_data_count = int(len(found_data_list) * 4 / 5)
# print("train_data_count",train_data_count,len(found_data_list))
# train_data_list = found_data_list[0:train_data_count]
# valid_data_list = found_data_list[train_data_count:]

# print(train_data_list[0].imgname,   train_data_list[0].cropbox,
#     train_data_list[0].boxlist,
#     train_data_list[0].classlist)
    
for  i,item in enumerate( found_data_list): 
    crop=item.cropbox
    w = crop[2]-crop[0]
    h = crop[3]-crop[1]
    image = Image.open(item.imgname)
    image = image.crop(item.cropbox)
    new_name=os.path.join(global_config.TFRECORD_PATH,"image",str(i)+"_"+item.imgname.split("_")[-1:][0])
    image.save(new_name)
    print(item.cropbox,item.imgname,item.boxlist)
    root = Element("annotation")
    child1 = Element("folder")
    child1.text = "Hello world"
    child2 = Element("filename")
    child2.text=new_name
    child3 = Element("size")
    width=Element("width")
    width.text=str(w)
    height=Element("height")
    height.text=str(h)
    depth=Element("depth")
    depth.text=str(3)
    child3.append(width)
    child3.append(height)
    child3.append(depth)

    root.append(child1)
    root.append(child2)
    root.append(child3)
    
    for j, box in enumerate(item.boxlist):
        child4 = Element("object")
        ename = Element("name")
        classs=int(item.classlist[j])
        if(classs==2):
            ename.text="lifevest_person"
        else:
             ename.text="no_lifevest_person"
        print(item.classlist[j])
        child4.append(ename)

        bndbox= Element("bndbox")
        xmin=Element("xmin")
        xmin.text=(str(box[0]))
        ymin=Element("ymin")
        ymin.text=(str(box[1]))
        xmax=Element("xmax")
        xmax.text=(str(box[2]))
        ymax=Element("ymax")
        ymax.text=(str(box[3]))
        bndbox.append(xmin)
        bndbox.append(ymin)
        bndbox.append(xmax)
        bndbox.append(ymax)
        child4.append(bndbox)
        root.append(child4)
        
    tree_str = tostring(root, encoding="unicode")
    #print(tree_str)
    with open(new_name[0:-4]+".xml","w") as f:
        f.write(tree_str)

# train_file="train.record"
# valid_file="valid.record"
# write_record(os.path.join(global_config.TFRECORD_PATH,train_file),train_data_list)
# write_record(os.path.join(global_config.TFRECORD_PATH,valid_file),valid_data_list)
# with open(os.path.join(global_config.TFRECORD_PATH, 'label_map.pbtxt'), 'w') as f:
#   label_map = """
#   item {
#   id: 1
#   name: "lifevest_person"
#   },
#   item {
#   id: 2
#   name: "no_lifevest_person"
#   }
#   """
#   f.write(label_map)
