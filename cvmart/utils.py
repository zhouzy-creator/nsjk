import io
import xml.etree.ElementTree as ET
from PIL import Image
import tensorflow as tf
import os
import global_config
import sys
sys.path.append("/project/train/src_repo/tf-models/research")
from object_detection.utils import dataset_util
imgcount=0
labels={"lifevest":1,"lifevest_person":2,"no_lifevest_person":3,"no_lifevest":4}
class CropItem:
  def __init__(self, name):
    self.imgname = name
  def setdata(self,cropbox,boxlist,classlist,box_w,box_h):
    self.cropbox=cropbox
    self.boxlist=boxlist
    self.classlist=classlist
    self.box_w= box_w
    self.box_h=box_h 
    
##############

def get_XML_PATHS(path_str):
  XML_PATHS=[]
  paths = os.listdir(path_str)
  for p in paths:
    if(p[-4:]==".xml"):
      XML_PATHS.append(os.path.join(path_str,p))
  return XML_PATHS
def get_IMG_PATHS(path_str):
  IMG_PATHS=[]
  paths = os.listdir(path_str)
  for p in paths:
    lowstr=p[-4:].lower()
    if(lowstr==".jpg" or lowstr==".png"):
      IMG_PATHS.append(os.path.join(path_str,p))
  return IMG_PATHS


def create_tf_example(item):
    global imgcount
    name= item.imgname
    imgname= name[0:-4]+".jpg"
    #with tf.io.gfile.GFile(imgname, 'rb') as fid:
    #    encoded_jpg = fid.read()
    #encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(name)
    image = image.crop(item.cropbox)
    imgcount=imgcount+1

    new_name=os.path.join(global_config.TFRECORD_PATH,"image",str(imgcount)+".jpg")
    image.save(new_name)
    with tf.io.gfile.GFile(new_name, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)

    width  =item.box_w
    height = item.box_h
    filename = new_name.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in enumerate(item.boxlist):

        xmins.append(row[0] / width)
        xmaxs.append(row[2] / width)
        ymins.append(row[1] / height)
        ymaxs.append(row[3] / height)
        img_class =item.classlist[index]
        ## labels={"lifevest":1,"lifevest_person":2,"no_lifevest_person":3,"no_lifevest":4}
        ## trans to  lifevest_person":1,"no_lifevest_person":2
        if img_class==2:
          img_class =1
          classes_str="lifevest_person"
        if img_class==3:
          img_class=2
          classes_str="no_lifevest_person"
        classes_text.append(classes_str.encode('utf8'))
        classes.append(img_class)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def box_cut(name,hsplit,wsplit):
    tree = ET.parse(name)
    root = tree.getroot()
    boxs=[]
    classes=[]
    size =root.find('size')
    iw=int(size[0].text)
    ih=int(size[1].text)
    box_w = int(iw/wsplit)
    box_h = int(ih/hsplit)
    cropbox=[]
    for i in range(wsplit):
        for j in range(hsplit):
            box = (box_w * i, box_h * j, box_w * (i + 1), box_h * (j + 1))

            subbox=[]
            subcls=[]
            offx=0
            offy=0
            for member in root.findall('object'):
                name = member[0].text
                if(name[-6:]=="person"):
                    xmin  = int(float(member[1][0].text))
                    ymin  = int(float(member[1][1].text))
                    xmax  = int(float(member[1][2].text)) 
                    ymax  = int(float(member[1][3].text))

                    x1 = max(xmin, box[0])  # 得到左下顶点的横坐标
                    y1 = max(ymin, box[1])  # 得到左下顶点的纵坐标
                    x2 = min(xmax, box[2])  # 得到右上顶点的横坐标
                    y2 = min(ymax, box[3])  # 得到右上顶点的纵坐标

                    # 计算相交矩形的面积
                    w = x2 - x1
                    h = y2 - y1
                    if w <=0 or h <= 0:
                        continue

                    if(xmin<box[0]):
                        if(offx>0 ):
                            print("xxxxxxxxxxx",name,i,j)
                        offx=min(xmin-box[0] ,  offx)

                    if(ymin<box[1]):
                        if(offy>0 ):
                            print("xxxxxxxxxxx",name,i,j)
                        offy=min(ymin-box[1],offy)

                    if(xmax>box[2]): 
                        if(offx<0 ):
                            print("xxxxxxxxxxx",name,i,j)
                        offx=max(xmax-box[2] ,  offx)
                        
                    if( ymax>box[3]):
                        if(offy<0 ):
                            print("xxxxxxxxxxx",name,i,j)
                        offy=max(ymax-box[3],offy)

            #print("offset",offx,offy)    
            #print( x1,y1,x2,y2,box[0],box[1])
            #print( x1-box[0],y1-box[1],x2-box[0],y2-box[1])
            #print( xmin-box[0],ymin-box[1],xmax-box[0],ymax-box[1])
                    subbox.append([ xmin-box[0],ymin-box[1],xmax-box[0],ymax-box[1]])
                    subcls.append(labels[member[0].text])
            if(offx!=0 or offy!=0):
                    #print("box",box)
                box=( box[0]+offx,box[1]+offy,box[2]+offx,box[3]+offy)
                    #print("change",box)
                anobox=[]
                for subb in subbox:
                    subb=( subb[0]-offx,subb[1]-offy,subb[2]-offx,subb[3]-offy)
                    anobox.append(subb)
                subbox = anobox
            #print("change2",subbox)
            #print("append",i,j,hsplit,wsplit,len(subbox),len(subcls))
            cropbox.append(box)
            boxs.append(subbox)
            classes.append(subcls)
    return cropbox,boxs,classes,box_w,box_h

def get_crop_item(xml_list,h,w):
  data_list=[]
  for name in xml_list:
    xml= name[0:-4]+".xml"
    boxs,boxlist,classlist,box_w,box_h=box_cut(xml,h,w)
    for i,box in enumerate(boxs):
      if(len(boxlist[i])>0):
        item= CropItem(name)
        item.setdata(box,boxlist[i],classlist[i],box_w,box_h)
        data_list.append(item)  
  return data_list  
  #labels={"lifevest":1,"lifevest_person":2,"no_lifevest_person":3,"no_lifevest":4}

def write_record(output_path, item_list):
    writer = tf.io.TFRecordWriter(output_path)
    for  item in item_list:
        tf_example = create_tf_example(item)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecords: {}'.format(output_path))
