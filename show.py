import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np

colors = ["#112233", "#ce0000", "#FF0080", "#5B5B00", "#796400"]
labels = {"lifevest": 1, "lifevest_person": 2, "no_lifevest_person": 3, "no_lifevest": 4}
DATA_SETPATH = "../dataset/au"


def get_XML_PATHS(path_str):
    XML_PATHS = []
    paths = os.listdir(path_str)
    for p in paths:
        if (p[-4:] == ".xml"):
            XML_PATHS.append(os.path.join(path_str, p))
    return XML_PATHS


def get_IMG_PATHS(path_str):
    IMG_PATHS = []
    paths = os.listdir(path_str)
    for p in paths:
        lowstr = p[-4:].lower()
        if (lowstr == ".jpg" or lowstr == ".png"):
            IMG_PATHS.append(os.path.join(path_str, p))
    return IMG_PATHS


def plt_bboxes(img, boxs, classes):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """

    plt.imshow(img)

    for i, box in enumerate(boxs):
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[2])
        ymax = int(box[3])
        cls_id = classes[i]
        # print(colors[cls_id])
        rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                             ymax - ymin, fill=False,
                             edgecolor=colors[cls_id],
                             linewidth=2)
        class_name = str(cls_id)
        plt.gca().add_patch(rect)
        plt.gca().text(xmin, ymin - 2,
                       '{:s} '.format(class_name),
                       bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                       fontsize=12, color='white')


def parse_xml(name):
    tree = ET.parse(name)
    root = tree.getroot()
    boxs = []
    classes = []
    print("----------------",len(root.findall('object')))
    for member in root.findall('object'):
        name = member[0].text
        #print(name[-6:],member[0].text)
        if (name[-6:] == "person"):
            xmin = int(float(member[1][0].text))
            ymin = int(float(member[1][1].text))
            xmax = int(float(member[1][2].text))
            ymax = int(float(member[1][3].text))
            print(xmin, ymin, xmax, ymax)
            print(member[0].text)
            cls = labels[member[0].text]
            # print(cls)
            boxs.append((xmin, ymin, xmax, ymax))
            classes.append(cls)
    return boxs, classes


IMAGE_PATHS = get_IMG_PATHS(DATA_SETPATH)

for name in IMAGE_PATHS:
    xmlname = name[0:-4] + ".xml"
    imgname = name[0:-4] + ".jpg"
    print(xmlname)
    h = w = 3
    boxlist, classes = parse_xml(xmlname)
    fig=plt.figure(figsize=(20, 20))
    im = Image.open(name)
    # im,imglist=image_cut(imgname,h,w)
    plt_bboxes(np.array(im), boxlist, classes)
    p,fname=os.path.split(xmlname)
    plt.title(fname)
    plt.show()

# #crop,boxlist,classes,box_w,box_h=box_cut(xmlname,h,w)
# print(len(imglist),len(boxlist),len(classes))
# for i,im in enumerate(imglist):
#   plt_bboxes(np.array(im),boxlist[i],classes[i])
#   plt.show()