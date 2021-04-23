import cv2
import numpy as np
import os

src_path = "cut_1/"
save_path = "mixup_1/"
dst = cv2.imread("beijing1.jpg")
a = dst.shape
H = a[0]
W = a[1]
print("H", H)
print("W", W)
imagelist = os.listdir(src_path)
print("222222", len(imagelist))

centers = ((600, 600), (700, 500), (800, 300), (295, 600), (300, 450))
for center in centers:
    for image in imagelist:
        # print("11111111",image)
        image_pre, ext = os.path.splitext(image)
        img_file = src_path + image
        print("333333", img_file)
        src_img = cv2.imread(img_file)
        h = src_img.shape[0]
        w = src_img.shape[1]

        # 融合的图片尺寸过大时,按比例压缩,不改变宽高比
        if h + center[1] > H or w + center[0] > W:
            print("aaaaaa")
            # src_img = cv2.resize(src_img, (int(h/1.5), int(w/1.5)))
            src_img = cv2.resize(src_img, (0, 0), fx=0.75, fy=0.75, interpolation=cv2.INTER_NEAREST)
            h = src_img.shape[0]
            w = src_img.shape[1]
            if h + center[1] > H or w + center[0] > W:
                print("bbbbbb")
                src_img = cv2.resize(src_img, (0, 0), fx=0.75, fy=0.75, interpolation=cv2.INTER_NEAREST)
                h = src_img.shape[0]
                w = src_img.shape[1]
                if h + center[1] > H or w + center[0] > W:
                    print("ccccc")
                    src_img = cv2.resize(src_img, (0, 0), fx=0.75, fy=0.75, interpolation=cv2.INTER_NEAREST)
                    h = src_img.shape[0]
                    w = src_img.shape[1]
                    if h + center[1] > H or w + center[0] > W:
                        print("ddddd")
                        src_img = cv2.resize(src_img, (0, 0), fx=0.75, fy=0.75, interpolation=cv2.INTER_NEAREST)
                        h = src_img.shape[0]
                        w = src_img.shape[1]
                        if h + center[1] > H or w + center[0] > W:
                            print("eeeee")
                            src_img = cv2.resize(src_img, (0, 0), fx=0.75, fy=0.75, interpolation=cv2.INTER_NEAREST)
                            h = src_img.shape[0]
                            w = src_img.shape[1]

        src_mask = 255 * np.ones(src_img.shape, src_img.dtype)
        normal_clone = cv2.seamlessClone(src_img, dst, src_mask, center, cv2.NORMAL_CLONE)
        cv2.imwrite(save_path + image_pre + str(int(center[0] / 100)) + ".jpg", normal_clone)
