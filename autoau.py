import numpy as np
import inspect
from PIL import Image, ImageOps, ImageEnhance
import math
import random
import xml.etree.ElementTree as ET
import global_config

def translate_bbox(image, bboxes, pixels, replace, shift_horizontal):
    """Equivalent of PIL Translate in X/Y dimension that shifts image and bbox.
        Args:
          image: 3D uint8 Tensor.
          bboxes: 2D Tensor that is a list of the bboxes in the image. Each bbox
            has 4 elements (min_x, min_y, max_x, max_y) of type float with values
            between [0, 1].
          pixels: An int. How many pixels to shift the image and bboxes
          replace: A one or three value 1D tensor to fill empty pixels.
          shift_horizontal: Boolean. If true then shift in X dimension else shift in
            Y dimension.
        Returns:
          A tuple containing a 3D uint8 Tensor that will be the result of translating
          image by pixels. The second element of the tuple is bboxes, where now
          the coordinates will be shifted to reflect the shifted image.
        """
    if shift_horizontal:
        image = translate_x(image, pixels, replace)
    else:
        image = translate_y(image, pixels, replace)

    # Convert bbox coordinates to pixel values.
    image_height = image.shape[0]
    image_width = image.shape[1]
    # pylint:disable=g-long-lambda
    wrapped_shift_bbox = lambda bbox: _shift_bbox(bbox, image_height, image_width, pixels, shift_horizontal)
    # 格式正确
    # pylint:enable=g-long-lambda
    bboxes = np.array([box for box in list(map(wrapped_shift_bbox, bboxes)) if box is not None])
    return image, bboxes


def distort_image_with_autoaugment(image, bboxes, augmentation_name):
    """
        image: 输入图片需要是RGB图，且是0-255的整数，具体样式为：[height, width, 3]
            0 ---------> x
            |
            |
            |
            v
            y
        bboxes: [xmin, ymin, xmax, ymax]
        augmentation_name: 选择v0-3中的一个策略，等概率的选择策略中的一个sub_policy执行。
                           或是选择test，将特定数据增强算法放在里面
        添加新算法，除函数本身定义外，需在NAME_TO_FUNC、level_to_arg中对应添加。
    """
    available_policies = {'v0': policy_v0, 'v1': policy_v1, 'v2': policy_v2,
                          'v3': policy_v3, 'v4': policy_v4, 'test': policy_vtest,
                          'custom': policy_custom}
    if augmentation_name not in available_policies:
        raise ValueError('Invalid augmentation_name: {}'.format(augmentation_name))

    policy = available_policies[augmentation_name]()

    augmentation_hparams = {
        "cutout_max_pad_fraction": 0.75,
        "cutout_bbox_replace_with_mean": False,
        "cutout_const": 100,
        "translate_const": 250,
        "cutout_bbox_const": 50,
        "translate_bbox_const": 120}

    return build_and_apply_nas_policy(policy, image, bboxes, augmentation_hparams)







class AutoAugmenter(object):
    """Applies the AutoAugment policy to `image` and `bboxes`.
    Args:
      image: `Tensor` of shape [height, width, 3] representing an image.
      bboxes: `Tensor` of shape [N, 4] representing ground truth boxes that are
        normalized between [0, 1].
      augmentation_name: The name of the AutoAugment policy to use. The available
        options are `v0`, `v1`, `v2`, `v3` and `test`. `v0` is the policy used for
        all of the results in the paper and was found to achieve the best results
        on the COCO dataset. `v1`, `v2` and `v3` are additional good policies
        found on the COCO dataset that have slight variation in what operations
        were used during the search procedure along with how many operations are
        applied in parallel to a single image (2 vs 3).
    Returns:
      A tuple containing the augmented versions of `image` and `bboxes`.
    """

    def __init__(self, augmentation_name='v4'):
        self.augmentation_name = augmentation_name

    def normalizer(self, image, annots):
        h, w = image.shape[0], image.shape[1]
        ratio = np.array([w, h, w, h], dtype=int)
        annots[:, :4] = annots[:, :4] / ratio
        return annots

    def unnormalizer(self, image, annots):
        h, w = image.shape[0], image.shape[1]

        ratio = np.array([w, h, w, h], dtype=int)
        annots[:, :4] = annots[:, :4] * ratio

        return annots.astype(np.float32)
    def parse_xml(self,xmlname):
        tree = ET.parse(xmlname)
        root = tree.getroot()
        boxs=[]
        classes=[]
        for member in root.findall('object'):
            box=[]
            box.append( int(member[1][0].text))
            box.append(int(member[1][1].text))
            box.append(int(member[1][2].text))
            box.append(int(member[1][3].text))
            boxs.append(box)
            classes.append(member[0].text)
        return boxs,classes

    def __call__(self, sample,image_name):
        #image_name, annots = sample['img'], sample['annot']
        # annots = self.normalizer(image, annots)
        # bboxes = annots[:, 0:4]
        # image = np.uint8(image * 255)
        bboxes ,classes= self.parse_xml(sample)
        image = np.array(Image.open(image_name))
        print(bboxes,classes)
        image, bboxes = distort_image_with_autoaugment(image, bboxes, self.augmentation_name)
        #
        # annots[:, 0:4] = bboxes
        # annots = self.unnormalizer(image, annots)
        # image = image.astype(np.float32) / 255.0
        print(len(bboxes),bboxes)
       # sample = {'img': image, 'annot': annots}
        return sample



sample=os.path.join(global_config.TFRECORD_PATH,"image","0_1.xml")
imgname=os.path.join(global_config.TFRECORD_PATH,"image","0_1.jpg")
au=AutoAugmenter('test')
print(au(sample,imgname))