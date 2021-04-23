import sys
import os

project_root = '/project/train/cvmart'
sys.path.append(os.path.join(project_root, 'tf-models/research'))
sys.path.append(os.path.join(project_root, 'tf-models/research/slim'))

DATA_PATH="/project/train/NJSK"
TFRECORD_PATH="/project/train/cvmart/dataset"
