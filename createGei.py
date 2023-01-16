import os
import cv2
import numpy as np
from skimage import img_as_ubyte

import argparse
import imageio

START = "START"
FINISH = "FINISH"
WARNING = "WARNING"
FAIL = "FAIL"


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--input_path', default='', type=str,
                    help='Root path of raw dataset.')
parser.add_argument('--output_path', default='', type=str,
                    help='Root path for output.')

opt = parser.parse_args()

INPUT_PATH = opt.input_path
OUTPUT_PATH = opt.output_path

def cut_pickle(seq_info, test= False):
    seq_path = os.path.join(INPUT_PATH, *seq_info)
    if test:
        out_dir = os.path.join(OUTPUT_PATH, seq_info[2], 'test')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_dir = os.path.join(OUTPUT_PATH, seq_info[2], 'test', 'gei' + seq_info[1][-2:]  + seq_info[0] + '.png')
    else:
        out_dir = os.path.join(OUTPUT_PATH, seq_info[2])
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_dir = os.path.join(OUTPUT_PATH, seq_info[2], 'gei' + seq_info[1][-2:] + seq_info[0] + '.png')
    print(out_dir)
    frame_list = os.listdir(seq_path)
    frame_list.sort()
    tmp = np.zeros((128, 88))
    for _frame_name in frame_list:
        frame_path = os.path.join(seq_path, _frame_name)
        img = cv2.imread(frame_path)[:, :, 0]
        if img is not None:
            # Save the img
            tmp = np.sum([img, tmp], axis=0)
    if len(frame_list) > 0:
        tmp = tmp / len(frame_list)
        tmp = tmp.astype(np.uint8)
        imageio.imwrite(out_dir, tmp)



id_list = os.listdir(INPUT_PATH)
id_list.sort()

for _id in id_list:
    seq_type = os.listdir(os.path.join(INPUT_PATH, _id))
    seq_type.sort()
    for i,_seq_type in enumerate(seq_type):
        view = os.listdir(os.path.join(INPUT_PATH, _id, _seq_type))
        view.sort()
        for _view in view:
            seq_info = [_id, _seq_type, _view]
            print(i)
            if i > 3:
                cut_pickle(seq_info, test= True)
            else:
                cut_pickle(seq_info)