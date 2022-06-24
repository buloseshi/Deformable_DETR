#这个文件用来对数据集进行预处理，将原始的数据集分转换成可以被模型读入的VOC格式
import os
import cv2
import random
import numpy as np

from tqdm import tqdm

#这个函数用来读入数据集，返回的是文件列表
def get_datas(data_dir, endswith):
    items = [os.path.join(data_dir, item) for item in os.listdir(data_dir)]
    sub_dirs = [item for item in items if os.path.isdir(item)]
    files = [item for item in items if item.endswith(endswith)]
    for sub_dir in sub_dirs:
        _files = get_datas(sub_dir, endswith)
        files += _files
    files.sort()
    return files
#如果路径不存在，就创建该路径
def makedirs(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

#用来创建voc标签格式
def vis_anno_label(jpg_files, txt_files, data_dir, xml_dir, list_dir, split_num):
    data_list = []
    for jpg_file, txt_file in tqdm(zip(jpg_files, txt_files)):
        item_dir, item_file = os.path.split(jpg_file)

        img = cv2.imdecode(np.fromfile(jpg_file, dtype=np.uint8), -1)
        with open(txt_file, 'r', encoding='UTF-8') as f:
            line, position = [item.split(' ') for item in f.read().split('\n')]

        position = [int(item) for item in position if item]
        x1, y1, x2, y2 = position
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        #voc标签格式
        anno = f'''<annotation>
    <filename>{item_file}</filename>
    <size>
        <height>{img.shape[0]}</height>
        <width>{img.shape[1]}</width>
        <depth>3</depth>
    </size>
    <object>
        <name>Groove</name>
        <bndbox>
            <xmin>{x1}</xmin>
            <ymin>{y1}</ymin>
            <xmax>{x2}</xmax>
            <ymax>{y2}</ymax>
        </bndbox>
    </object>
</annotation>'''
        makedirs(item_dir.replace(data_dir, xml_dir))
        xml_file = jpg_file.replace(data_dir, xml_dir)[:-4] + '.xml'
        with open(xml_file, 'w', encoding='UTF-8') as f:
            f.write(anno)

        data_list.append(f'{jpg_file} {xml_file}\n')

    random.shuffle(data_list)
    with open(os.path.join(list_dir, 'train.txt'), 'w', encoding='UTF-8') as f:
        for item in data_list[:split_num]:
            f.write(item)
    with open(os.path.join(list_dir, 'val.txt'), 'w', encoding='UTF-8') as f:
        for item in data_list[split_num:]:
            f.write(item)

    with open(os.path.join(list_dir, 'label_list.txt'), 'w', encoding='UTF-8') as f:
        f.write('Groove\n')

vis_anno_label(get_datas('./dataset/img', '.jpg'),
               get_datas('./dataset/img', '.txt'),
               './dataset/img',
               './dataset/xml',
               './dataset',
               2500)