# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import os
import os.path as osp
import shutil
from paddlex.utils import is_pic, get_encoding


class X2VOC(object):
    def __init__(self):
        pass

    def convert(self, image_dir, txt_dir, dataset_save_dir):
        """转换。
        Args:
            image_dir (str): 图像文件存放的路径。
            txt_dir (str): 与每张图像对应的txt文件的存放路径。
            dataset_save_dir (str): 转换后数据集存放路径。
        """
        assert osp.exists(image_dir), "The image folder does not exist!"
        assert osp.exists(txt_dir), "The txt folder does not exist!"
        if not osp.exists(dataset_save_dir):
            os.makedirs(dataset_save_dir)
        # Convert the image files.
        new_image_dir = osp.join(dataset_save_dir, "JPEGImages")
        os.makedirs(new_image_dir, exist_ok=True)
        for img_name in os.listdir(image_dir):
            if is_pic(img_name):
                shutil.copyfile(
                    osp.join(image_dir, img_name),
                    osp.join(new_image_dir, img_name))
        # Convert the txt files.
        xml_dir = osp.join(dataset_save_dir, "Annotations")
        os.makedirs(xml_dir, exist_ok=True)
        self.txt2xml(new_image_dir, txt_dir, xml_dir)


class Txt2VOC(X2VOC):
    """将使用LabelMe标注的数据集转换为VOC数据集。
    """

    def __init__(self):
        pass

    def txt2xml(self, image_dir, txt_dir, xml_dir):
        import xml.dom.minidom as minidom
        i = 0
        for img_name in os.listdir(image_dir):
            img_name_part = osp.splitext(img_name)[0]
            txt_file = osp.join(txt_dir, img_name_part + ".txt")
            i += 1
            if not osp.exists(txt_file):
                os.remove(osp.join(image_dir, img_name))
                continue
            xml_doc = minidom.Document()
            root = xml_doc.createElement("annotation")
            xml_doc.appendChild(root)
            node_folder = xml_doc.createElement("folder")
            node_folder.appendChild(xml_doc.createTextNode("JPEGImages"))
            root.appendChild(node_folder)
            node_filename = xml_doc.createElement("filename")
            node_filename.appendChild(xml_doc.createTextNode(img_name))
            root.appendChild(node_filename)
            with open(txt_file, mode="r",
                      encoding=get_encoding(txt_file)) as j:
                img_h, img_w = 256, 256
                
                node_size = xml_doc.createElement("size")
                node_width = xml_doc.createElement("width")
                node_width.appendChild(xml_doc.createTextNode(str(img_w)))
                node_size.appendChild(node_width)
                node_height = xml_doc.createElement("height")
                node_height.appendChild(xml_doc.createTextNode(str(img_h)))
                node_size.appendChild(node_height)
                node_depth = xml_doc.createElement("depth")
                node_depth.appendChild(xml_doc.createTextNode(str(3)))
                node_size.appendChild(node_depth)
                root.appendChild(node_size)

                txts = j.readlines()
                for line in txts:
                    box = line.split()[1:]
                    box = [float(i) for i in box]
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    xmin = (x - w/2)*img_w
                    xmax = (x + w/2)*img_w
                    ymin = (y - h/2)*img_h
                    ymax = (y + h/2)*img_h
                    label = 'ship'
                    node_obj = xml_doc.createElement("object")
                    node_name = xml_doc.createElement("name")
                    node_name.appendChild(xml_doc.createTextNode(label))
                    node_obj.appendChild(node_name)
                    node_diff = xml_doc.createElement("difficult")
                    node_diff.appendChild(xml_doc.createTextNode(str(0)))
                    node_obj.appendChild(node_diff)
                    node_box = xml_doc.createElement("bndbox")
                    node_xmin = xml_doc.createElement("xmin")
                    node_xmin.appendChild(xml_doc.createTextNode(str(xmin)))
                    node_box.appendChild(node_xmin)
                    node_ymin = xml_doc.createElement("ymin")
                    node_ymin.appendChild(xml_doc.createTextNode(str(ymin)))
                    node_box.appendChild(node_ymin)
                    node_xmax = xml_doc.createElement("xmax")
                    node_xmax.appendChild(xml_doc.createTextNode(str(xmax)))
                    node_box.appendChild(node_xmax)
                    node_ymax = xml_doc.createElement("ymax")
                    node_ymax.appendChild(xml_doc.createTextNode(str(ymax)))
                    node_box.appendChild(node_ymax)
                    node_obj.appendChild(node_box)
                    root.appendChild(node_obj)
            with open(osp.join(xml_dir, img_name_part + ".xml"), 'w') as fxml:
                xml_doc.writexml(
                    fxml,
                    indent='\t',
                    addindent='\t',
                    newl='\n',
                    encoding="utf-8")


data_dir = 'dataset/training_dataset'
img_dir = osp.join(data_dir, 'Images')
txt_dir = osp.join(data_dir, 'txts')
dataset_save_dir = osp.join(data_dir, 'ship_detect')
txt2voc = Txt2VOC()
txt2voc.convert(img_dir, txt_dir, dataset_save_dir)
