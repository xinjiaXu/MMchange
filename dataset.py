import cv2
import os
import numpy
import torch.utils.data
import json
import torch
class Dataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''

    def __init__(self, dataset, file_root='data/', transform=None,
                 image_dir_a='A', image_dir_b='B', label_dir='label', list_path=None):
        """
        dataset: dataset name, e.g. NJU2K_NLPR_train
        file_root: root of data_path, e.g. ./data/
        """
        file_root = os.path.abspath(file_root)
        split_root = os.path.join(file_root, dataset)
        if list_path is None:
            candidates = [
                os.path.join(split_root, 'list', dataset + '.txt'),
                os.path.join(file_root, 'list', dataset + '.txt'),
                os.path.join(file_root, dataset + '.txt'),
                os.path.join(os.path.dirname(os.path.normpath(file_root)), dataset + '.txt'),
            ]
            list_path = next((path for path in candidates if os.path.isfile(path)), None)
        if list_path:
            with open(list_path, encoding='utf-8') as name_file:
                self.file_list = [name.strip() for name in name_file if name.strip()]
        else:
            label_root = os.path.join(split_root, label_dir)
            self.file_list = sorted(name for name in os.listdir(label_root) if not name.startswith('.'))

        self.pre_images = [os.path.join(split_root, image_dir_a, x) for x in self.file_list]
        self.post_images = [os.path.join(split_root, image_dir_b, x) for x in self.file_list]
        self.gts = [os.path.join(split_root, label_dir, x) for x in self.file_list]
        self.transform = transform

    def __len__(self):
        return len(self.pre_images)

    def __getitem__(self, idx):
        pre_image_name = self.pre_images[idx]
        label_name = self.gts[idx]
        post_image_name = self.post_images[idx]
        pre_image = cv2.imread(pre_image_name)
        label = cv2.imread(label_name, 0)
        post_image = cv2.imread(post_image_name)
        img = numpy.concatenate((pre_image, post_image), axis=2)
        # if self.transform:
        #     [pre_image, label, post_image] = self.transform(pre_image, label, post_image)
        #
        # return pre_image, label, post_image

        if self.transform:
            [img, label] = self.transform(img, label)
        # # 读取json文件
        # with open('/media/WD_2T/ZYJ/315_cd_data/LEVIR_256/256/total.json', 'r', encoding='utf-8') as file:
        #     data = json.load(file)
        # dictionary = dict(data)
        # output_text_A = dictionary.get(pre_image_name)
        # output_text_B = dictionary.get(post_image_name)
        # model_text_decode, preprocess = clip.load("ViT-B/32", device=device)
        # text_A = clip.tokenize(output_text_A).to(device)
        # text_B = clip.tokenize(output_text_B).to(device)
        # text_A_features = model_text_decode.encode_text(text_A)
        # text_B_features = model_text_decode.encode_text(text_B)

        return img, label,pre_image_name,post_image_name

    def get_img_info(self, idx):
        img = cv2.imread(self.pre_images[idx])
        return {"height": img.shape[0], "width": img.shape[1]}
