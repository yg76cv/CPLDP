import os
import sys

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
from PIL import Image,ImageOps
import glob
import random

import clip

random.seed(1143)







def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

def augment(img_in, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            info_aug['trans'] = True

    return img_in, img_tar, info_aug



device = "cuda" if torch.cuda.is_available() else "cpu"



def populate_train_list(underwater_images_path):

	image_list_underwater = glob.glob(underwater_images_path + "*")
	train_list = sorted(image_list_underwater)
	#print(train_list)
	random.shuffle(train_list)
	label_list=[]
	for image_path in train_list:
		label_path=image_path.replace('UIEB_890raw','UIEB_890')
		#label_path = image_path.replace('raw-890-s', 'reference-890')
		label_list.append(label_path)

	return train_list,label_list

def _read_id_list(*candidate_paths):
    """按顺序尝试读取 txt，返回去掉空行/换行后的列表；都不存在就返回空列表."""
    for p in candidate_paths:
        if p and os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
    return []
	

class underwater_loader(data.Dataset):

	def __init__(self, underwater_images_path,reference_path,detype,data_path):
		self.green_ids = []
		self.blue_ids = []
		self.gb_ids = []
		self.dark_ids = []
		self.haze_ids = []

		# ---- 这几个 dict 列表用于“真正采样”（含扩充系数）----
		self.gr_dict_ids = []
		self.bl_dict_ids = []
		self.gb_dict_ids = []
		self.dark_dict_ids = []
		self.haze_dict_ids = []

		# raw id（用于 de_type 映射）
		self.raw_green_ids = []
		self.raw_blue_ids = []
		self.raw_gb_ids = []

		self.de_type = detype
		print(self.de_type)

		self.underwater_path=underwater_images_path
		print(self.underwater_path)

		self.reference_path = reference_path
		self.prompt_path=data_path
		self._init_ids()
		self._merge_ids()




		self.size = 256
		# ============ 核心改动：不用 glob 的 train_list，而是用 sample_ids ============
		if hasattr(self, "sample_ids") and len(self.sample_ids) > 0:
			self.train_list, self.label_list = self._build_train_list_from_sample_ids()
		else:
			# 兜底
			self.train_list, self.label_list = populate_train_list(underwater_images_path)

		# 创建文件名到退化类型的映射
		self.file_to_de_type = self._create_file_to_de_type_map()

		self.data_list = self.train_list
		print("Total training examples (Underwater):", len(self.train_list))


	def _init_ids(self):
		if 'blue' in self.de_type or 'green' in self.de_type or 'green_blue' in self.de_type:
			self._init_color_ids()
		if 'dark' in self.de_type:
			self._init_dark_ids()
		if 'haze' in self.de_type:
			self._init_haze_ids()


	def _init_color_ids(self):
		refgreen_file = os.path.join(self.prompt_path, "green.txt")
		refblue_file = os.path.join(self.prompt_path, "blue.txt")
		refgb_file = os.path.join(self.prompt_path, "green-blue.txt")

		self.raw_green_ids = _read_id_list(refgreen_file)
		self.raw_blue_ids = _read_id_list(refblue_file)
		self.raw_gb_ids = _read_id_list(refgb_file)

		# 创建 dict 列表（用于采样），并按原逻辑扩充
		if 'green' in self.de_type:
			self.gr_dict_ids = [{"clean_id": x, "de_type": 0} for x in self.raw_green_ids]
			self.gr_dict_ids = self.gr_dict_ids * 3
			random.shuffle(self.gr_dict_ids)
			print("gg Color Ids : {}".format(len(self.gr_dict_ids)))


		if 'blue' in self.de_type:
			self.bl_dict_ids = [{"clean_id": x, "de_type": 1} for x in self.raw_blue_ids]
			self.bl_dict_ids = self.bl_dict_ids * 3
			random.shuffle(self.bl_dict_ids)
			print("bb Color Ids : {}".format(len(self.bl_dict_ids)))

		if 'green_blue' in self.de_type:
			self.gb_dict_ids = [{"clean_id": x, "de_type": 2} for x in self.raw_gb_ids]
			self.gb_dict_ids = self.gb_dict_ids * 6
			random.shuffle(self.gb_dict_ids)
			print("gb Color Ids : {}".format(len(self.gb_dict_ids)))

		self.num_clean = len(self.gr_dict_ids) + len(self.bl_dict_ids) + len(self.gb_dict_ids)
		print("Total Color Ids : {}".format(self.num_clean))

	def _init_dark_ids(self):
		dark_file = os.path.join(self.prompt_path, "Dark.txt")
		self.dark_ids = _read_id_list(dark_file)

		# 与 stage2 映射一致：dark = 3
		self.dark_dict_ids = [{"clean_id": x, "de_type": 3} for x in self.dark_ids]
		self.num_dark = len(self.dark_dict_ids)
		print("Total Dark Ids : {}".format(self.num_dark))

	def _init_haze_ids(self):
		haze_file = os.path.join(self.prompt_path, "haze.txt")
		self.haze_ids = _read_id_list(haze_file)

		# 原逻辑：haze 扩充 *3
		self.haze_ids = self.haze_ids * 3
		random.shuffle(self.haze_ids)

		# 与 stage2 映射一致：haze = 4
		self.haze_dict_ids = [{"clean_id": x, "de_type": 4} for x in self.haze_ids]
		self.num_haze = len(self.haze_dict_ids)
		print("Total Haze Ids : {}".format(self.num_haze))

	def _merge_ids(self):
		self.sample_ids = []
		if "green" in self.de_type:
			self.sample_ids += self.gr_dict_ids
		if "blue" in self.de_type:
			self.sample_ids += self.bl_dict_ids
		if "green_blue" in self.de_type:
			self.sample_ids += self.gb_dict_ids
		if "dark" in self.de_type:
			self.sample_ids += self.dark_dict_ids
		if "haze" in self.de_type:
			self.sample_ids += self.haze_dict_ids

		print(len(self.sample_ids))
		random.shuffle(self.sample_ids)

	def _create_file_to_de_type_map(self):
		file_to_de_type = {}

		# 注意：用 basename 做 key，避免 txt 里出现路径片段导致匹配失败
		for file_id in getattr(self, "raw_green_ids", []):
			file_to_de_type[os.path.basename(file_id)] = 0
		for file_id in getattr(self, "raw_blue_ids", []):
			file_to_de_type[os.path.basename(file_id)] = 1
		for file_id in getattr(self, "raw_gb_ids", []):
			file_to_de_type[os.path.basename(file_id)] = 2
		for file_id in getattr(self, "dark_ids", []):
			file_to_de_type[os.path.basename(file_id)] = 3
		for file_id in getattr(self, "haze_ids", []):
			file_to_de_type[os.path.basename(file_id)] = 4

		return file_to_de_type

	def _build_train_list_from_sample_ids(self):
		train_list, label_list = [], []

		for s in self.sample_ids:
			clean_id = s["clean_id"]  # txt 里每行的 id（通常是文件名）
			base = os.path.basename(clean_id)

			# underwater path
			if os.path.isabs(clean_id) or clean_id.startswith(self.underwater_path):
				under_path = clean_id
			else:
				under_path = os.path.join(self.underwater_path, clean_id)

			# reference path：默认同名配对
			ref_path = os.path.join(self.reference_path, base)

			train_list.append(under_path)
			label_list.append(ref_path)

		return train_list, label_list

	def __getitem__(self, index):
		data_underwater_path = self.train_list[index]
		data_label_path = self.label_list[index]

		# 提取文件名（不带扩展名）
		file_name = os.path.basename(data_underwater_path)

		# 获取退化类型
		deg_type = self.file_to_de_type.get(file_name, 3)  # 默认为dark类型


		data_underwater = Image.open(data_underwater_path).convert("RGB")
		label_data = Image.open(data_label_path).convert("RGB")


		if "result" not in data_underwater_path:
			data_underwater = data_underwater.resize((self.size, self.size), Image.BICUBIC)
		if "result" not in data_label_path:
			label_data = label_data.resize((self.size, self.size), Image.BICUBIC)

		data_underwater, label_data, _ = augment(data_underwater, label_data)

		data_underwater = np.asarray(data_underwater) / 255.0
		data_underwater_output = torch.from_numpy(data_underwater).float().permute(2, 0, 1)

		label_data = np.asarray(label_data) / 255.0
		label_data_output = torch.from_numpy(label_data).float().permute(2, 0, 1)

		# ============== 新增: 返回退化类型 ==============
		return data_underwater_output, label_data_output, deg_type

	def __len__(self):
		return len(self.train_list)

