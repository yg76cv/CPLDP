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

def augment(img_in, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            info_aug['trans'] = True

    return img_in, info_aug



device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
#load clip
model, preprocess = clip.load("ViT-B/32", device=device, download_root="./clip_model/")#ViT-B/32
for para in model.parameters():
	para.requires_grad = False

def _read_id_list(*candidate_paths):
	for p in candidate_paths:
		if p and os.path.exists(p):
			with open(p, 'r', encoding='utf-8') as f:
				return [line.strip() for line in f if line.strip()]
	return []

class underwater_loader(data.Dataset):

	def __init__(self, underwater_images_path,reference_images_path, detype,data_path):
		self.underwater_path = underwater_images_path
		self.reference_path = reference_images_path
		self.de_type = detype
		self.prompt_path=data_path

		# 初始化各类退化图像ID
		self.green_ids = []
		self.blue_ids = []
		self.gb_ids = []  # green-blue
		self.dark_ids = []
		self.haze_ids = []
		self.reference_ids = []

		# 初始化并合并ID
		self._init_ids()
		self._merge_ids()

		self.size = 256
		print(f"Total training examples: {len(self.sample_ids)}")

	def _init_ids(self):
		if 'blue' in self.de_type or 'green' in self.de_type or 'green_blue' in self.de_type:
			self._init_color_ids()
		if 'dark' in self.de_type:
			self._init_dark_ids()
		if 'haze' in self.de_type:
			self._init_haze_ids()
		self._init_reference_ids()

	def _init_color_ids(self):
		# 读取各类颜色退化图像的ID文件
		green_file = os.path.join(self.prompt_path, "green.txt")
		blue_file = os.path.join(self.prompt_path, "blue.txt")
		gb_file = os.path.join(self.prompt_path, "green-blue.txt")

		# 读取并存储ID
		self.green_ids = [os.path.join(self.underwater_path,id_.strip())  for id_ in open(green_file)]
		self.blue_ids = [os.path.join(self.underwater_path,id_.strip())  for id_ in open(blue_file)]
		self.gb_ids = [os.path.join(self.underwater_path,id_.strip())  for id_ in open(gb_file)]

		if 'green' in self.de_type:
			self.gr_ids = [{"clean_id": x, "de_type": 0} for x in self.green_ids]
			self.gr_ids = self.gr_ids * 3
			random.shuffle(self.gr_ids)
			self.gr_counter = 0
		if 'blue' in self.de_type:
			self.bl_ids = [{"clean_id": x, "de_type": 1} for x in self.blue_ids]
			self.bl_ids = self.bl_ids * 3
			random.shuffle(self.bl_ids)
			self.bl_counter = 0
		if 'green_blue' in self.de_type:
			self.gb_ids = [{"clean_id": x, "de_type": 2} for x in self.gb_ids]
			self.gb_ids = self.gb_ids * 6
			random.shuffle(self.gb_ids)
			self.gb_counter = 0

		self.num_clean = len(self.gb_ids)+len(self.bl_ids)+len(self.gr_ids)
		print("Total Color Ids : {}".format(self.num_clean))

	def _init_dark_ids(self):
		temp_ids = []
		dark = self.prompt_path + "Dark.txt"
		temp_ids += [os.path.join(self.underwater_path,id_.strip())  for id_ in open(dark)]
		self.dark_ids = [{"clean_id": x, "de_type": 3} for x in temp_ids]


		self.dark_counter = 0

		self.num_dark = len(self.dark_ids)
		print("Total Dark Ids : {}".format(self.num_dark))

	def _init_haze_ids(self):
		temp_ids = []
		rs = self.prompt_path + "haze.txt"
		temp_ids += [os.path.join(self.underwater_path,id_.strip())  for id_ in open(rs)]
		self.haze_ids = [{"clean_id": x, "de_type": 4} for x in temp_ids]
		self.haze_ids = self.haze_ids * 3
		random.shuffle(self.haze_ids)
		self.haze_counter = 0
		self.num_haze = len(self.haze_ids)
		print("Total Haze Ids : {}".format(self.num_haze))

	def _init_reference_ids(self):
		reference_files = glob.glob(self.reference_path + "*.png") + glob.glob(self.reference_path + "*.jpg")
		#print(reference_files)
		self.reference_ids = [{"clean_id": x, "de_type": 5} for x in reference_files]
		self.reference_counter = 0
		self.num_reference = len(self.reference_ids)
		print("Total Reference Ids : {}".format(self.num_reference))

	def _merge_ids(self):
		self.sample_ids = []
		if "green" in self.de_type:
			self.sample_ids += self.gr_ids
		if "blue" in self.de_type:
			self.sample_ids += self.bl_ids
		if "green_blue" in self.de_type:
			self.sample_ids += self.gb_ids
		if "dark" in self.de_type:
			self.sample_ids += self.dark_ids
		if "haze" in self.de_type:
			self.sample_ids += self.haze_ids
		if self.reference_ids:
			self.sample_ids += self.reference_ids

		print(len(self.sample_ids))
		random.shuffle(self.sample_ids)
		

	def __getitem__(self, index):
		sample = self.sample_ids[index]
		de_id = sample["de_type"]
		underwater_id = sample["clean_id"]
		#underwaterid_path=os.path.join(self.underwater_path, underwater_id)
		underwaterid_path =  underwater_id
		# 加载干净图像
		underwater_img = Image.open(underwaterid_path).convert('RGB')
		underwater_img = underwater_img.resize((self.size, self.size), Image.BICUBIC)
		underwater_img, _ = augment(underwater_img)  # 数据增强

		# 转换为张量
		underwater_img = np.asarray(underwater_img) / 255.0
		underwater_img = torch.from_numpy(underwater_img).float().permute(2, 0, 1)

		# 预处理用于CLIP
		clip_normalizer = transforms.Normalize(
			(0.48145466, 0.4578275, 0.40821073),
			(0.26862954, 0.26130258, 0.27577711)
		)
		img_resize = transforms.Resize((224, 224))
		img_clip = img_resize(underwater_img)
		img_clip = clip_normalizer(img_clip.unsqueeze(0))

		# 使用CLIP编码图像
		with torch.no_grad():
			image_features = model.encode_image(img_clip)



		# 返回图像特征和退化类型标签
		return image_features.squeeze(0), de_id

	def __len__(self):
		"""返回数据集大小"""
		return len(self.sample_ids)



