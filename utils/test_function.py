import torch
import os
import time
import numpy as np
from PIL import Image
import glob
import torchvision
import clip
import torch.nn as nn
import torchvision.transforms as transforms
from collections import OrderedDict

# 加载CLIP模型（用于退化类型预测）
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, download_root="./clip_model/")


# 文本编码器类（用于处理退化提示）
class TextEncoder(nn.Module):
	def __init__(self, clip_model):
		super().__init__()
		self.transformer = clip_model.transformer
		self.positional_embedding = clip_model.positional_embedding
		self.ln_final = clip_model.ln_final
		self.text_projection = clip_model.text_projection
		self.dtype = clip_model.dtype

	def forward(self, prompts, tokenized_prompts):
		x = prompts + self.positional_embedding.type(self.dtype)
		x = x.permute(1, 0, 2)  # NLD -> LND
		x = self.transformer(x)
		x = x.permute(1, 0, 2)  # LND -> NLD
		x = self.ln_final(x).type(self.dtype)
		x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
		return x


# 退化类型预测函数
def predict_degradation_type(image_tensor, deg_text_features):
	"""
	预测输入图像的退化类型并返回对应的提示向量

	参数:
		image_tensor: 输入图像张量 [1, 3, H, W]
		deg_text_features: 所有退化类型的文本特征 [5, 512]

	返回:
		condition_vector: 退化提示向量 [1, 512]
		deg_type: 预测的退化类型 (0-4)
	"""
	# 预处理图像
	clip_normalizer = transforms.Normalize(
		(0.48145466, 0.4578275, 0.40821073),
		(0.26862954, 0.26130258, 0.27577711)
	)
	img_resize = transforms.Resize((224, 224))

	img_clip = img_resize(image_tensor)
	img_clip = clip_normalizer(img_clip)

	# 提取图像特征
	with torch.no_grad():
		image_features = model.encode_image(img_clip)

	image_features = image_features.to(torch.float32)  # 转换为半精度
	deg_text_features = deg_text_features.to(torch.float32)  # 转换为半精度


# 计算与各类退化提示的相似度
	similarities = image_features @ deg_text_features.T

	# 获取相似度最高的退化类型
	deg_type = torch.argmax(similarities).item()

	# 返回对应的退化提示向量
	return deg_text_features[deg_type].unsqueeze(0), deg_type


def lowlight(image_path, image_list_path, result_list_path, DCE_net, deg_text_features, size=256):
	"""
	增强低光图像

	参数:
		image_path: 输入图像路径
		image_list_path: 输入图像目录路径
		result_list_path: 输出结果目录路径
		DCE_net: 增强网络模型
		deg_text_features: 退化类型文本特征 [5, 512]
		size: 调整大小尺寸 (默认256)
	"""
	# 加载图像
	data_lowlight = Image.open(image_path)
	data_lowlight = data_lowlight.resize((size, size), Image.ANTIALIAS)
	data_lowlight = np.asarray(data_lowlight) / 255.0

	# 转换为张量
	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
	data_lowlight = data_lowlight.unsqueeze(0).cuda()  # [1, C, H, W]

	# 预测退化类型并获取提示向量
	condition_vector, deg_type = predict_degradation_type(data_lowlight, deg_text_features)

	# 使用退化提示增强图像
	enhanced_image = DCE_net(data_lowlight, condition=condition_vector)

	# 准备输出路径
	output_path = image_path.replace(image_list_path, result_list_path)
	output_path = output_path.replace('.JPG', '.png').replace('.jpg', '.png')

	# 创建输出目录
	output_dir = os.path.dirname(output_path)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	# 保存结果
	torchvision.utils.save_image(enhanced_image, output_path)

	return deg_type  # 返回预测的退化类型


def inference(image_list_path, result_list_path, DCE_net, deg_text_features_path, size=256):
	"""
	批量推理函数

	参数:
		image_list_path: 输入图像目录路径
		result_list_path: 输出结果目录路径
		DCE_net: 增强网络模型
		deg_text_features_path: 退化文本特征文件路径
		size: 调整大小尺寸 (默认256)
	"""
	# 加载退化文本特征
	deg_text_features = torch.load(deg_text_features_path).to(device)

	with torch.no_grad():
		filePath = image_list_path
		file_list = os.listdir(filePath)

		print("Inferencing...")
		deg_counts = [0] * 5  # 统计各类退化数量

		for file_name in file_list:
			test_list = glob.glob(os.path.join(filePath, file_name, '*')) if os.path.isdir(os.path.join(filePath, file_name)) else [
				os.path.join(filePath, file_name)]

			for image in test_list:
				if image.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
					start_time = time.time()
					deg_type = lowlight(image, image_list_path, result_list_path,
										DCE_net, deg_text_features, size)
					deg_counts[deg_type] += 1
					print(f"Processed: {os.path.basename(image)} | Deg type: {deg_type} | Time: {time.time() - start_time:.2f}s")

		# 打印退化类型统计
		print("\nDegradation type statistics:")
		deg_names = ["Green", "Blue", "Green-Blue", "Dark", "Haze"]
		for i, count in enumerate(deg_counts):
			print(f"{deg_names[i]}: {count} images")