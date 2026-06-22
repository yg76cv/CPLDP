from turtle import forward
import torchvision.transforms as transforms
import torch
import clip
import torch.nn as nn
from torch.nn import functional as F
from CLIP.clip import load

device = "cuda" if torch.cuda.is_available() else "cpu"
#load clip
model, preprocess = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="./clip_model/")#"ViT-B/32"
model.to(device)
for para in model.parameters():
	para.requires_grad = False

def get_clip_score(tensor,words):
	score=0
	for i in range(tensor.shape[0]):
		#image preprocess
		clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
		img_resize = transforms.Resize((224,224))
		image2=img_resize(tensor[i])
		image=clip_normalizer(image2).unsqueeze(0)
		#get probabilitis
		text = clip.tokenize(words).to(device)
		logits_per_image, logits_per_text = model(image, text)
		probs = logits_per_image.softmax(dim=-1)
		#2-word-compared probability
		# prob = probs[0][0]/probs[0][1]#you may need to change this line for more words comparison
		prob = probs[0][0]
		score =score + prob

	return score


class L_clip(nn.Module):
	def __init__(self):
		super(L_clip,self).__init__()
		for param in self.parameters(): 
			param.requires_grad = False
  
	def forward(self, x, light):
		k1 = get_clip_score(x,["dark","normal light"])
		if light:
			k2 = get_clip_score(x,["noisy photo","clear photo"])
			return (k1+k2)/2
		return k1

class Prompts(nn.Module):
	def __init__(self,initials=None):
		super(Prompts,self).__init__()
		if initials!=None:
			text = clip.tokenize(initials).cuda()
			with torch.no_grad():
				self.text_features = model.encode_text(text).cuda()
		else:
			self.text_features=torch.nn.init.xavier_normal_(nn.Parameter(torch.cuda.FloatTensor(2,512))).cuda()

	def forward(self,tensor):
		for i in range(tensor.shape[0]):
			image_features=tensor[i]
			nor=torch.norm(self.text_features,dim=-1, keepdim=True)
			similarity = (model.logit_scale.exp() * image_features @ (self.text_features/nor).T).softmax(dim=-1)
			if(i==0):
				probs=similarity
			else:
				probs=torch.cat([probs,similarity],dim=0)
		return probs

clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
img_resize = transforms.Resize((224,224))

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

def _clip_preprocess_batch(img: torch.Tensor) -> torch.Tensor:
    """
    img: [B,3,H,W]，建议范围 [0,1]
    return: [B,3,224,224] normalized
    """
    img = img.clamp(0, 1)
    img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)

    mean = img.new_tensor(_CLIP_MEAN)[None, :, None, None]
    std  = img.new_tensor(_CLIP_STD)[None, :, None, None]
    img = (img - mean) / std
    return img


def clip_similarity_from_feature(img: torch.Tensor, text_features: torch.Tensor, model) -> torch.Tensor:
    """
    返回每张图与 text_features 的 cosine sim:
    - img: [B,3,H,W]
    - text_features: [1,512] 或 [K,512]
    return:
    - 如果 text_features 是 [1,512] => sim: [B]
    - 如果 text_features 是 [K,512] => sim: [B,K]
    """
    x = _clip_preprocess_batch(img)              # [B,3,224,224]
    img_feat = model.encode_image(x)             # [B,512]
    img_feat = F.normalize(img_feat, dim=-1)

    text_feat = F.normalize(text_features, dim=-1)
    if text_feat.dim() == 2 and text_feat.size(0) == 1:
        sim = (img_feat * text_feat).sum(dim=-1)     # [B]
    else:
        sim = img_feat @ text_feat.t()               # [B,K]
    return sim


def get_clip_score_from_feature(tensor,text_features):
	score=0
	text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
	for i in range(tensor.shape[0]):
		image2=img_resize(tensor[i])
		image=clip_normalizer(image2.reshape(1,3,224,224))
  
		image_features = model.encode_image(image)
		# 归一化图像特征
		image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
		# 直接计算余弦相似度 (范围[-1, 1])
		cosine_sim = (image_features_norm @ text_features_norm.T).squeeze()

		# 将相似度映射到[0,1]范围
		normalized_sim = (cosine_sim + 1) / 2.0
		score += normalized_sim.item()
	# 	image_nor=image_features.norm(dim=-1, keepdim=True)
	# 	nor= text_features.norm(dim=-1, keepdim=True)
	# 	similarity = (100.0 * (image_features/image_nor) @ (text_features/nor).T).softmax(dim=-1)
	# 	probs = similarity
	# 	prob = probs[0][0]
	# 	score =score + prob
	# score=score/tensor.shape[0]
	return score / tensor.shape[0]

#
# class L_clip_from_feature(nn.Module):
# 	def __init__(self):
# 		super(L_clip_from_feature,self).__init__()
# 		for param in self.parameters():
# 			param.requires_grad = False
#
# 	def forward(self, x, text_features):
# 		k1 = get_clip_score_from_feature(x,text_features)
# 		return k1

class L_clip_from_feature(nn.Module):
    """
    输出一个可反传的 similarity（越大越好）
    训练里可以用 loss = w * (1 - sim01.mean()) 或 loss = -w * sim.mean()
    """
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        for p in self.clip_model.parameters():
            p.requires_grad = False

    def forward(self, x, text_features):
        # cosine sim in [-1,1]
        sim = clip_similarity_from_feature(x, text_features, self.clip_model)  # [B] or [B,K]

        # 如果是 [B,K]，你可以取某一列或取均值，这里默认取均值
        if sim.dim() == 2:
            sim = sim.mean(dim=1)

        # 映射到 [0,1]，与你训练代码里 (1 - clip_sim) 的设计一致
        sim01 = (sim + 1.0) / 2.0
        return sim01.mean()

#for clip reconstruction loss
res_model, res_preprocess = load("RN101", device=device, download_root="./clip_model/")
for para in res_model.parameters():
	para.requires_grad = False

def l2_layers(pred_conv_features, input_conv_features,weight):
	weight=torch.tensor(weight).type(pred_conv_features[0].dtype)
	return weight@torch.tensor([torch.square(x_conv - y_conv).mean() for x_conv, y_conv in
			zip(pred_conv_features, input_conv_features)],requires_grad=True)/len(weight)

def get_clip_score_MSE(pred,inp,weight):
	score=0
	for i in range(pred.shape[0]):

		pred_img=img_resize(pred[i])
		pred_img=clip_normalizer(pred_img.reshape(1,3,224,224))
		pred_image_features = res_model.encode_image(pred_img)

		inp_img=img_resize(inp[i])
		inp_img=clip_normalizer(inp_img.reshape(1,3,224,224))
		inp_image_features = res_model.encode_image(inp_img)
		
		MSE_loss_per_img=0
		for feature_index in range(len(weight)):
				MSE_loss_per_img=MSE_loss_per_img+weight[feature_index]*F.mse_loss(pred_image_features[1][feature_index].squeeze(0),inp_image_features[1][feature_index].squeeze(0))
		score = score + MSE_loss_per_img
	return score


class L_clip_MSE(nn.Module):
	def __init__(self):
		super(L_clip_MSE,self).__init__()
		for param in self.parameters(): 
			param.requires_grad = False
		
	def forward(self, pred, inp,weight=[1.0,1.0,1.0,1.0,0.5]):
		res = get_clip_score_MSE(pred,inp,weight)
		return res


class four_margin_loss(nn.Module):
	def __init__(self,dis1=0.7,dis2=0.3):
		super(four_margin_loss, self).__init__()
		self.margin_loss_L=nn.MarginRankingLoss(dis1)
		self.margin_loss_S=nn.MarginRankingLoss(dis2)
		self.clip_loss=L_clip_from_feature()
	
	def forward(self,tensor0,tensor3,labels,num,*tensor_mid):
		loss_inp_ref=self.margin_loss_L(tensor0,tensor3,labels)
		if num==2:
			print(tensor0,tensor3)
			return loss_inp_ref
		elif num==3:
			print(tensor0,tensor_mid,tensor3)
			loss_inp_semi1=self.margin_loss_L(tensor0,tensor_mid[0],labels)
			loss_semi1_ref=self.margin_loss_S(tensor_mid[0],tensor3,labels)
			return loss_inp_ref+loss_inp_semi1+loss_semi1_ref

		elif num==4:
			print(tensor0,tensor_mid,tensor3)
			loss_inp_semi1=self.margin_loss_L(tensor0,tensor_mid[0],labels)
			loss_semi1_semi2=self.margin_loss_S(tensor_mid[0],tensor_mid[1],labels)
			loss_semi2_ref=self.margin_loss_S(tensor_mid[1],tensor3,labels)
			return loss_inp_ref+loss_inp_semi1+loss_semi1_semi2+loss_semi2_ref
