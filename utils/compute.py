import os
from glob import glob
from PIL import Image
import torch
import pyiqa
from torchvision import transforms
from tqdm import tqdm
from torchvision.transforms import InterpolationMode

def inferencer(reference_list_path, result_list_path):
    # 文件夹路径（替换为你自己的路径）
    gt_folder = reference_list_path       # GT图像路径
    gen_folder = result_list_path  # 增强图像路径

    # 获取文件列表并排序
    gt_files = sorted(glob(os.path.join(gt_folder, '*')))
    gen_files = sorted(glob(os.path.join(gen_folder, '*')))

    # 加载PSNR和SSIM指标
    psnr_metric = pyiqa.create_metric('psnr')
    ssim_metric = pyiqa.create_metric('ssim')

    # 定义预处理：resize 和 ToTensor
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])

    # 存储结果
    psnr_scores, ssim_scores = [], []

    # 遍历每一对图像
    for gt_path, gen_path in tqdm(zip(gt_files, gen_files), total=len(gt_files)):
        gt_img = transform(Image.open(gt_path).convert('RGB')).unsqueeze(0)
        gen_img = transform(Image.open(gen_path).convert('RGB')).unsqueeze(0)

        psnr = psnr_metric(gen_img, gt_img).item()
        ssim = ssim_metric(gen_img, gt_img).item()

        psnr_scores.append(psnr)
        ssim_scores.append(ssim)

    # 计算平均值
    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    avg_ssim = sum(ssim_scores) / len(ssim_scores)

    print(f'Average PSNR: {avg_psnr:.4f}')
    print(f'Average SSIM: {avg_ssim:.4f}')
    return avg_psnr,avg_ssim
