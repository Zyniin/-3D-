# import argparse
# from skimage import io
# from skimage.metrics import peak_signal_noise_ratio, structural_similarity
#
# # 创建命令行参数解析器
# parser = argparse.ArgumentParser(description='比较图像的质量指标')
# parser.add_argument('original_image', type=str, help='原始图像的文件路径')
# parser.add_argument('processed_image', type=str, help='处理后的图像的文件路径')
# args = parser.parse_args()
#
# # 读取原始图像和处理后的图像
# img_original = io.imread(args.original_image)
# img_processed = io.imread(args.processed_image)
#
# # 计算峰值信噪比（PSNR）
# psnr_value = peak_signal_noise_ratio(img_original, img_processed)
#
# # 计算结构相似性指数（SSIM）
# ssim_value = structural_similarity(img_original, img_processed, win_size=3, multichannel=True)
#
# # 打印结果
# print('PSNR:', psnr_value)
# print('SSIM:', ssim_value)

import argparse
from skimage import io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='比较图像的质量指标')
parser.add_argument('image_pairs', nargs='+', type=str, help='图片对的文件路径')
args = parser.parse_args()

# 遍历每一组图片
for pair in args.image_pairs:
    # 解析图片路径
    original_image, processed_image = pair.split(',')

    # 读取原始图像和处理后的图像
    img_original = io.imread(original_image)
    img_processed = io.imread(processed_image)

    # 计算峰值信噪比（PSNR）
    psnr_value = peak_signal_noise_ratio(img_original, img_processed)

    # 计算结构相似性指数（SSIM）
    ssim_value = structural_similarity(img_original, img_processed, win_size=3, multichannel=True)

    # 打印结果
    print(f'图片对: {original_image}, {processed_image}')
    print('PSNR:', psnr_value)
    print('SSIM:', ssim_value)
    print('---------------------')