# -*- coding: utf-8 -*-
import os, sys
import torch
import torchvision
import numpy as np
import cv2 as cv
from unet_model import *

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = 'D:/PycharmProjects/COMP9517-Surfing-Project/u_net/save_model/unet_road_model.pt'

unet = Unet().to(device)
model_dict = unet.load_state_dict(torch.load(model_path))


# print(model_dict)

def test(unet):
    # model_dict=unet.load_state_dict(torch.load(model_path))
    root_dir = './CrackForest-dataset-master/test/'
    fileNames = os.listdir(root_dir)
    # print(fileNames)
    for f in fileNames:
        image = cv.imread(os.path.join(root_dir, f), cv.IMREAD_GRAYSCALE)
        # print(image)
        h, w = image.shape
        # print(image.shape)
        img = np.float32(image) / 255.0
        img = np.expand_dims(img, 0)
        x_input = torch.from_numpy(img).view(1, 1, h, w)
        # probs = unet(x_input.cuda())
        probs = unet(x_input.to(device))
        # print(probs,probs.shape)#torch.Size([1, 2, 320, 480])
        m_label_out_ = probs.transpose(1, 3).transpose(1, 2).contiguous().view(-1, 2)
        # print(m_label_out_,m_label_out_.shape)#torch.Size([153600, 2])
        grad, output = m_label_out_.data.max(dim=1)
        # print(m_label_out_.data.max(dim=1))
        output[output > 0] = 255
        predic_ = output.view(h, w).cpu().detach().numpy()

        # print(predic_)
        # print(predic_.max())
        # print(predic_.min())

        # print(predic_.argmax(-1))
        # print(predic_.shape)
        """显示测试结果"""
        result = cv.resize(np.uint8(predic_), (w, h))
        # cv.imshow("input", image)
        #
        # cv.imshow("unet-segmentation-demo", result)
        # cv.waitKey(0)
        """将结果保存在测试seg目录下"""
        # result = cv.resize(np.uint8(predic_), (w, h))
        result_image_path = os.path.join('./CrackForest-dataset-master/png_img_dir', f)  # 存放测试结果
        cv.imwrite(result_image_path, result)
    # cv.destroyAllWindows()


if __name__ == '__main__':
    test(unet)