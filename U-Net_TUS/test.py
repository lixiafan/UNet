from net import *
import os
from utils import *
from data import *
import cv2
from torchvision.utils import  save_image
from torch.utils.data import DataLoader
'''
net = NET()
net.cuda()
net = nn.DataParallel(net)
net.load_state_dict(torch.load('net.pth')
'''
net = UNet().cuda()
net = nn.DataParallel(net)
# weights = 'params/unet_voc.pth'# 百度网盘上下载的权重文件
weights = 'params/unet_TUS.pth'# 自定义的网络保存的权重参数
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights),False)
    print('successfully load weight!')
else:
    print('no load weight!')
_input = r'/home/inspur/asc22/lxf/U-Net_TUS/test_image/58.PNG'
# _input = input('please input the path of your image:') # F:\Model_U-Net\test_image\0005
img = keep_image_size_open(_input) # 输出大小为256x256的image
img_data = transform(img).cuda()
# print(img_data.shape)
img_data = torch.unsqueeze(img_data,dim=0) # 降维
out = net(img_data)
# print(out)
save_image(out,'result/result_550epoch.jpg')



