from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image
import time
from tqdm import trange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weight_path = 'params/weight_unet_voc.pth'
data_path = 'F:\Model_U-Net\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007'
save_path = 'train_image'

if __name__ == '__main__':
    data_loader = DataLoader(MyDataset(data_path),batch_size=1,shuffle=True)
    net = UNet().to(device)
    if os.path.exists(weight_path):
        '''net = NET()
        net.cuda()
        net = nn.DataParallel(net)
        net.load_state_dict(torch.load('net.pth')
        '''
        # net.cuda()
        net = nn.DataParallel(net)
        net.load_state_dict(torch.load(weight_path),False)
        print('successfully load weight')
    else:
        print('not successfully load weight')

    opt = optim.Adam(net.parameters())
    loss_fun = nn.BCELoss()

    epoch = 1
    n_epoch = 10
    # # 加上进度条，trange(i) 是 tqdm(range(i)) 的另一种写法
    # tbar = trange(100, unit='batch', ncols=100)
    start_time = time.time()
    while epoch != n_epoch:
        for i,(image,segment_image) in enumerate(data_loader):
            image, segment_image = image.to(device),segment_image.to(device)
            out_img = net(image)
            train_loss = loss_fun(out_img,segment_image)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i % 5 == 0:# 每5个batch打印损失
                print(f'{epoch}-{i}-train_loss===>{train_loss.item()}')
            if i % 50 == 0:
                torch.save(net.state_dict(),weight_path)

            _image = image[0]
            _segment_image = segment_image[0]
            _out_image = out_img[0]

            img = torch.stack([_image,_segment_image,_out_image],dim = 0)
            save_image(img,f'{save_path}/{i}.png')
        epoch += 1
        # tbar.set_description('Epoch %d/%d ### ' % (epoch + 1, n_epoch))
    end_time = time.time()
    print('   Time:{}'.format(end_time - start_time)) # Time:5793.720459461212