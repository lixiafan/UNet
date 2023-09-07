from PIL import Image

# 对数据集图片进行等比缩放，粘贴到左上角
def keep_image_size_open(path,size = (256,256)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB',(temp,temp),(0,0,0))
    mask.paste(img,(0,0))
    mask = mask.resize(size)
    return mask