import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, sampler
import torchvision.transforms as transforms
import numpy as np
import argparse
from Model import SalDA
from scipy import ndimage
from scipy import misc
from Arguments import get_args
from PIL import Image, ImageOps
from load_imglist import ImageList
import torchvision.transforms.functional as TF


global args #defining variables that can be accessed globally
global device #defining variables that can be accessed globally
global count
count = 1
args = get_args()
device = torch.device("cuda" if args.cuda else "cpu")

def te(test_loader, model):
    model.eval()
    print('entered test')
    for i, (ip, gt) in enumerate(test_loader):
        ip = ip.to(device)
        predict(model, ip, i)

def predict(model, image_stimuli,img_no):
    global count
    print('predict image [{}/5000]'.format(args.batch_size*img_no))
    predicted = model(image_stimuli)

    for i in range(predicted.shape[0]):
        img= predicted[i]
        img = img.detach().cpu().numpy().reshape(240,320)

        # max-min normlize
        img = (img - img.min()) / (img.max() - img.min())
        img = img * 255

        img = ndimage.gaussian_filter(img,sigma=2)
        img = Image.fromarray(img)

        img = Image.fromarray(np.uint8(img), 'L')
        img = TF.resize(img, (480, 640))
        # img.show()
        img.save("./result/SalDA/"+ "out_{:04d}.png".format(count))
        count +=1

test_loader = torch.utils.data.DataLoader(
                ImageList(root = args.root_path, fileList = args.test_list,transform = None),
                batch_size = args.batch_size, 
                shuffle = False,
                num_workers = 0,
                pin_memory = True)



model = SalDA()
model.load_state_dict(torch.load('./Models/Model_Save_new/SalDA/SalDA_25.pth')['state_dict'])
model = model.to(device)
# print(model)
print('model loaded and test started')
model.eval()
te(test_loader, model)
