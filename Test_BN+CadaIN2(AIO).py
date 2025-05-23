import torch
import torch.nn as nn
import numpy as np
import os
from time import time
import math
from skimage.metrics import structural_similarity as ssim
from argparse import ArgumentParser
import glob
import cv2
from PIL import Image
from Forward_functions import FrFT_forward
from Forward_functions import adaptive_instance_normalization as adain
parser = ArgumentParser(description='LFM_BN_CAdaIN2(AIO)')

parser.add_argument('--epoch_num', type=int, default=100, help='epoch number of model')
parser.add_argument('--epoch_num_now', type=int, default=100, help='epoch number of model')
parser.add_argument('--layer_num', type=int, default=5, help='phase number of DPUNet')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--frft_order', type=float, default=0.1, help='from{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data/test', help='training or test data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--test_name', type=str, default='unnatural6', help='name of test set')
args = parser.parse_args()

epoch_num = args.epoch_num
epoch_num_now = args.epoch_num_now
layer_num = args.layer_num
group_num = args.group_num
gpu_list = args.gpu_list
test_name = args.test_name
frft_order = args.frft_order

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
torch.set_default_tensor_type(torch.cuda.FloatTensor)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation (SE) Block"""
    def __init__(self, channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ProxNet(nn.Module):
    """CPNet + SENet"""
    def __init__(self, in_nc=1, feature=64, kernel_size=3, padding=1):
        super(ProxNet, self).__init__()
        self.relu = nn.ReLU()
        self.eta1 = nn.Linear(in_nc, feature)
        self.eta2 = nn.Linear(in_nc, feature)
        self.eta3 = nn.Linear(in_nc, feature)
        self.eta4 = nn.Linear(in_nc, feature)
        self.beta1 = nn.Linear(in_nc, feature)
        self.beta2 = nn.Linear(in_nc, feature)
        self.beta3 = nn.Linear(in_nc, feature)
        self.beta4 = nn.Linear(in_nc, feature)

        self.eta11 = nn.Linear(feature, feature)
        self.eta21 = nn.Linear(feature, feature)
        self.eta31 = nn.Linear(feature, feature)
        self.eta41 = nn.Linear(feature, feature)
        self.beta11 = nn.Linear(feature, feature)
        self.beta21 = nn.Linear(feature, feature)
        self.beta31 = nn.Linear(feature, feature)
        self.beta41 = nn.Linear(feature, feature)

        self.conv_layer1 = nn.Conv2d(in_channels=in_nc, out_channels=feature, kernel_size=kernel_size, padding=padding, bias=False).to(device)
        self.conv_layer2 = nn.Conv2d(in_channels=feature, out_channels=feature, kernel_size=kernel_size, padding=padding, bias=False).to(device)
        self.conv_layer3 = nn.Conv2d(in_channels=feature, out_channels=in_nc, kernel_size=kernel_size, padding=padding, bias=False).to(device)

        self.SENet = SEBlock(feature, reduction_ratio=16)

    def forward(self, x_param, x_img):
        x_input = x_img.unsqueeze(1)

        eta1 = self.eta11(self.relu(self.eta1(x_param))).reshape(1, 64, 1, 1)
        eta2 = self.eta21(self.relu(self.eta2(x_param))).reshape(1, 64, 1, 1)
        eta3 = self.eta31(self.relu(self.eta3(x_param))).reshape(1, 64, 1, 1)
        eta4 = self.eta41(self.relu(self.eta4(x_param))).reshape(1, 64, 1, 1)
        beta1 = self.beta11(self.relu(self.beta1(x_param))).reshape(1, 64, 1, 1)
        beta2 = self.beta21(self.relu(self.beta2(x_param))).reshape(1, 64, 1, 1)
        beta3 = self.beta31(self.relu(self.beta3(x_param))).reshape(1, 64, 1, 1)
        beta4 = self.beta41(self.relu(self.beta4(x_param))).reshape(1, 64, 1, 1)

        output1 = self.relu(adain(self.conv_layer1(x_input.float()), eta1, beta1))
        output2 = self.relu(adain(self.conv_layer2(output1), eta2, beta2))
        output3 = self.relu(adain(self.conv_layer2(output2), eta3, beta3))
        output4 = self.relu(adain(self.conv_layer2(output3), eta4, beta4))
        output5 = self.SENet(output4)
        output6 = self.conv_layer3(output5)
        x_pred = output6.squeeze(1)
        return x_pred + x_img

class DUNet(torch.nn.Module):
    """(4) BN+CAdaIN2 (AIO): Two parameters of AdaIN are generated by two different fully connected layers
    with physical parameters (such as p)."""
    def __init__(self, LayerNo):
        super(DUNet, self).__init__()
        self.step_size = nn.Parameter(0.5 * torch.ones(LayerNo))
        self.LayerNo = LayerNo
        onelayer = []
        for i in range(LayerNo):
            onelayer.append(ProxNet())
        self.fcs = nn.ModuleList(onelayer)

    def forward(self, measurement, frft_order):
        x = torch.ones_like(measurement)
        for i in range(self.LayerNo):
            z_hat = FrFT_forward(x, frft_order)
            x = torch.abs(FrFT_forward(measurement * torch.exp(1j * torch.angle(z_hat)), 4-frft_order))   #AP
            x = self.fcs[i](frft_order,x)
        x_final = x.clone()
        return x_final

def psnr(img1,img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100

    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# Building the model
model = DUNet(layer_num).to(device)
model = nn.DataParallel(model)

# Load pre-trained model with epoch number
model_name = f"BN+CAdaIN2(AIO)_{epoch_num}_layer_{layer_num}"
model_dir = os.path.join(args.model_dir, model_name)
model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, epoch_num_now)))
model.eval()

test_dir = os.path.join(args.data_dir, test_name)
filepaths = glob.glob(test_dir + '/*.png')
result_dir = os.path.join(args.result_dir, test_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
ImgNum = len(filepaths)
PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
Prediction_all = np.zeros([ImgNum, 256, 256], dtype=np.float32)
Time_all = np.zeros([1, ImgNum], dtype=np.float32)
print('\n')
print("PR Reconstruction Start")

# Start Reconstruction
with torch.no_grad():
    for i in range(ImgNum):
        imgName = filepaths[i]
        imsize = 256
        img_orig = Image.open(imgName)
        img_np = img_orig.resize((imsize, imsize))
        img_np = np.array(img_np).astype(np.float32)
        img_np = img_np / np.max(img_np)
        img_np_output = torch.from_numpy(img_np).unsqueeze(0).to(device)
        frft_order = torch.tensor(np.array([args.frft_order]), dtype=torch.float32).to(device)
        z = FrFT_forward(img_np_output, frft_order)
        FrFT_abs = torch.abs(z)
        Rawdata = FrFT_abs ** 2
        Y = torch.clamp(Rawdata, min=0.0)
        measurement = torch.sqrt(Y)

        start = time()
        x_output = model(measurement, frft_order)
        end = time()

        # Calculation and Print
        loss_mse = torch.mean(torch.pow(x_output - img_np_output, 2))
        X_rec = x_output.squeeze(0).cpu().numpy()
        X_rec = np.clip(X_rec, 0.0, 1.0)
        rec_PSNR = psnr(X_rec * 255, img_np * 255)
        rec_SSIM = ssim(X_rec * 255, img_np * 255, data_range=255)
        print("Run time is %.4f, MSE is %.2f, PSNR is %.2f, SSIM is %.2f" % ((end - start), loss_mse, rec_PSNR, rec_SSIM))
        PSNR_All[0, i] = rec_PSNR
        SSIM_All[0, i] = rec_SSIM
        Prediction_all[i, :, :] = X_rec
        Time_all[0, i] = end - start
        im_rec_rgb = np.clip(X_rec * 255, 0, 255).astype(np.uint8)
        resultName = imgName.replace(args.data_dir, args.result_dir)
        cv2.imwrite("%s_SFrFPR_%d_DUNet_order_%d_epoch_%d_PSNR_%.2f_SSIM_%.6f.png" % (
         resultName, epoch_num, args.frft_order, epoch_num, rec_PSNR, rec_SSIM), im_rec_rgb)
    print(np.mean(PSNR_All))
    print(np.mean(SSIM_All))
    print(np.mean(Time_all))
print("SFrFPR Reconstruction End")


