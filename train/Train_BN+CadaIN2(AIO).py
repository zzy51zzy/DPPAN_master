import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
from Forward_functions_v2 import FrFT_forward
from Forward_functions_v2 import adaptive_instance_normalization as adain
from torch.cuda.amp import autocast as autocast

parser = ArgumentParser(description='LFM_BN+CAdaIN2(AIO)')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=100, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=5, help='phase number of DPUNet')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--batch_size', type=int, default=10, help='group number for training')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--use_amp', type=str, default='True', help='use amp for training')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--image_dir', type=str, default='data/BSD6000', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')

args = parser.parse_args()

start_epoch = args.start_epoch
end_epoch = args.end_epoch
layer_num = args.layer_num
image_dir = args.image_dir
group_num = args.group_num
gpu_list = args.gpu_list
use_amp = args.use_amp
batch_size = args.batch_size

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GrayDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L')  # 'L' = grayscale
        if self.transform:
            image = self.transform(image)
        return image

class SEBlock(nn.Module):
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

# Define DUNet
class DUNet(torch.nn.Module):
    """(4) BN+CAdaIN2 (AIO): Two parameters of AdaIN are generated by two different fully connected layers 
    with physical parameters (such as p)."""
    def __init__(self, LayerNo):  # layer_num
        super(DUNet, self).__init__()
        self.step_size = nn.Parameter(0.5*torch.ones(LayerNo))
        onelayer = []
        self.LayerNo = LayerNo
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

# Building the training dataset
transform = transforms.Compose([transforms.ToTensor()])           # [H,W] → [1,H,W], values in [0,1]
dataset = GrayDataset(image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=4)

# Building the model
model = DUNet(layer_num).to(device)
model = nn.DataParallel(model)

# Setting lr_scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=end_epoch, eta_min=1e-6)
scaler = torch.cuda.amp.GradScaler()

# Setting the path to the saved model
model_name = f"BN+CAdaIN2(AIO)_{end_epoch}_layer_{layer_num}"
model_dir = os.path.join(args.model_dir, model_name)
log_file_name = os.path.join(args.log_dir, f"Log_{model_name}.txt")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if start_epoch > 0:
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (pre_model_dir, start_epoch)))


# Training loop
if __name__ == "__main__":
    for epoch_i in range(start_epoch+1, end_epoch+1):
        for batch_i, batch in enumerate(dataloader):
            batch_x = batch.squeeze(1).to(device)
            frft_order = torch.FloatTensor([np.random.uniform(0.000000, 0.900000)]).to(device)
            frft_order = (int(frft_order * 10) + 1) / 10.0
            frft_order = torch.tensor(np.array([frft_order]), dtype=torch.float32).to(device)
            z = FrFT_forward(batch_x, frft_order)
            FrFT_abs = torch.abs(z)
            Rawdata = FrFT_abs ** 2
            Y = torch.clamp(Rawdata, min=0.0)
            measurement = torch.sqrt(Y)

            optimizer.zero_grad()
            x_output = model(measurement, frft_order)
            FrFT_abs_hat = torch.abs(FrFT_forward(x_output, frft_order))
            loss_measurement = torch.mean(torch.pow(FrFT_abs_hat - measurement, 2))
            loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))
            loss_all = loss_discrepancy

            # use amp to accelerate training process
            if use_amp:
                with autocast():
                    scaler.scale(loss_all).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                loss_all.backward()
                optimizer.step()

            scheduler.step()

            output_data = "[%02d/%02d] Total Loss: %.4f, Measurement Loss: %.4f\n" % (epoch_i, end_epoch, loss_all.item(), loss_measurement.item() )
            print(output_data)

        output_file = open(log_file_name, 'a')
        output_file.write(output_data)
        output_file.close()

        # Saving a set of model parameters every 10 epochs
        if epoch_i % 10 == 0:
            torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters
