import torch
import torch.nn as nn
import numpy as np
import os
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from argparse import ArgumentParser
from Forward_functions_v2 import FrFT_forward
from Forward_functions_v2 import adaptive_instance_normalization as adain
from torch.cuda.amp import autocast as autocast

parser = ArgumentParser(description='LFM_DPPAN')

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

class ProxNet(nn.Module):
    """CPNet + CSENet"""
    def __init__(self, in_nc=1, feature=64, kernel_size=3, padding=1):
        super(ProxNet, self).__init__()
        feature1 = 256         # 64*4
        channels = feature
        reduction_ratio = 16

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

        self.conv_layer1 = nn.Conv2d(in_channels=in_nc, out_channels=feature, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv_layer2 = nn.Conv2d(in_channels=feature, out_channels=feature, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv_layer3 = nn.Conv2d(in_channels=feature, out_channels=in_nc, kernel_size=kernel_size, padding=padding, bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fcw1 = nn.Linear(in_nc, feature1)   # 1 - 256
        self.fcb1 = nn.Linear(in_nc, channels//reduction_ratio)   # 1 - 4
        self.fcw2 = nn.Linear(in_nc, feature1)   # 1 - 256
        self.fcb2 = nn.Linear(in_nc, feature)      # 1 - 64
        self.fcw11 = nn.Linear(feature1, feature1)
        self.fcw21 = nn.Linear(feature1, feature1)

        nn.init.xavier_normal_(self.fcw1.weight)
        nn.init.zeros_(self.fcw1.bias)
        nn.init.xavier_normal_(self.fcb1.weight)
        nn.init.zeros_(self.fcb1.bias)
        nn.init.xavier_normal_(self.fcw2.weight)
        nn.init.zeros_(self.fcw2.bias)
        nn.init.xavier_normal_(self.fcb2.weight)
        nn.init.zeros_(self.fcb2.bias)
        nn.init.xavier_normal_(self.fcw11.weight)
        nn.init.zeros_(self.fcw11.bias)
        nn.init.xavier_normal_(self.fcw21.weight)
        nn.init.zeros_(self.fcw21.bias)

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

        Linear1_weight = self.fcw11(self.relu(self.fcw1(x_param))).reshape(4, 64)
        Linear1_bias = self.fcb1(x_param).reshape(4)
        Linear2_weight = self.fcw21(self.relu(self.fcw2(x_param))).reshape(64, 4)
        Linear2_bias = self.fcb2(x_param).reshape(64)

        b, c, _, _ = output4.size()
        output4_mid = self.avg_pool(output4).view(b, c)
        a1 = torch.nn.functional.linear(output4_mid, Linear1_weight, Linear1_bias)
        a2 = self.relu(a1)
        a3 = torch.nn.functional.linear(a2, Linear2_weight, Linear2_bias)
        output5 = torch.nn.Sigmoid()(a3).view(b, c, 1, 1)
        output5_fin = output4 * output5.expand_as(output4)

        output6 = self.conv_layer3(output5_fin)
        x_pred = output6.squeeze(1)

        return x_pred + x_img

class DPPAN(torch.nn.Module):
    """(5)DPPAN: The SENet is replaced by CSENet, where the linear layers are determined by other fully connected layers according to physical parameters (such as p)."""
    def __init__(self, LayerNo):  # layer_num
        super(DPPAN, self).__init__()
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
model = DPPAN(layer_num).to(device)
model = nn.DataParallel(model)

# Setting lr_scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=end_epoch, eta_min=1e-6)
scaler = torch.cuda.amp.GradScaler()

# Setting the path to the saved model
model_name = f"DPPAN_{end_epoch}_layer_{layer_num}"
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