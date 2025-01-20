#%%
import torch
import torch.nn as nn
import numpy as np
import xarray as xr

import cv2
import datetime
import os
import random
import time
import matplotlib.pyplot as plt

from models.model_bayesian import BayesianUNetPP, BayesianUNetPP_3Layers, BayesianUNetPP_2Layers
from models.model_plusplus2_dropout import NestedUNet2 as DropoutUNetpp
from models.model_plusplus2_mu_sigma import NestedUNet2 as UNetPPDeepEnsemble
from models.calibrated_regression import CalibratedClimateModel
from torch.utils.data import Dataset
from train_utils import *
from evaluators.utils import *
from args_parser import get_args

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import warnings

from PIL import Image
import logging

warnings.filterwarnings("ignore")

# zamana bağlı ayır, 

import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

def save_model(state_dict, train_losses, val_losses, epoch):
    torch.save(
        {
            "state_dict": state_dict,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "epoch": epoch,
        },
        os.path.join(save_path, "best.pth"),
    )


def save_plot(train_losses, val_losses, loss_type):

    _, ax = plt.subplots()

    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    plt.savefig(os.path.join(loss_path, "loss_" + loss_type + ".png"), dpi=300)
    plt.cla()

def save_model_ens(state_dict, train_losses, val_losses, epoch, ens_num):
    torch.save(
        {
            "state_dict": state_dict,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "epoch": epoch,
        },
        os.path.join(save_path, "best_" + ens_num + ".pth"),
    )


def save_plot_ens(train_losses, val_losses, loss_type, ens_num):

    _, ax = plt.subplots()

    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    plt.savefig(os.path.join(loss_path, "loss_" + ens_num + ".png"), dpi=300)
    plt.cla()

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

PE = generate_PE(288,192)
print("Generated PE shape:", PE.shape)
plot_freq = 1
prereport_freq = 100

# From the user take the years and months as integers with argparse
parser = get_args()

seed_everything(parser.seed)

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

days = parser.days
years = parser.years
months = parser.months
batch_size = parser.batch_size
epochs = parser.epochs
device_name = parser.device
prediction_month = parser.prediction_month
positional_encoding = parser.positional_encoding
template_path = parser.template_path
model_name = parser.model
coef = parser.kl_coef
lr = parser.lr
weight_decay = parser.weight_decay
num_networks = parser.ensembles
best_num = parser.best_num
NUM_SAMPLING = 100 # bayesianda kaç tane üzerinden sample alalım bu parametre
model_name_for_ensemble = parser.model_name
template_path = THIS_FOLDER + "/" + template_path
main_path = template_path

loss_path = os.path.join(main_path, "plots/CIMP6_year-%d-month-%d-pred-%d" % (years, months, prediction_month))
save_path = os.path.join(main_path, "weights/CIMP6_year-%d-month-%d-+%d" % (years, months, prediction_month))

# Create directories if they don't exist
if not os.path.exists(loss_path):
    os.makedirs(loss_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

else:
    print("DIRECTORY ALREADY EXISTS, continuing")
    time.sleep(1)

######## DATA #################################################################
# main_path_cesm = "/mnt/data/CMIP6_REGRID/Amon/tas/tas.nc"

# Reading TAS data, filtering date and model
main_path_cesm = "../../../tas.nc"
vars_handle = xr.open_dataset(os.path.join(THIS_FOLDER, main_path_cesm))
vars_handle = vars_handle.sel(model="ACCESS-CM2", time=slice("2000-01-01", "2014-12-31"))

# Reading ACCESS daily data 
access_daily_data_path = "../../../tas_day_ACCESS-CM2_historical_r10i1p1f1_gn_20000101-20141231.nc"
access_daily_handle = xr.open_dataset(os.path.join(THIS_FOLDER, access_daily_data_path))
access_daily_handle = access_daily_handle.sel(time=slice("2000-01-01", "2014-12-31"))

# Load longitude and latitude
altitude_map = np.load(f'{THIS_FOLDER}/../info/meritdem_aligned_image.npy') / 10 #altitude yüklüyor
# lon = np.array(vars_handle["lon"])
# lat = np.array(vars_handle["lat"])
lon = np.array(access_daily_handle["lon"])
lat = np.array(access_daily_handle["lat"])

# Create a grid of longitude and latitude
lon_grid, lat_grid = np.meshgrid(lon, lat)
llgrid = np.concatenate(
    (np.expand_dims(lon_grid, axis=0), np.expand_dims(lat_grid, axis=0)),
    axis=0,
)

# deniz altitude ufak farklılıklar oluşmaması için sabitlenmiş
da = altitude_map # altitude mapini gride ekliyorux
sea = altitude_map[157]
da[160:] = sea


daily_shape = access_daily_handle["tas"][0].shape
da = cv2.resize(da, (daily_shape[1], daily_shape[0]))

# Get grid height and width
grid_height = da.shape[0]
grid_width = da.shape[1]

tas_mean = np.load(f"{THIS_FOLDER}/../statistics/"+parser.climate_var+"_mean.npy")
daily_tas_mean = cv2.resize(tas_mean, (daily_shape[1], daily_shape[0]))

tas_std = np.load(f"{THIS_FOLDER}/../statistics/"+parser.climate_var+"_std.npy")
daily_tas_std = cv2.resize(tas_std, (daily_shape[1], daily_shape[0]))

# Standardization parameters
attribute_norm_vals = {
    parser.climate_var: (
        daily_tas_mean,
        daily_tas_std,
    ),
}

###############################################################################

###### PREPARE DATA ###########################################################

# Select attributes
attribute_name = parser.climate_var
# which ensembles to use (cmip'in içinden hangileri)
ens_ids = np.arange(0, 1)

# Create template
# hangi ayları kullanacağımız ve periyodikliklerini ayarladığımız kısım
# +3, prediction, -3 month
# month_idxs = []
# for i in range(years): # 0, 1, 2
#     for j in range(months): # 0, 1
#         month_idxs.append(-12 * (i + 1) - (j + 1)) # -13, -14, -25, -26, -37, -38
#         month_idxs.append(-12 * (i + 1) + (j + 1)) # -11, -10, -23, -22, -35, -34
#     month_idxs.append(-12 * (i + 1)) # -12, -24, -36


day_idxs = list(range(-1, -days-11, -1))
# for i in range(years):
#     prev_years_day_idxs = list(range(-365*(i+1), -days - 365*(i+1), -1))
#     day_idxs.extend(prev_years_day_idxs)

day_idxs = sorted(day_idxs)


# current year's previous months (ocak ayı için month=2 ise, aralık - kasım aylarını da arraye ekliyoruz)
# for j in range(months):
#     month_idxs.append(-(j + 1)) # -1, -2

# print(month_idxs)

# (18, 288, 192) 18 -> Time, H, W
# month_idxs = sorted(month_idxs)
# input_template = np.array(month_idxs)
# input_template -= (prediction_month - 1)
# input_template = np.expand_dims(input_template, axis=0)

daily_input_template = np.array(day_idxs)
daily_input_template = np.expand_dims(daily_input_template, axis=0)

# Input and output size
# input_size = input_template.shape[1]
# output_size = 1  # Can change

daily_input_size = daily_input_template.shape[1]
output_size = 1  # Can change


# Eski değerler
# # CMIP6 için.
# # start_idx to validation_start_idx -> train set
# start_idx = 0
# validation_start_idx = 1700
# # validation_start_idx to end_idx -> validation set
# end_idx = 1800  # not inclusive

# --- MONTHLY DATA ---
# TODO: Fix these values
# start_idx = 0
# validation_start_idx = 134
# end_idx = 141


# --- DAILY DATA ---
# TODO: Fix these values
# daily_start_idx = 0
# daily_validation_start_idx = 4100
# daily_end_idx = 4350


daily_start_idx = 0
daily_validation_start_idx = 4383 # --> 5130
daily_end_idx = 4749 # --> 5460

# Create input and output tensors
print("Creating input and output tensors...")
# input_idx_tensor = torch.zeros((end_idx - start_idx, input_size), dtype=torch.long)
# month_idx_tensor = torch.arange(start_idx, end_idx).long()

# (1973 ocak 2016) (1975 ocak index=0 oluyor bizim datasetimizde)
# input_idx_tensor[month_idx_tensor - start_idx, :] = torch.tensor(
#     input_template
# ) + month_idx_tensor.unsqueeze(1)

# Create daily input and output tensors
daily_input_idx_tensor = torch.zeros((daily_end_idx - daily_start_idx, daily_input_size), dtype=torch.long)
daily_day_idx_tensor = torch.arange(daily_start_idx, daily_end_idx).long()

daily_input_idx_tensor[daily_day_idx_tensor - daily_start_idx, :] = torch.tensor(
    daily_input_template
) + daily_day_idx_tensor.unsqueeze(1)

print("Creating handle...")
# istediğimiz temperature attribute'unu çekiyorux
# handle = vars_handle[attribute_name]

# Load data to memory
print("Loading data to memory...")
# istediğimiz ensembleları da ekledik
# vars_data = np.array(handle[ens_ids, :, :, :])
# vars_data = np.array(handle)

access_daily_handle = access_daily_handle[attribute_name]
daily_vars_data = np.expand_dims(np.array(access_daily_handle), 0)

def get_month_index(day_idx):
    """
    Given a zero-based day index (counting from 1 Jan 2000 as day_idx=0),
    return the zero-based month index (0 for January, 1 for February, etc.).
    """
    start_date = datetime.date(2000, 1, 1)
    # Find the actual date corresponding to the day index
    current_date = start_date + datetime.timedelta(days=day_idx)
    # Return zero-based month index
    return 12 * (current_date.year - 2000) + (current_date.month - 1)

class Reader(Dataset):
    def __init__(self, split="train", calib=False):
        # --- MONTHLY DATA ---

        # sadece indexleri data gibi kullanarak asıl veriyi _getitem_'da bu seçtiğimiz indexlere göre yüklüyorux
        self.calib = calib
        # valid_months = np.where((input_idx_tensor < 0).sum(1) == 0)[0]

        # if split == "train":
        #     valid_months = valid_months[valid_months < validation_start_idx]
        #     #valid_months, calib_months = train_test_split(valid_months, test_size=0.2)
        # elif split == "val":
        #     valid_months = valid_months[valid_months >= validation_start_idx]

        # self.data_list = [
        #     (ens_idx, month_idx)
        #     for ens_idx in range(vars_data.shape[0])
        #     for month_idx in valid_months
        # ]
        # if split == "train":
        #     self.calib_data_list = [
        #         (ens_idx, month_idx)
        #         for ens_idx in range(vars_data.shape[0])
        #         for month_idx in valid_months
        #     ]

        # --- DAILY DATA ---
        valid_days = np.where((daily_input_idx_tensor < 0).sum(1) == 0)[0]

        if split == "train":
            valid_days = valid_days[valid_days < daily_validation_start_idx]
            #valid_days, calib_days = train_test_split(valid_days, test_size=0.2)
        elif split == "val":
            valid_days = valid_days[valid_days >= daily_validation_start_idx]

        self.daily_data_list = [
            (ens_idx, day_idx)
            for ens_idx in range(daily_vars_data.shape[0])
            for day_idx in valid_days
        ]
        if split == "train":
            self.daily_calib_data_list = self.daily_data_list


            
        # HISTOGRAM
        # all_values = []
        # for month_idx in valid_months:
        #     flat = (vars_data[0, month_idx, :, :].flatten() - attribute_norm_vals[attribute_name][0].flatten()) / attribute_norm_vals[attribute_name][1].flatten()
        #     all_values.append(flat)
        
    def __len__(self):
        # TODO: Yusuf Hocaya sor, len için hangisini almalıyız?
        # if self.calib == True:
        #     return len(self.calib_data_list)
        # else:
        #     return len(self.data_list)
        
        if self.calib == True:
            return len(self.daily_calib_data_list)
        else:
            return len(self.daily_data_list)

    def __getitem__(self, idx):
        # input idx is daily index, we need to calculate the month index
        # self.calib_data_list[idx] aslında şu tarzda bi veri [0, (1, 13, 14, 25, 26)] (ensemble index, month idleri), [3, (23, 36, 37, 45, 46)}
        if self.calib:
            (ens_idx, day_idx) = self.daily_calib_data_list[idx]
            # month_idx = get_month_index(day_idx)
        else:
            (ens_idx, day_idx) = self.daily_data_list[idx]
            # month_idx = get_month_index(day_idx)

        # if self.calib == True:
        #     (ens_idx, month_idx) = self.calib_data_list[idx]
        # else:
        #     (ens_idx, month_idx) = self.data_list[idx]

        daily_input_data = daily_vars_data[ens_idx, daily_input_idx_tensor[day_idx, :], :, :]
        daily_output_data = daily_vars_data[ens_idx, day_idx, :, :]

        # input_data = vars_data[ens_idx, input_idx_tensor[month_idx, :], :, :]
        # output_data = vars_data[ens_idx, month_idx, :, :]

        # Standerize data
        daily_input_data = (
            daily_input_data - attribute_norm_vals[attribute_name][0]
        ) / attribute_norm_vals[attribute_name][1]
        daily_output_data = (
            daily_output_data - attribute_norm_vals[attribute_name][0]
        ) / attribute_norm_vals[attribute_name][1]

        # input_data = (
        #     input_data - attribute_norm_vals[attribute_name][0]
        # ) / attribute_norm_vals[attribute_name][1]
        # output_data = (
        #     output_data - attribute_norm_vals[attribute_name][0]
        # ) / attribute_norm_vals[attribute_name][1]
        
        # print("Shape after normalization:", input_data.shape)

        # Add area array to input data
        if "lstmpcl" in model_name.lower():
            pass
        else:
            # decide to concatenate positional encoding
            # sil
            # if not model_name == "CMIP6-withoutDA" and not model_name == "CMIP6-UNet-Attention-withoutDA":
            #     # bu elevation verisi ile ilgili eklemek isterseniz
            #     input_data = np.concatenate((input_data, np.expand_dims(da, axis=0)), axis=0)
            
            # NOTE: Yemrenin ekledikleri

            daily_input_data = np.concatenate((daily_input_data, np.expand_dims(da, axis=0)), axis=0) # -> 1 channel
            daily_input_data = np.concatenate((daily_input_data, llgrid), axis=0) # -> 2 channels
            
            # input_data = np.concatenate((input_data, np.expand_dims(da, axis=0)), axis=0) # -> 1 channel
            # input_data = np.concatenate((input_data, llgrid), axis=0) # -> 2 channels

            # print("Should add PE:", positional_encoding)
            if positional_encoding:
                # bu positional encoding ile ilgili eklemek isterseniz
                # print("Shape before adding PE:", input_data.shape)
                input_data = np.concatenate((input_data, np.expand_dims(PE, axis=0)), axis=0)
                # print("Shape after adding PE:", input_data.shape)

        # Create tensors
        daily_input_data = torch.tensor(daily_input_data, dtype=torch.float32)
        daily_output_data = torch.tensor(daily_output_data, dtype=torch.float32)

        # input_data = torch.tensor(input_data, dtype=torch.float32)
        # # print("Final input tensor shape:", input_data.shape)
        # output_data = torch.tensor(output_data, dtype=torch.float32)

        # return daily_input_data, daily_output_data, input_data, output_data
        return daily_input_data, daily_output_data


# We use train and calibration dataset as the same dataset, then evaluate on validation
train_dataset = Reader(split="train", calib=False)
calib_dataset = Reader(split="train", calib=False)
val_dataset = Reader(split="val")

# DATA KISMI BURAYA KADAR

path = os.path.dirname(os.getcwd())

if parser.best_num == -1:
    model_weight_path = path + "/experiments/" + model_name + "/weights/CIMP6_year-%d-month-%d-+%d/best.pth" % (3, 2, 1)
else:
    model_weight_path = path + "/experiments/" + model_name + "/weights/CIMP6_year-%d-month-%d-+%d/best_%d.pth" % (3, 2, 1, best_num)

# FOR DEBUG 
# Before model initialization, add this debug print
# print("Input size:", input_size)
# print("Total channels after adding 1:", input_size + 1)
# print("Number of months in month_idxs:", len(month_idxs))

# print("Model input channels:", input_size + 1)

print("Input size:", daily_input_size)
print("Total channels after adding 1:", daily_input_size + 1)
print("Number of months in month_idxs:", len(day_idxs))

print("Model input channels:", daily_input_size + 1)

# pick one of the models for our experiments.
# as you can see, there is bayesian networks with different layer sizes, ensemble and dropout
if "2layers" in model_name.lower():
    model = BayesianUNetPP_2Layers(input_channels=input_size, num_classes=output_size, device=device_name).to(device_name)
if "3layers" in model_name.lower():
    # model = BayesianUNetPP_3Layers(input_channels=input_size + 2, num_classes=output_size, device=device_name).to(device_name)
    # model = BayesianUNetPP_3Layers(input_channels=input_size, num_classes=output_size, device=device_name).to(device_name)
    model = BayesianUNetPP_3Layers(input_channels=daily_input_size + 3, num_classes=output_size, device=device_name).to(device_name)
elif "bayesian" in model_name.lower():
    model = BayesianUNetPP(input_channels=input_size, num_classes=output_size, device=device_name).to(device_name)
elif "ensemble" in model_name.lower():
    networks = []
    for num in range(num_networks):
        networks.append(UNetPPDeepEnsemble(input_channels=input_size, num_classes=output_size, device=device_name).to(device_name))
else:
    print("Dropout")
    model = DropoutUNetpp(input_channels=input_size, num_classes=output_size, device=device_name).to(device_name)

#model.load_state_dict(torch.load(model_weight_path)["state_dict"])
# # Create loaders
# ENSEMBLE içn HER ZAMAN SHUFFLE YAP FARKLI BATCH
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=4
)
# print(len(train_loader))
is_validation = True
if is_validation:
    batch_size = 1
validation_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)

###############################################################################

###### Training functions #####################################################

# Train loops for bayesian models
def bayesian_train_epoch(model, optimizer, loader, criterion):
    model.train()
    KL_loss = 0
    MSE_loss = 0
    
    for i, (daily_input_data, daily_output_data) in enumerate(tqdm(loader)):
        daily_input_data = daily_input_data.to(device_name)
        daily_output_data = daily_output_data.to(device_name)

        # input_data = input_data.to(device_name)
        # output_data = output_data.to(device_name)

        output_pred = model(daily_input_data)
        output_pred = output_pred.reshape(-1, 144, 192)

        optimizer.zero_grad()
                
        #model.fcn.requires_grad_(False)
        # our custom conv los function
        # MOST OF THE TIMES MSE criterion
        conv_loss = criterion(output_pred, daily_output_data)
        #conv_loss.backward(retain_graph=True)
        # coef = KL LOSS CONTRIBUTON COEFFICIENT, MSE İLE DENGESİNİ SAĞLAMAK İÇİN, tercih
        # 16 batch size ama hard coded verildi değişmesi gerek
        bayes_loss = (1.0/coef)*model.KL_loss() / 16
        loss = conv_loss + bayes_loss
        loss.backward()

        optimizer.step()

        MSE_loss += conv_loss.item() / 16
        KL_loss += bayes_loss.item() / 16

    return MSE_loss / len(loader), KL_loss / len(loader)


def bayesian_val_epoch(model, loader, criterion):

    model.eval()
    KL_loss = 0
    MSE_loss = 0

    for i, (input_data, output_data) in enumerate(tqdm(loader)):

        input_data = input_data.to(device_name)
        output_data = output_data.to(device_name)
        
        output_pred = model(input_data)

        output_pred = output_pred.reshape(-1, 144, 192)

        conv_loss = criterion(output_pred, output_data)
        bayes_loss = (1.0/coef)*model.KL_loss()

        MSE_loss += conv_loss.item()
        KL_loss += bayes_loss.item()

    return MSE_loss / len(loader), KL_loss / len(loader)

# Train loops for normal models
def train_epoch(model, optimizer, loader, criterion):
    model.train()
    total_loss = 0
    
    for i, (input_data, output_data) in enumerate(loader):

        input_data = input_data.to(device_name)
        output_data = output_data.to(device_name)
        # if "lstm" in model_name.lower() or "kamu" in model_name.lower():
        #     input_data = torch.unsqueeze(input_data, dim=2)
        output_pred = model(input_data)

        optimizer.zero_grad()

        # if model_name == "CMIP6-UNet-Attention-withoutDA" or "Attention" in model_name:
        #     output_pred = output_pred[0]
        # else:
        #     output_pred = output_pred.squeeze(1)
    
        # if "resnext" in model_name:
        #     if prediction_month == 1:
        #         output_pred = output_pred.reshape(-1, 192, 288)
        #     else:
        #         output_pred = output_pred.reshape(-1, prediction_month, 192, 288)
        # else:
        loss = criterion(output_pred, output_data)

        # loss.backward(retain_graph=True)
        loss.backward(retain_graph=False)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def val_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    for i, (input_data, output_data) in enumerate(loader):

        input_data = input_data.to(device_name)
        output_data = output_data.to(device_name)
        
        if "lstm" in model_name.lower():
            input_data = torch.unsqueeze(input_data, dim=2)

        output_pred = model(input_data)
        if model_name == "CMIP6-UNet-Attention-withoutDA" or "Attention" in model_name:
            output_pred = output_pred[0]
        else:
            output_pred = output_pred.squeeze(1)

        if "resnext" in model_name:
            if prediction_month == 1:
                output_pred = output_pred.reshape(-1, 192, 288)
            else:
                output_pred = output_pred.reshape(-1, prediction_month, 192, 288)
            
        loss = criterion(output_pred, output_data)

        total_loss += loss.item()

    return total_loss / len(loader)

# Train loops for ensemble models
def train_ensemble_epoch(model, optimizer, loader):
    print(model_name)
    model.train()
    total_loss = 0
    criterion = torch.nn.MSELoss()
    
    for i, (input_data, output_data) in enumerate(loader):

        input_data = input_data.to(device_name)
        output_data = output_data.to(device_name)

        mu, logvar = model(input_data)
        #output_pred = mu + random.uniform(0, 1)*
        # print(mu[0, :3, :3])
        # print(sigma[0, :3, :3])
        
        optimizer.zero_grad()
                
        #loss = nll_criterion2(sigma2, mu, output_data) + 2*criterion(output_pred, output_data)
        if "dynamics" not in model_name.lower():
            loss = nll_criterion(logvar, mu, output_data)
        else:
            loss = ensemble_dynamics_criterion(logvar, mu, output_data)
        # print(nll_criterion2(sigma2, mu, output_data), criterion(output_pred, output_data))
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        total_loss += loss.item()
    

    return total_loss / len(loader)


def val_ensemble_epoch(model, loader):
    model.eval()
    total_loss = 0
    criterion = torch.nn.MSELoss()
    for i, (input_data, output_data) in enumerate(loader):

        input_data = input_data.to(device_name)
        output_data = output_data.to(device_name)
        
        mu, logvar = model(input_data)
        # output_pred = mu + random.uniform(0, 0.1)*sigma2

        if "dynamics" not in model_name.lower():
                loss = nll_criterion(logvar, mu, output_data)
        else:
            loss = ensemble_dynamics_criterion(logvar, mu, output_data)
        total_loss += loss.item()

    return total_loss / len(loader)

def val_epoch_calibration_error(model, loader):

    model.eval()
    total_loss = 0
    preds = []
    targets = []
    for i, (input_data, output_data) in enumerate(loader):

        samples = []
        for i in range(100):
            input_data = input_data.to(device_name)
            output_data = output_data.to(device_name)
            
            output_pred = model(input_data)
            output_pred = output_pred.squeeze(1)
            samples.append(output_pred.detach().cpu().numpy())

        samples = np.mean(samples, axis=0)
        preds.append(samples)
        targets.append(output_data.detach().cpu().numpy())
    return np.asarray(preds), np.asarray(targets)

def val_epoch_calibration_error(model, loader):

    model.eval()
    total_loss = 0
    preds = []
    targets = []
    for i, (input_data, output_data) in enumerate(loader):

        samples = []
        for i in range(100):
            input_data = input_data.to(device_name)
            output_data = output_data.to(device_name)
            
            output_pred = model(input_data)
            output_pred = output_pred.squeeze(1)
            samples.append(output_pred.detach().cpu().numpy())

        samples = np.mean(samples, axis=0)
        preds.append(samples)
        targets.append(output_data.detach().cpu().numpy())
    return np.asarray(preds), np.asarray(targets)



###############################################################################

print("Model initializing: ", end="")
if positional_encoding:
    input_size += 1

# Create optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
r_schedual = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

best_model = None
best_loss = 99999

# kirli kod, repetition var
print("Training starting...")
if "bayesian" in model_name.lower():

    MSE_train_losses = []
    KL_train_losses = []
    MSE_val_losses = []
    KL_val_losses = []
    val_losses = []

    for epoch in range(epochs):
            
        start_time = time.time()

        MSE_train_loss, KL_train_loss = bayesian_train_epoch(model, optimizer, train_loader, criterion)

        with torch.no_grad():
            MSE_val_loss, KL_val_loss = bayesian_val_epoch(model, validation_loader, criterion)
        r_schedual.step()

        MSE_val_losses.append(MSE_val_loss)
        KL_val_losses.append(KL_val_loss)
        KL_train_losses.append(KL_train_loss)
        MSE_train_losses.append(MSE_train_loss)

        val_loss = MSE_val_loss+KL_val_loss
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            model = model.cpu()
            best_model = model.state_dict()
            model = model.to(device_name)
            save_model(best_model, MSE_train_loss, MSE_val_loss, epoch)

        print(np.asarray(KL_train_losses).shape)
        save_plot(KL_train_losses, KL_val_losses, "KL")
        save_plot(MSE_train_losses, MSE_val_losses, "MSE")

        report = "Epoch: %d, Train MSE Loss: %.6f, Train KL Loss: %.6f, Val MSE Loss: %.6f, Val KL Loss: %.6f, Time: %.2f" % (
                epoch, MSE_train_loss, KL_train_loss, MSE_val_loss, KL_val_loss, time.time() - start_time
            )
        print(report)

elif "ensemble" in model_name.lower():
    train_losses = np.zeros((num_networks,))
    val_losses = np.zeros((num_networks,))
    for j in range(best_num, num_networks):
        best_loss = 1000
        model = networks[j]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        r_schedual = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        for epoch in range(epochs):

            start_time = time.time()

            train_loss = train_ensemble_epoch(model, optimizer, train_loader)
            with torch.no_grad():
                val_loss = val_ensemble_epoch(model, validation_loader)
            r_schedual.step()

            if epoch == 0:
                train_losses[j] = train_loss
                val_losses[j] = val_loss
            else:
                np.append(train_losses[j], train_loss)
                np.append(val_losses[j], val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                model = model.cpu()
                best_model = model.state_dict()
                model = model.to(device_name)
                print("hello")
                save_model_ens(best_model, train_losses[j], val_losses[j], epoch, str(j))

            if epoch % plot_freq == 0:
                save_plot_ens(train_losses[j], val_losses[j], "NLL", str(j))

            report = "Ensemble: %d, Epoch: %d, Train Loss: %.6f, Val Loss: %.6f, Time: %.2f"% (
                    j, epoch, train_loss, val_loss, time.time() - start_time
                )

            f = open(template_path+"/output.txt", "a+")
            f.write(report+'\n')
            f.close()
            print(report)
else:
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        
        start_time = time.time()

        train_loss = train_epoch(model, optimizer, train_loader, criterion)
        with torch.no_grad():
            val_loss = val_epoch(model, validation_loader, criterion)
        r_schedual.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            model = model.cpu()
            best_model = model.state_dict()
            model = model.to(device_name)
            save_model(best_model, train_losses, val_losses, epoch)

        if epoch % plot_freq == 0:
            save_plot(train_losses, val_losses, "MSE")

        report = "Epoch: %d, Train Loss: %.6f, Val Loss: %.6f, Time: %.2f" % (
                epoch, train_loss, val_loss, time.time() - start_time
            )
        f = open(template_path+"/output.txt", "a+")
        f.write(report+'\n')
        f.close()
        print(report)

# %%
