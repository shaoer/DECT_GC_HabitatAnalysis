{\rtf1\ansi\ansicpg936\cocoartf2818
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import torch\
from torch import nn\
import torch.nn.init as init\
from torch.utils.data import DataLoader, Dataset\
import pandas as pd\
from sklearn.preprocessing import StandardScaler, MinMaxScaler\
import numpy as np\
import csv\
import os\
\
# Set random seeds to ensure reproducibility\
torch.manual_seed(42)\
np.random.seed(42)\
# Create folder if it doesn't exist\
def create_folder(folder_path):\
    if not os.path.exists(folder_path):\
        os.makedirs(folder_path)\
}
