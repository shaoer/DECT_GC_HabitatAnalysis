{\rtf1\ansi\ansicpg936\cocoartf2818
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Load data and preprocess\
def load_data(file_path):\
    print("Loading data...")\
    try:\
        data = pd.read_csv(file_path)\
    except Exception as e:\
        print(f"Error loading data: \{e\}")\
        return None, None\
    feature_columns = data.columns[3:]\
    scaler = StandardScaler()\
    data[feature_columns] = scaler.fit_transform(data[feature_columns])\
    return data, feature_columns\
# Define dataset class\
class PatientDataset(Dataset):\
    def __init__(self, data):\
        print("Initializing dataset...")\
        self.instances = []\
        for _, instance_data in data.iterrows():\
            instance = instance_data['MaskID']\
            features = torch.tensor(instance_data.filter(regex='^original_').values, dtype=torch.float32)\
            self.instances.append((instance_data['PatientID'], instance, features))\
\
    def __len__(self):\
        return len(self.instances)\
\
    def __getitem__(self, idx):\
        patient, instance, features = self.instances[idx]\
        return patient, instance, features\
}
