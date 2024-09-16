{\rtf1\ansi\ansicpg936\cocoartf2818
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Main function\
def main():\
    # User-configurable paths\
    data_path = 'path/to/your/data.csv'\
    save_path_attention_values = 'path/to/save/attention_values'\
\
    data, feature_columns = load_data(data_path)\
    if data is None or feature_columns is None:\
        return\
\
    dataset = PatientDataset(data)\
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)\
\
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\
    model = AttentionMIL(feature_dim=len(feature_columns)).to(device)\
\
    model.eval()\
    weighted_attention_records = []\
\
    # Preprocess with MinMaxScaler\
    scaler = MinMaxScaler()\
    all_features = data.filter(regex='^original_').values\
    scaler.fit(all_features)\
\
    with torch.no_grad():\
        for patient, instance, features in dataloader:\
            features = features.cpu().numpy()\
            features = scaler.transform(features)\
            features = torch.tensor(features, dtype=torch.float32).to(device)\
\
            attention_weights = model(features)\
\
            # Compute weighted attention value\
            weighted_attention_value = torch.sum(attention_weights * features) / torch.sum(attention_weights)\
            weighted_attention_records.append((patient.item(), instance.item(), weighted_attention_value.item()))\
\
    try:\
        create_folder(save_path_attention_values)\
        with open(os.path.join(save_path_attention_values, 'weighted_attention_values.csv'), mode='w', newline='') as file:\
            writer = csv.writer(file)\
            writer.writerow(['Patient ID', 'Instance ID', 'Weighted Attention Value'])\
            for record in weighted_attention_records:\
                writer.writerow(record)\
        print("Weighted attention values have been saved to 'weighted_attention_values.csv'")\
    except Exception as e:\
        print(f"Error saving results: \{e\}")\
\
if __name__ == '__main__':\
    main()\
}