{\rtf1\ansi\ansicpg936\cocoartf2818
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Define Attention Mechanism Model\
class AttentionMIL(nn.Module):\
    def __init__(self, feature_dim=41):  # Modify feature dimension to match your data\
        super(AttentionMIL, self).__init__()\
        self.attention = nn.Sequential(\
            nn.Linear(feature_dim, 128),\
            nn.ELU(),\
            nn.Dropout(0.6),\
            nn.Linear(128, feature_dim),\
            nn.Softmax(dim=-1)\
        )\
\
        # Apply He initialization to all linear layers\
        for m in self.modules():\
            if isinstance(m, nn.Linear):\
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')\
                if m.bias is not None:\
                    init.constant_(m.bias, 0)\
\
    def forward(self, bag):\
        attention_weights = self.attention(bag)\
        return attention_weights\
}
