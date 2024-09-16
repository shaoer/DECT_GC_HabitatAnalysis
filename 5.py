{\rtf1\ansi\ansicpg936\cocoartf2818
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import pandas as pd\
import numpy as np\
from sklearn.model_selection import KFold, GridSearchCV\
from sklearn.preprocessing import StandardScaler\
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\
from sklearn.linear_model import LogisticRegression\
from sklearn.svm import SVC\
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\
from sklearn.neighbors import KNeighborsClassifier\
from sklearn.metrics import (\
    roc_curve,\
    auc,\
    roc_auc_score,\
    accuracy_score,\
    recall_score,\
    precision_score,\
    f1_score,\
    confusion_matrix,\
)\
import warnings\
import os\
\
warnings.filterwarnings('ignore')\
\
RANDOM_SEED = 42\
# Read data\
data_path = 'path/to/your/data.csv'  # Update this path to your data file\
data = pd.read_csv(data_path)\
feature_columns = [col for col in data.columns if col.startswith('original')]\
X = data[feature_columns]\
y = data['group']\
\
# Standardize data\
scaler = StandardScaler()\
X_scaled = scaler.fit_transform(X)\
\
# LDA dimensionality reduction\
def apply_lda(X, y):\
    n_features = X.shape[1]\
    n_classes = len(np.unique(y))\
    lda_n_components = min(n_features, n_classes - 1)\
    lda = LinearDiscriminantAnalysis(n_components=lda_n_components)\
    lda_X = lda.fit_transform(X, y)\
    lda_coefficients = lda.coef_[0]\
    return lda_X, lda, lda_coefficients\
}