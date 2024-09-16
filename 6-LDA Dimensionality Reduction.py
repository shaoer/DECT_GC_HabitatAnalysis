{\rtf1\ansi\ansicpg936\cocoartf2818
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Define function to compute AUC confidence intervals\
def bootstrap_auc(y_true, y_scores, n_bootstraps=2000, alpha=0.95):\
    rng = np.random.RandomState(RANDOM_SEED)\
    bootstrapped_scores = []\
    for _ in range(n_bootstraps):\
        indices = rng.randint(0, len(y_scores), len(y_scores))\
        if len(np.unique(y_true[indices])) < 2:\
            continue\
        score = roc_auc_score(y_true[indices], y_scores[indices])\
        bootstrapped_scores.append(score)\
    sorted_scores = np.array(bootstrapped_scores)\
    sorted_scores.sort()\
    confidence_lower = sorted_scores[int((1 - alpha) / 2 * len(sorted_scores))]\
    confidence_upper = sorted_scores[int((1 + alpha) / 2 * len(sorted_scores))]\
    return np.mean(sorted_scores), confidence_lower, confidence_upper\
\
# Define functions to compute specificity and negative predictive value\
def specificity_score(y_true, y_pred):\
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()\
    return tn / (tn + fp)\
\
def negative_predictive_value(y_true, y_pred):\
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()\
    return tn / (tn + fn) if (tn + fn) != 0 else 0\
\
}
