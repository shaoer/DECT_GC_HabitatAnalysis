{\rtf1\ansi\ansicpg936\cocoartf2818
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Define model evaluation function\
def evaluate_model(cv, X, y, model, model_name, param_grid, seed):\
    fold_results = []\
    roc_auc_train_list = []\
    roc_auc_validation_list = []\
    train_confusion_matrices = []\
    validation_confusion_matrices = []\
\
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)\
    grid_search.fit(X, y)\
    best_model = grid_search.best_estimator_\
\
    for fold_index, (train_index, validation_index) in enumerate(cv.split(X, y)):\
        X_train, X_validation = X[train_index], X[validation_index]\
        y_train = y.iloc[train_index].reset_index(drop=True)\
        y_validation = y.iloc[validation_index].reset_index(drop=True)\
        \
        # Apply LDA for dimensionality reduction\
        X_train_reduced, lda, lda_coefficients = apply_lda(X_train, y_train)\
        X_validation_reduced = lda.transform(X_validation)\
        \
        best_model.fit(X_train_reduced, y_train)\
        y_train_pred = best_model.predict(X_train_reduced)\
        y_train_proba = (\
            best_model.predict_proba(X_train_reduced)[:, 1]\
            if hasattr(best_model, "predict_proba")\
            else best_model.decision_function(X_train_reduced)\
        )\
        y_validation_pred = best_model.predict(X_validation_reduced)\
        y_validation_proba = (\
            best_model.predict_proba(X_validation_reduced)[:, 1]\
            if hasattr(best_model, "predict_proba")\
            else best_model.decision_function(X_validation_reduced)\
        )\
        \
        train_auc, train_auc_lower, train_auc_upper = bootstrap_auc(y_train, y_train_proba)\
        train_accuracy = accuracy_score(y_train, y_train_pred)\
        train_recall = recall_score(y_train, y_train_pred)\
        train_precision = precision_score(y_train, y_train_pred)\
        train_f1 = f1_score(y_train, y_train_pred)\
        train_specificity = specificity_score(y_train, y_train_pred)\
        train_npv = negative_predictive_value(y_train, y_train_pred)\
        \
        try:\
            fpr_validation, tpr_validation, _ = roc_curve(y_validation, y_validation_proba)\
            validation_auc = auc(fpr_validation, tpr_validation)\
        except ValueError as e:\
            print(f"Warning: Error calculating ROC for fold \{fold_index+1\}: \{e\}")\
            validation_auc = 0\
        validation_auc, validation_auc_lower, validation_auc_upper = bootstrap_auc(y_validation, y_validation_proba)\
        validation_accuracy = accuracy_score(y_validation, y_validation_pred)\
        validation_recall = recall_score(y_validation, y_validation_pred)\
        validation_precision = precision_score(y_validation, y_validation_pred)\
        validation_f1 = f1_score(y_validation, y_validation_pred)\
        validation_specificity = specificity_score(y_validation, y_validation_pred)\
        validation_npv = negative_predictive_value(y_validation, y_validation_pred)\
        \
        fold_results.append(\{\
            'Model': model_name,\
            'Fold': fold_index + 1,\
            'Train AUC (95% CI)': f"\{train_auc:.3f\} (\{train_auc_lower:.3f\}-\{train_auc_upper:.3f\})",\
            'Train Accuracy': f"\{train_accuracy:.3f\}",\
            'Train Sensitivity': f"\{train_recall:.3f\}",\
            'Train Specificity': f"\{train_specificity:.3f\}",\
            'Train PPV': f"\{train_precision:.3f\}",\
            'Train NPV': f"\{train_npv:.3f\}",\
            'Train F1-score': f"\{train_f1:.3f\}",\
            'Validation AUC (95% CI)': f"\{validation_auc:.3f\} (\{validation_auc_lower:.3f\}-\{validation_auc_upper:.3f\})",\
            'Validation Accuracy': f"\{validation_accuracy:.3f\}",\
            'Validation Sensitivity': f"\{validation_recall:.3f\}",\
            'Validation Specificity': f"\{validation_specificity:.3f\}",\
            'Validation PPV': f"\{validation_precision:.3f\}",\
            'Validation NPV': f"\{validation_npv:.3f\}",\
            'Validation F1-score': f"\{validation_f1:.3f\}"\
        \})\
\
        # Confusion matrices\
        train_cm = confusion_matrix(y_train, y_train_pred)\
        validation_cm = confusion_matrix(y_validation, y_validation_pred)\
        train_confusion_matrices.append(train_cm)\
        validation_confusion_matrices.append(validation_cm)\
        \
        # Collect AUC scores for train and validation sets\
        roc_auc_train_list.append(train_auc)\
        roc_auc_validation_list.append(validation_auc)\
\
    avg_train_auc = np.mean(roc_auc_train_list)\
    avg_validation_auc = np.mean(roc_auc_validation_list) if roc_auc_validation_list else 0\
\
    return fold_results, avg_train_auc, avg_validation_auc, train_confusion_matrices, validation_confusion_matrices, lda_coefficients\
}
