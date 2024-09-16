# DECT_GC_HabitatAnalysis
Here's a complete README written in English based on your code:

---

# Attention Mechanism Model and Machine Learning Model Evaluation

## Introduction

This project consists of two main components:

1. **Attention Mechanism Model**: An implementation of an attention mechanism model using PyTorch to process medical imaging data. The model computes attention weights over input features and generates weighted attention values for further analysis.

2. **Machine Learning Model Evaluation**: Training and evaluation of various machine learning models (Logistic Regression, Support Vector Machine, Random Forest, Gradient Boosting, and K-Nearest Neighbors). The models are optimized using cross-validation and grid search, and various performance metrics are calculated.

## Table of Contents

- [Attention Mechanism Model and Machine Learning Model Evaluation](#attention-mechanism-model-and-machine-learning-model-evaluation)
  - [Introduction](#introduction)
  - [Table of Contents](#table-of-contents)
  - [Installation and Usage](#installation-and-usage)
    - [Prerequisites](#prerequisites)
    - [Installation Steps](#installation-steps)
    - [Usage](#usage)
      - [1. Attention Mechanism Model](#1-attention-mechanism-model)
      - [2. Machine Learning Model Evaluation](#2-machine-learning-model-evaluation)
  - [Project Structure](#project-structure)
  - [License](#license)
  - [Authors](#authors)
  - [Acknowledgments](#acknowledgments)

## Installation and Usage

### Prerequisites

- Python 3.x
- PyTorch
- NumPy
- Pandas
- scikit-learn

### Installation Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/your_username/your_repository.git
   ```

2. **Navigate to the project directory**

   ```bash
   cd your_repository
   ```

3. **Create a virtual environment (optional)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate     # For Windows
   ```

4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *Ensure that you have a `requirements.txt` file in the project root directory, listing all the required packages.*

### Usage

#### 1. Attention Mechanism Model

- **Data Preparation**

  - Place your data file in the appropriate directory.
  - Update the `data_path` variable in `attention_model_main.py` to point to your data file.

- **Run the main script**

  ```bash
  python attention_model_main.py
  ```

- **Output**

  - A `weighted_attention_values.csv` file containing the weighted attention values will be generated and saved in the specified output directory.

#### 2. Machine Learning Model Evaluation

- **Data Preparation**

  - Place your data file in the appropriate directory.
  - Update the `data_path` variable in `model_evaluation_main.py` to point to your data file.

- **Run the main script**

  ```bash
  python model_evaluation_main.py
  ```

- **Output**

  - Evaluation results CSV files and confusion matrices for each model will be generated and saved in the specified results directory.

## Project Structure

```
your_repository/
├── README.md
├── requirements.txt
├── attention_model_main.py
├── attention_model/
│   ├── dataset.py
│   ├── attention_model.py
│   └── utils.py
├── model_evaluation_main.py
└── model_evaluation/
    ├── data_preprocessing.py
    ├── evaluate_model.py
    ├── metrics.py
    └── utils.py
```

- **attention_model_main.py**: Main script for the attention mechanism model.
- **attention_model/**: Contains modules and functions related to the attention model.
  - **data_loading.py**: Functions for data loading and preprocessing.
  - **dataset.py**: Custom dataset class definitions.
  - **attention_model.py**: Model architecture and definitions.
  - **utils.py**: Utility functions, such as folder creation.
- **model_evaluation_main.py**: Main script for machine learning model evaluation.
- **model_evaluation/**: Contains modules and functions related to model evaluation.
  - **data_preprocessing.py**: Data preprocessing functions, including standardization and dimensionality reduction.
  - **evaluate_model.py**: Model training and evaluation functions.
  - **metrics.py**: Functions to compute evaluation metrics.
  - **utils.py**: Utility functions, such as saving results.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Your Name** - [your_email@example.com](mailto:your_email@example.com)

## Acknowledgments

- Thanks to [OpenAI](https://www.openai.com/) for providing resources and support.
- Inspiration from related open-source projects and literature.

---

**Note**:

- **Data Path Configuration**: Please update the `data_path` variable in the code to point to the location of your data files.
- **Dependencies**: Ensure all necessary Python packages are installed. You can install them via the `requirements.txt` file.
- **Virtual Environment**: It's recommended to use a virtual environment to manage project dependencies and avoid conflicts with other projects.

**Contact**: If you have any questions or suggestions, feel free to contact me via email.

---

Here is an example of what the `requirements.txt` file might look like:

```
torch
numpy
pandas
scikit-learn
matplotlib
seaborn
```

Please adjust it according to the actual packages and versions you are using.

---

**Additional Information**:

- **Data Files**: Make sure your data files are in the correct format and contain the necessary columns expected by the scripts.
- **Model Customization**: You can modify the model architectures and parameters in the code to suit your specific needs.
- **Results Interpretation**: The output CSV files will contain detailed metrics for each model, which can be used to compare performance.

---

Feel free to customize the README further to include any other information specific to your project.
