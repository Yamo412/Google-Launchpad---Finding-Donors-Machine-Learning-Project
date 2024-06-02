# Google Launchpad - Finding Donors Machine Learning Project

This project is a part of the Google Launchpad program and aims to build a machine learning model to predict potential donors for CharityML. The goal is to maximize the donation yield while minimizing the number of letters sent.

## Project Overview

CharityML is a fictional charity organization that relies on donations to support its mission. The organization wants to optimize its donation solicitation efforts by targeting individuals who are more likely to donate. This project involves building and evaluating several supervised learning models to identify potential donors based on the provided census data.

## Files and Directories

- `finding_donors.ipynb`: The Jupyter Notebook containing the complete analysis, data preprocessing, model training, and evaluation.
- `visuals.py`: A Python script with visualization functions used in the notebook.
- `census.csv`: The dataset containing census data used for training and evaluating the models.

## Installation

To run this project, you will need Python 3.x and the following libraries:

- NumPy
- pandas
- scikit-learn
- Matplotlib
- IPython

You can install these dependencies using `pip`:

```bash
pip install numpy pandas scikit-learn matplotlib ipython
```

## Usage

- 1 Clone the repository: ```git clone https://github.com/yourusername/google-launchpad-finding-donors.git
cd google-launchpad-finding-donors```

- 2 Open the Jupyter Notebook: ```jupyter notebook finding_donors.ipynb```

- 3 Follow the steps in the notebook to run the analysis and model evaluations.

## Project Structure

- Data Preprocessing:
  - Data Exploration: Understanding the dataset and identifying the features and target variable.
  - Data Cleaning: Handling missing values and converting categorical variables into numerical values using one-hot encoding.
  - Feature Scaling: Applying logarithmic transformation to skewed continuous features and scaling the features.
    
- Model Training and Evaluation:
  - Model Selection: Choosing several supervised learning algorithms for evaluation (e.g., Decision Trees, SVM, AdaBoost).
  - Model Evaluation: Using metrics such as accuracy, F1-score, and training/prediction time to evaluate the models.
  - Hyperparameter Tuning: Optimizing the models using techniques like Grid Search with cross-validation.
 
- Visualization:
  - The visuals.py script contains functions to visualize the distribution of features, model performance, and feature importance. The visualizations are integrated into the Jupyter Notebook to provide   insights into the data and model performance.

- Results:
  - The results of the model evaluations are presented in the notebook, along with the final model selection and feature importance analysis.

- Conclusion:
  - This project demonstrates the process of building and evaluating machine learning models to solve a real-world problem. By following the steps in this project, you will gain insights into data preprocessing, model selection, and evaluation techniques.
