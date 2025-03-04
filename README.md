# Potential Customer Conversion Prediction Using Decision Trees and Random Forests

Developed machine learning models to predict lead conversions for an EdTech company, using Decision Trees and Random Forests to prioritize high-potential customers and reduce misclassification, enabling more efficient resource allocation in marketing and sales strategies.

## Project Overview

This project focuses on predicting which leads are most likely to convert to paying customers for an online education startup. By analyzing user behaviors and attributes, such as profile completion, website activity, and referral sources, the models support data-driven decision-making to optimize outreach efforts and increase conversion rates.

## Dataset

The **ExtraaLearn leads dataset** contains:
- Lead demographic information.
- Engagement behaviors (website visits, time spent, page views).
- Marketing interactions (email, phone, digital ads).
- Conversion status (whether the lead became a paying customer).

Key features include:
- Age
- Occupation (Professional, Student, Unemployed)
- First interaction channel (Website, Mobile App)
- Profile completion percentage
- Website activity metrics (visits, time spent, page views)
- Marketing exposure (print media, digital media, referrals)
- Last activity type (Email, Phone, Website)
- Conversion status (target variable)

## Objectives

- Analyze lead behavior to identify key conversion drivers.
- Build predictive models to classify leads as likely to convert or not.
- Reduce misclassification of high-value leads.
- Provide business recommendations for targeted marketing strategies.
- Support operational resource allocation by scoring leads for follow-up prioritization.

## Methods

### Data Preprocessing:
- Handled categorical encoding with **OneHotEncoder** and **LabelEncoder**.
- Normalized numerical features using **MinMaxScaler**.
- Performed exploratory data analysis (EDA) to identify trends and feature importance.
- Split data into training and test sets for model evaluation.

### Model Development:
Two machine learning models were developed and tuned:

- **Decision Tree Classifier**:
  - Tuned with grid search over **max depth**, **leaf nodes**, and **criterion**.
  - Identified overfitting through early plateauing of test accuracy and performance drop-offs.
  - Achieved **81% test accuracy**.

- **Random Forest Classifier**:
  - Tuned with grid search over **n_estimators**, **max depth**, **sample weights**, and **feature selection**.
  - Applied **class balancing** to address skewed conversion rates.
  - Reduced overfitting and improved generalization with **86% test accuracy**.

### Evaluation:
- Used **classification reports**, **confusion matrices**, and **ROC curves** to assess model performance.
- Identified key misclassification patterns, especially false positives and false negatives in conversion predictions.
- Measured improvements in recall and precision for the positive (conversion) class.

## Results

- **Random Forest model achieved ~86% test accuracy**, outperforming the baseline Decision Tree.
- Misclassification analysis revealed:
  - High confusion between actual converters and non-converters in the Decision Tree model.
  - Random Forest significantly reduced false negatives, improving lead targeting.
- Feature importance indicated that leads with **high profile completion**, **frequent website visits**, and **referral-based sources** had higher conversion likelihoods.

## Business/Scientific Impact

- Enabled data-driven lead scoring to prioritize follow-ups and allocate marketing resources efficiently.
- Recommended focusing marketing campaigns on high-conversion profiles, particularly leads engaging through educational channels and referrals.
- Advised periodic retraining of the Random Forest model to adapt to changing lead behaviors over time.
- Provided insights to improve data collection around engagement features that influence conversion.

## Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- GridSearchCV

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/potential-customer-conversion-prediction.git
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

4. Open and run the notebook to:
   - Load and preprocess the dataset.
   - Train and tune Decision Tree and Random Forest models.
   - Evaluate model performance.
   - Analyze misclassification patterns and feature importance.

## Future Work

- Explore advanced ensemble methods like **Gradient Boosting** or **XGBoost** for improved performance.
- Introduce **SMOTE** or other techniques to handle class imbalance more effectively.
- Implement real-time lead scoring in a production environment.
- Integrate additional behavioral data from customer interactions for deeper insights.
- Extend the model to predict customer lifetime value (CLV) alongside conversion.
