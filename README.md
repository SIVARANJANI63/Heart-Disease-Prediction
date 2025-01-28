# Heart Disease Prediction Using SVM

This project demonstrates a Machine Learning pipeline to predict heart disease using a Support Vector Machine (SVM). The dataset used is the widely available `heart.csv`, which contains features relevant to heart disease prediction. The pipeline incorporates preprocessing, feature scaling, handling class imbalance, and hyperparameter tuning.

---

## **Project Overview**

### **Features in Dataset:**
- **age**: Age of the individual
- **sex**: Gender (0 = female, 1 = male)
- **cp**: Chest pain type (4 values)
- **trestbps**: Resting blood pressure
- **chol**: Serum cholesterol in mg/dl
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- **restecg**: Resting electrocardiographic results (values 0, 1, 2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise-induced angina (1 = yes; 0 = no)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: The slope of the peak exercise ST segment
- **ca**: Number of major vessels (0-3) colored by fluoroscopy
- **thal**: 0 = normal; 1 = fixed defect; 2 = reversible defect
- **target**: 1 = presence of heart disease; 0 = absence

### **Pipeline Steps:**
1. **Loading Data**: The dataset is loaded from `heart.csv`.
2. **Handling Missing Values**: Missing values are imputed using the mean strategy.
3. **Encoding Categorical Variables**: Categorical features are one-hot encoded.
4. **Class Imbalance Handling**: Synthetic Minority Oversampling Technique (SMOTE) is applied to balance the target classes.
5. **Feature Scaling**: MinMaxScaler is used to normalize feature values.
6. **Model Training**: Support Vector Machine (SVM) with RBF kernel is trained.
7. **Hyperparameter Tuning**: GridSearchCV is used to optimize hyperparameters.
8. **Evaluation**: Model performance is assessed using accuracy and classification metrics.

---

## **Dependencies**

The following Python libraries are required:

- **pandas**: For data manipulation.
- **scikit-learn**: For machine learning algorithms and preprocessing.
- **imbalanced-learn**: For handling class imbalance (SMOTE).
- **numpy**: For numerical computations.

To install all dependencies, run:
```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
numpy
pandas
scikit-learn
imbalanced-learn
```

---

## **How to Run**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/HeartDiseasePrediction.git
   cd HeartDiseasePrediction
   ```

2. Ensure the `heart.csv` dataset is placed in the project directory.

3. Run the script:
   ```bash
   python train_model.py
   ```

---

## **Code Explanation**
### **`train_model.py`**
The core script that includes the pipeline implementation.

- **Loading and Preprocessing Data**:
  - Missing values are handled using `SimpleImputer`.
  - Categorical variables (`sex`, `cp`, `restecg`, `exang`, `slope`, `thal`) are one-hot encoded.

- **Class Imbalance Handling**:
  - SMOTE is applied to generate synthetic samples for the minority class.

- **Feature Scaling**:
  - Features are scaled between 0 and 1 using `MinMaxScaler`.

- **SVM Training and Hyperparameter Tuning**:
  - `GridSearchCV` performs 5-fold cross-validation to tune hyperparameters (`C`, `gamma`, `tol`, and `max_iter`).

- **Model Evaluation**:
  - Metrics such as accuracy and classification report are used for evaluation.

---

## **Results**
- The best hyperparameters for the SVM model are printed after `GridSearchCV` tuning.
- The final model's accuracy and classification report are displayed for the test dataset.

---

## **Project Structure**
```
HeartDiseasePrediction/
├── heart.csv                 # Dataset
├── train_model.py            # Main Python script for training
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
```

---

## **Future Enhancements**
- Add support for other machine learning models (e.g., Random Forest, Logistic Regression).
- Use advanced feature selection techniques to improve accuracy.
- Create a web application for user interaction (e.g., using Streamlit).

---

## **References**
- [SMOTE Documentation](https://imbalanced-learn.org/stable/over_sampling.html#smote-variants)
- [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)
- [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
