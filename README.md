# Multiple Disease Prediction System

## Overview
The **Multiple Disease Prediction System** is a machine learning-based project designed to predict the likelihood of a person having specific diseases based on input medical parameters. This system supports the prediction of:

- **Diabetes**
- **Heart Disease**
- **Parkinson's Disease**

## Features
- Uses separate models for different diseases.
- Predicts the probability of disease based on medical test parameters.
- Provides an easy-to-use interface for input and prediction.
- Implements data preprocessing, feature selection, and machine learning algorithms.

## Technologies Used
- **Programming Language:** Python
- **Frameworks & Libraries:**
  - NumPy
  - Pandas
  - Scikit-learn
  - Streamlit (for GUI)
- **Machine Learning Models:**
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - Decision Tree

## Project Files
- `Multiple disease prediction system - diabetes.ipynb` - Jupyter Notebook for diabetes prediction.
- `Multiple disease prediction system - heart.ipynb` - Jupyter Notebook for heart disease prediction.
- `Multiple disease prediction system - Parkinsons.ipynb` - Jupyter Notebook for Parkinson's disease prediction.
- `multiple_disease_pred.py` - Python script containing the combined model logic.

## How to Run the Project
1. **Clone the Repository**
   ```sh
   git clone (https://github.com/NiyatiPatel229/Multiple-Disease-Prediction-System)
   cd multiple-disease-prediction
   ```
2. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```
3. **Run Jupyter Notebook (For Model Training & Testing)**
   ```sh
   jupyter notebook
   ```
4. **Run the Python Script**
   ```sh
   streamlit run multiple_disease_prediction.py
   ```

## Dataset
The project uses datasets related to diabetes, heart disease, and Parkinson's disease. These datasets contain medical test results and are preprocessed before training.

## Model Performance
Each model is evaluated using accuracy, precision, recall, and F1-score. Hyperparameter tuning is applied to improve performance.

## Future Improvements
- Integrate a web-based UI using Flask/Streamlit.
- Extend support for more diseases.
- Optimize the models for better accuracy.

## Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests.

## License
This project is open-source and available under the MIT License.

