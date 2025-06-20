# Smart-Credit-Scoring-Risk-Assessment-System

Here’s a complete and clean README.md template for your Credit Default Prediction App using PCA, SMOTE, and RandomForest, deployed with Streamlit:

💳 Credit Default Risk Prediction App
This project is a machine learning-powered web application that predicts the probability of credit default based on customer financial data. It handles class imbalance using SMOTE, reduces feature space via PCA, and provides predictions using an ensemble RandomForest model.

🔍 Key Features
Trained on a real-world imbalanced dataset (~8% defaulters)

Uses Random Oversampling + PCA for robust prediction

Achieves 91% accuracy and 0.84 ROC AUC

Allows users to upload CSV data and view predicted default probabilities

Deployed using Streamlit for fast and interactive use

Includes SHAP explainability (optional)

Supports dynamic threshold tuning

🛠️ Technologies Used
Python 🐍

Scikit-learn

Pandas & NumPy

Streamlit

SHAP (optional)

Git LFS (for large models)

🧠 Machine Learning Pipeline
Data Preprocessing

Missing value imputation

Feature engineering (e.g., BalanceToIncome)

Feature Scaling + PCA

Scaled 12 original features using StandardScaler

Applied PCA to extract top 5 principal components

Handling Class Imbalance

Used SMOTE to oversample the minority class (defaults)

Model

RandomForestClassifier (ensemble model)

Trained on 12 scaled features + 5 PCA components (total 17 features)

Deployment

Built an interactive app using Streamlit

Upload CSV → Predict → Download results

🚀 How to Run the App Locally
Clone the repository

Install dependencies
pip install -r requirements.txt
Run the Streamlit app
streamlit run app.py
📁 File Structure
bash
Copy
Edit
├── streamlit_app.py                        # Streamlit app
├── credit_risk_model.pkl (7z(extract it))     # Trained RandomForest model
├── scaler_model.pkl             # Scaler used before PCA
├── pca_model.pkl                # PCA transformer (5 components)
├── requirements.txt             # Python dependencies
├── sample_input.csv             # Sample file to test the app
└── README.md                    # Project description
📥 Sample Input Format
Your input .csv file should include the following columns:

RevolvingUtilizationOfUnsecuredLines, age, NumberOfTime30-59DaysPastDueNotWorse,
DebtRatio, MonthlyIncome, NumberOfOpenCreditLinesAndLoans,
NumberOfTimes90DaysLate, NumberRealEstateLoansOrLines,
NumberOfTime60-89DaysPastDueNotWorse, NumberOfDependents
The app will automatically handle feature engineering, scaling, PCA, and prediction.

📌 Notes
The app automatically drops SeriousDlqin2yrs (if present)


📬 Contact
For questions or collaboration, feel free to reach out at:


GitHub: sanjay-0506

Email: m.sanjaykanth5@gmail.com


![Screenshot 2025-06-20 182512](https://github.com/user-attachments/assets/bc6a145c-90c8-4a22-9e45-9ceeff3bb50c)
![Screenshot 2025-06-20 182454](https://github.com/user-attachments/assets/9a4c1a43-4cad-4dfa-b3e8-ec9cb31c749a)

