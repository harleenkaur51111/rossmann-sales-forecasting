**# Rossmann Store Sales Forecasting**



**End-to-end machine learning project to forecast sales for Rossmann Pharmaceuticals across 1,115 stores, 6 weeks ahead of time.**



**## Results**

**- Model: LightGBM**

**- Validation RMSPE: 0.1383**

**- Training data: 1,017,209 records across 1,115 stores (2013-2015)**



**## Project Structure**

**- 01\_data\_loading\_cleaning.ipynb — Data loading, merging and cleaning**

**- 02\_feature\_engineering.ipynb — Feature creation**

**- 03\_modeling.ipynb — LightGBM model training**

**- 04\_mlops.ipynb — MLflow experiment tracking**

**- app.py — Flask web application for live predictions**



**## How to Run**

**1. Install dependencies: pip install -r requirements.txt**

**2. Run the Flask app: python app.py**

**3. Open browser at: http://127.0.0.1:5001**



**## Tech Stack**

**Python, Pandas, LightGBM, Flask, MLflow, Scikit-learn, Matplotlib, Seaborn**

