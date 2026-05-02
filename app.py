from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and features
model    = joblib.load('models/lgbm_model.pkl')
FEATURES = joblib.load('models/features.pkl')
  
# Store stats for encoding — load from clean file with only needed columns
import pandas as pd
train = pd.read_csv('data/train_clean.csv',
                    usecols=['Store', 'Sales'],
                    low_memory=False)
store_stats = train.groupby('Store')['Sales'].agg(
    StoreMeanSales='mean',
    StoreMedianSales='median',
    StoreStdSales='std'
).reset_index()
del train  # free memory immediately 

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Rossmann Sales Predictor</title>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; }
        body { font-family: 'Segoe UI', sans-serif; background:#0f172a; color:#e2e8f0; min-height:100vh; display:flex; align-items:center; justify-content:center; }
        .card { background:#1e293b; border-radius:16px; padding:40px; width:100%; max-width:520px; box-shadow:0 25px 50px rgba(0,0,0,0.4); }
        h1 { font-size:24px; font-weight:600; margin-bottom:6px; color:#f8fafc; }
        .sub { font-size:13px; color:#94a3b8; margin-bottom:32px; }
        .grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
        .field { display:flex; flex-direction:column; gap:6px; }
        .field.full { grid-column:1/-1; }
        label { font-size:12px; font-weight:500; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; }
        input, select { background:#0f172a; border:1px solid #334155; border-radius:8px; padding:10px 14px; color:#f8fafc; font-size:14px; outline:none; }
        input:focus, select:focus { border-color:#6366f1; }
        select option { background:#1e293b; }
        button { width:100%; margin-top:24px; padding:14px; background:#6366f1; color:#fff; border:none; border-radius:8px; font-size:15px; font-weight:600; cursor:pointer; }
        button:hover { background:#4f46e5; }
        .result { margin-top:24px; background:#0f172a; border-radius:10px; padding:20px; text-align:center; display:none; }
        .result-label { font-size:12px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; }
        .result-value { font-size:36px; font-weight:700; color:#34d399; margin-top:6px; }
        .result-sub { font-size:12px; color:#64748b; margin-top:4px; }
        .error { color:#f87171; font-size:13px; margin-top:12px; text-align:center; display:none; }
    </style>
</head>
<body>
<div class="card">
    <h1>Rossmann Sales Predictor</h1>
    <p class="sub">Forecast daily sales for any store — 6 weeks ahead</p>
    <div class="grid">
        <div class="field">
            <label>Store ID (1–1115)</label>
            <input type="number" id="store" min="1" max="1115" value="1" />
        </div>
        <div class="field">
            <label>Date</label>
            <input type="date" id="date" value="2015-09-01" />
        </div>
        <div class="field">
            <label>Promo active?</label>
            <select id="promo">
                <option value="1">Yes</option>
                <option value="0">No</option>
            </select>
        </div>
        <div class="field">
            <label>School Holiday?</label>
            <select id="school">
                <option value="0">No</option>
                <option value="1">Yes</option>
            </select>
        </div>
        <div class="field full">
            <label>State Holiday</label>
            <select id="holiday">
                <option value="0">None</option>
                <option value="1">Public Holiday</option>
                <option value="2">Easter</option>
                <option value="3">Christmas</option>
            </select>
        </div>
    </div>
    <button onclick="predict()">Predict Sales</button>
    <div class="result" id="result">
        <p class="result-label">Predicted Daily Sales</p>
        <p class="result-value" id="pred-value">€0</p>
        <p class="result-sub" id="pred-sub"></p>
    </div>
    <p class="error" id="error"></p>
</div>
<script>
async function predict() {
    const store = document.getElementById('store').value;
    const date  = document.getElementById('date').value;
    const promo = document.getElementById('promo').value;
    const school = document.getElementById('school').value;
    const holiday = document.getElementById('holiday').value;

    document.getElementById('result').style.display = 'none';
    document.getElementById('error').style.display  = 'none';

    const res = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({store, date, promo, school, holiday})
    });
    const data = await res.json();
    if (data.error) {
        document.getElementById('error').innerText = data.error;
        document.getElementById('error').style.display = 'block';
    } else {
        document.getElementById('pred-value').innerText = '€' + data.prediction.toLocaleString();
        document.getElementById('pred-sub').innerText   = 'Store #' + store + ' on ' + date;
        document.getElementById('result').style.display = 'block';
    }
}
</script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': 'LightGBM', 'version': '1.0'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data     = request.get_json()
        store_id = int(data['store'])
        date     = pd.to_datetime(data['date'])
        promo    = int(data['promo'])
        school   = int(data['school'])
        holiday  = int(data['holiday'])

        # Get store stats
        stats = store_stats[store_stats['Store'] == store_id]
        if len(stats) == 0:
            return jsonify({'error': 'Store ID not found'})

        store_mean   = stats['StoreMeanSales'].values[0]
        store_median = stats['StoreMedianSales'].values[0]
        store_std    = stats['StoreStdSales'].values[0]

        # Build feature row
        row = {
            'Store'                    : store_id,
            'DayOfWeek'                : date.dayofweek + 1,
            'Promo'                    : promo,
            'StateHoliday'             : holiday,
            'SchoolHoliday'            : school,
            'StoreType'                : 0,
            'Assortment'               : 0,
            'CompetitionDistance'      : 1000,
            'CompetitionOpenSinceMonth': 1,
            'CompetitionOpenSinceYear' : 2010,
            'Promo2'                   : 0,
            'Promo2SinceWeek'          : 0,
            'Promo2SinceYear'          : 0,
            'IsPromo2Active'           : 0,
            'CompetitionOpen'          : 60,
            'Year'                     : date.year,
            'Month'                    : date.month,
            'Day'                      : date.day,
            'WeekOfYear'               : date.isocalendar()[1],
            'Quarter'                  : date.quarter,
            'IsWeekend'                : int(date.dayofweek >= 5),
            'IsMonthStart'             : int(date.is_month_start),
            'IsMonthEnd'               : int(date.is_month_end),
            'StoreMeanSales'           : store_mean,
            'StoreMedianSales'         : store_median,
            'StoreStdSales'            : store_std
        }

        df   = pd.DataFrame([row])[FEATURES]
        pred = model.predict(df)[0]
        pred = float(np.expm1(pred))

        return jsonify({'prediction': round(pred, 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)