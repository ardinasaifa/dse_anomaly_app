import joblib

def load_model():

    iso = joblib.load(
        'models/isolation_forest.pkl'
    )

    scaler = joblib.load(
        'models/scaler.pkl'
    )

    features = joblib.load(
        'models/model_features.pkl'
    )

    return iso, scaler, features


def predict_anomaly(df):

    iso, scaler, features = load_model()

    X = df[features]

    X_scaled = scaler.transform(X)

    df['ANOMALY_SCORE_RAW'] = (
        iso.decision_function(X_scaled)
    )

    df['IS_ANOMALY'] = (
        iso.predict(X_scaled) == -1
    ).astype(int)

    return df


def generate_reason(row):

    reasons = []

    if row['EC_RATE'] < 0.3:
        reasons.append("EC rendah")

    if row['KPI_AVG_ACH'] < 50:
        reasons.append("KPI rendah")

    if row['BELOW_THRESHOLD_RATE'] > 0.7:
        reasons.append("Sering di bawah threshold")

    if row['SELLIN_PER_VISIT'] < 10000:
        reasons.append("Sellin per visit rendah")

    if row['EOM_CONCENTRATION'] > 0.4:
        reasons.append("Aktivitas menumpuk akhir bulan")

    return ", ".join(reasons)