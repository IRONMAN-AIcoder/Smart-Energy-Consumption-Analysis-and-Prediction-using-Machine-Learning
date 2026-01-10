from flask import *
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense
import joblib

_original_from_config = Dense.from_config

@classmethod
def patched_from_config(cls, config):
    config.pop("quantization_config", None)
    return _original_from_config(config)

Dense.from_config = patched_from_config


app = Flask(__name__)


df_long = pd.read_csv(
    "./organized_energy_data_1.csv",
    usecols=["time", "House overall [kW]", "device", "power_kW"],
    parse_dates=["time"]
)

df_long["hour"] = df_long["time"].dt.hour
df_long["date"] = df_long["time"].dt.date


df_features = pd.read_csv(
    "features_final.csv", 
    parse_dates=["time"]
).sort_values("time")


lstm_model = tf.keras.models.load_model(
    "lstm_energy_model_1.keras",
    compile=False
)

scaler = StandardScaler()
scaler.mean_ = np.load("lstm_scaler_mean_2.npy")
scaler.scale_ = np.load("lstm_scaler_scale_2.npy")

SEQ_LEN = 48

FEATURE_COLS = [
    'total_device_power',
    'total_device_lag_1',
    'total_device_lag_24',
    'total_device_lag_48',
    'total_device_roll_24',
    'house_lag_1',
    'house_lag_24',
    'house_lag_48',
    'house_roll_24',
    'hour',
    'weekday',
    'is_weekend'
]


def generate_charts():
    if not os.path.exists("static"):
        os.makedirs("static")

    hourly = df_long.groupby("hour")["House overall [kW]"].mean()
    daily = df_long.groupby("date")["House overall [kW]"].sum()
    device_usage = df_long.groupby("device", observed=False)["power_kW"].sum()

    plt.figure(figsize=(10, 4))
    plt.plot(hourly.index, hourly.values)
    plt.savefig("static/hourly.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(daily.index, daily.values)
    plt.xticks(rotation=45)
    plt.savefig("static/daily.png")
    plt.close()

    plt.figure(figsize=(7, 7))
    device_usage.plot(kind="pie", autopct="%1.1f%%")
    plt.savefig("static/device_pie.png")
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.bar(device_usage.index, device_usage.values)
    plt.xticks(rotation=45)
    plt.savefig("static/device_bar.png")
    plt.close()


def smart_suggestions(df):
    suggestions = []

    # 1. High evening consumption
    evening = df[df['hour'].between(18, 23)]
    if evening['House overall [kW]'].mean() > df['House overall [kW]'].mean() * 1.3:
        suggestions.append("High evening energy spikes detected — consider using timers or smart plugs.")

    # 2. Standby devices running at night
    night = df[df['hour'].between(0, 5)]
    standby = night.groupby("device")["power_kW"].mean()
    for dev, val in standby.items():
        if val > 0.05:
            suggestions.append(f"{dev} seems to consume power at night — check if it can be switched off.")

    # 3. Device using too much
    device_totals = df.groupby("device")["power_kW"].sum()
    high_use = device_totals[device_totals > device_totals.mean() * 1.5]
    for dev in high_use.index:
        suggestions.append(f"{dev} consumes significantly more power — consider energy-efficient alternatives.")

    # 4. Forecast hook
    suggestions.append("Forecast: Expected rise in usage tomorrow — run appliances during off-peak hours.")

    return suggestions


@app.route('/')
def home():
    return render_template("home.html")
@app.route("/dashboard")
def dashboard():
    # generate_charts()
    tips = smart_suggestions(df_long)

    return render_template(
        "dashboard.html",
        suggestions=tips,
        hourly_img="hourly.png",
        daily_img="daily.png",
        device_pie="device_pie.png",
        device_bar="device_bar.png"
    )

@app.route("/forecast")
def forecast_page():
    return render_template("prediction.html")

@app.route("/predict_auto")
def predict_auto():
    prediction = predict_next_house_power_from_features()
    return jsonify({"next_power_kW": prediction})

def safe_iloc(df, idx, col):
    try:
        return df.iloc[idx][col]
    except:
        return df.iloc[-1][col]
def append_user_reading(new_house_value):
    global df_features

    last_row = df_features.iloc[-1].copy()
    new_time = last_row["time"] + pd.Timedelta(hours=1)

    new_row = {
        "time": new_time,

        "power_kW":last_row["power_kW"],
        "House overall [kW]": new_house_value,

        "total_device_power": last_row["total_device_power"],
        "total_device_lag_1": last_row["total_device_power"],
        "total_device_lag_24": safe_iloc(df_features, -24, "total_device_power"),
        "total_device_lag_48": safe_iloc(df_features, -48, "total_device_power"),
        "total_device_roll_24": df_features["total_device_power"].tail(24).mean(),

        "house_lag_1": last_row["House overall [kW]"],
        "house_lag_24": safe_iloc(df_features, -24, "House overall [kW]"),
        "house_lag_48": safe_iloc(df_features, -48, "House overall [kW]"),
        "house_roll_24": df_features["House overall [kW]"].tail(24).mean(),

        "hour": new_time.hour,
        "weekday": new_time.weekday(),
        "is_weekend": int(new_time.weekday() >= 5)
    }

    df_features = pd.concat(
        [df_features, pd.DataFrame([new_row])],
        ignore_index=True
    )

    df_features.to_csv("features_final.csv", index=False)

def predict_next_house_power_from_features():
    X = df_features.tail(SEQ_LEN)[FEATURE_COLS]

    X_scaled = scaler.transform(X)
    X_input = np.expand_dims(X_scaled, axis=0)

    delta = lstm_model.predict(X_input, verbose=0)[0][0]

    last_house = df_features["House overall [kW]"].iloc[-1]
    next_house = last_house + delta

    return float(next_house)


@app.route("/predict_with_input", methods=["POST"])
def predict_with_input():
    data = request.get_json()

    if "house_power" not in data:
        return jsonify({"error": "Missing house_power"}), 400

    new_value = float(data["house_power"])
    append_user_reading(new_value)

    prediction = predict_next_house_power_from_features()

    return jsonify({
        "entered_value": new_value,
        "next_power_kW": prediction
    })
def predict_future(hours_ahead):
    global df_features

    temp_df = df_features.copy()
    predictions = []

    for _ in range(hours_ahead):
        X = temp_df.tail(SEQ_LEN)[FEATURE_COLS]

        X_scaled = scaler.transform(X)
        X_input = np.expand_dims(X_scaled, axis=0)

        delta = lstm_model.predict(X_input, verbose=0)[0][0]

        last_house = temp_df["House overall [kW]"].iloc[-1]
        next_house = last_house + delta
        predictions.append(float(next_house))

        # Build next row (same logic you already use)
        last_row = temp_df.iloc[-1]
        new_time = last_row["time"] + pd.Timedelta(hours=1)

        new_row = {
            "time": new_time,
            "House overall [kW]": next_house,

            "total_device_power": last_row["total_device_power"],
            "total_device_lag_1": last_row["total_device_power"],
            "total_device_lag_24": safe_iloc(temp_df, -24, "total_device_power"),
            "total_device_lag_48": safe_iloc(temp_df, -48, "total_device_power"),
            "total_device_roll_24": temp_df["total_device_power"].tail(24).mean(),

            "house_lag_1": last_house,
            "house_lag_24": safe_iloc(temp_df, -24, "House overall [kW]"),
            "house_lag_48": safe_iloc(temp_df, -48, "House overall [kW]"),
            "house_roll_24": temp_df["House overall [kW]"].tail(24).mean(),

            "hour": new_time.hour,
            "weekday": new_time.weekday(),
            "is_weekend": int(new_time.weekday() >= 5)
        }

        temp_df = pd.concat(
            [temp_df, pd.DataFrame([new_row])],
            ignore_index=True
        )

    return predictions
@app.route("/predict_week")
def predict_week():
    preds = predict_future(168)
    return jsonify({
        "hours": 168,
        "avg_kW": sum(preds) / len(preds),
        "daily_avg": [sum(preds[i:i+24]) / 24 for i in range(0, 168, 24)]
    })
@app.route("/predict_month")
def predict_month():
    preds = predict_future(720)
    return jsonify({
        "hours": 720,
        "avg_kW": sum(preds) / len(preds),
        "weekly_avg": [sum(preds[i:i+168]) / 168 for i in range(0, 720, 168)]
    })


app.run(debug=True, use_reloader=False)
