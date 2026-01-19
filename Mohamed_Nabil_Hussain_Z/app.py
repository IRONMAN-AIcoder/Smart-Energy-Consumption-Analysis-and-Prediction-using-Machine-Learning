from flask import *
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from keras.layers import Dense
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


_original_from_config = Dense.from_config

@classmethod
def patched_from_config(cls, config):
    config.pop("quantization_config", None)
    return _original_from_config(config)

Dense.from_config = patched_from_config

app = Flask(__name__)

df_features = pd.read_csv(
    "features_final.csv",
    parse_dates=["time"]
).sort_values("time").reset_index(drop=True)

df_features["hour"] = df_features["time"].dt.hour
df_features["date"] = df_features["time"].dt.date
df_features["week"] = df_features["time"].dt.to_period("W").astype(str)
df_features["month"] = df_features["time"].dt.to_period("M").astype(str)




lstm_model = tf.keras.models.load_model(
    "lstm_energy_model_device_1.keras",
    compile=False
)

scaler = joblib.load("scaler_device.pkl")
device_encoder = joblib.load("device_label_encoder.pkl")

SEQ_LEN = 48

FEATURE_COLS = [
    "device",
    "total_device_power",
    "total_device_lag_1",
    "total_device_lag_24",
    "total_device_lag_48",
    "total_device_roll_24",
    "house_lag_1",
    "house_lag_24",
    "house_lag_48",
    "house_roll_24",
    "hour",
    "weekday",
    "is_weekend"
]

DEVICE_ID_TO_NAME = dict(enumerate(device_encoder.classes_))

def decode_device_name(device_id):
    name = DEVICE_ID_TO_NAME.get(int(device_id), "Unknown")
    return name.replace("[kW]", "").strip()   

def generate_charts():
    os.makedirs("static", exist_ok=True)

    hourly = df_features.groupby("hour")["House overall [kW]"].mean()

    plt.figure(figsize=(12, 5))
    plt.plot(hourly.index, hourly.values, marker="o")
    plt.xlabel("Hour of Day", fontsize=12)
    plt.ylabel("Average Power (kW)", fontsize=12)
    plt.title("Average Hourly House Usage", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("static/hourly_avg_features.png", dpi=120)
    plt.close()

    daily = df_features.groupby("date")["House overall [kW]"].sum()

    plt.figure(figsize=(16, 6))
    plt.plot(daily.index, daily.values)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Total Energy (kW)", fontsize=12)
    plt.title("Daily Energy Consumption", fontsize=14)
    plt.xticks(rotation=45, fontsize=9)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig("static/daily_total_features.png", dpi=120)
    plt.close()

    weekly = df_features.groupby("week")["House overall [kW]"].sum()

    plt.figure(figsize=(14, 6))
    plt.plot(weekly.index, weekly.values, marker="o")
    plt.xlabel("Week", fontsize=12)
    plt.ylabel("Total Energy (kW)", fontsize=12)
    plt.title("Weekly Energy Consumption", fontsize=14)
    plt.xticks(rotation=45, fontsize=9)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig("static/weekly_total_features.png", dpi=120)
    plt.close()

    monthly = df_features.groupby("month")["House overall [kW]"].sum()

    plt.figure(figsize=(12, 6))
    plt.plot(monthly.index, monthly.values, marker="o")
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Total Energy (kW)", fontsize=12)
    plt.title("Monthly Energy Consumption", fontsize=14)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig("static/monthly_total_features.png", dpi=120)
    plt.close()

    device_usage = df_features.groupby("device")["power_kW"].sum()

    device_usage.index = device_usage.index.map(decode_device_name)

    plt.figure(figsize=(8, 8))
    device_usage.plot(
        kind="pie",
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": 11}
    )
    plt.ylabel("")
    plt.title("Device-wise Energy Consumption", fontsize=14)
    plt.tight_layout()
    plt.savefig("static/device_pie.png", dpi=120)
    plt.close()



    plt.figure(figsize=(14, 6))
    plt.bar(device_usage.index, device_usage.values)
    plt.xlabel("Device", fontsize=12)
    plt.ylabel("Total Energy (kW)", fontsize=12)
    plt.title("Device-wise Energy Consumption", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig("static/device_bar.png", dpi=120)
    plt.close()


def clean_device_name(name: str) -> str:
    return name.replace("[kW]", "").strip()

def smart_suggestions(df):
    suggestions = []

    if np.issubdtype(df["device"].dtype, np.number):
        id_to_name = dict(enumerate(device_encoder.classes_))

        def decode_device(d):
            return clean_device_name(id_to_name.get(int(d), "Unknown device"))
    else:
        def decode_device(d):
            return clean_device_name(str(d))

    evening = df[df["hour"].between(18, 23)]
    if evening["House overall [kW]"].mean() > df["House overall [kW]"].mean() * 1.3:
        suggestions.append(
            "High evening energy spikes detected — consider using timers or smart plugs."
        )

    night = df[df["hour"].between(0, 5)]
    standby = night.groupby("device")["power_kW"].mean()

    for dev_id, val in standby.items():
        if val > 0.05:
            dev_name = decode_device(dev_id)
            suggestions.append(
                f"{dev_name} seems to consume power at night — check if it can be switched off."
            )

    device_totals = df.groupby("device")["power_kW"].sum()
    high_use = device_totals[device_totals > device_totals.mean() * 1.5]

    for dev_id in high_use.index:
        dev_name = decode_device(dev_id)
        suggestions.append(
            f"{dev_name} consumes significantly more power — consider energy-efficient alternatives."
        )

    suggestions.append(
        "Forecast: Expected rise in usage tomorrow — run appliances during off-peak hours."
    )

    return suggestions



def safe_iloc(df, idx, col):
    try:
        return float(df.iloc[idx][col])
    except:
        return float(df.iloc[-1][col])

def safe_mean(series, window):
    if len(series) < window:
        return float(series.mean())
    return float(series.tail(window).mean())

def encode_device(device):
    if device in device_encoder.classes_:
        return int(device_encoder.transform([device])[0])
    return int(device_encoder.transform([device_encoder.classes_[0]])[0])


if "device" in df_features.columns and df_features["device"].dtype == object:
    df_features["device"] = df_features["device"].apply(encode_device)


def get_model_input(df):
    X = df.tail(SEQ_LEN).reindex(columns=FEATURE_COLS)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X

def predict_next_house_power_from_features():
    X = get_model_input(df_features)
    X_scaled = scaler.transform(X)
    X_input = np.expand_dims(X_scaled, axis=0)

    delta = lstm_model.predict(X_input, verbose=0)[0][0]
    last_house = float(df_features["House overall [kW]"].iloc[-1])
    next_house = last_house + float(delta)

    if np.isnan(next_house) or np.isinf(next_house):
        next_house = last_house

    return float(next_house)

def append_user_reading(new_house_value, device=None):
    global df_features

    last_row = df_features.iloc[-1]
    new_time = last_row["time"] + pd.Timedelta(hours=1)

    device_encoded = encode_device(device) if device else int(last_row["device"])

    new_row = {
        "time": new_time,
        "device": device_encoded,
        "House overall [kW]": float(new_house_value),

        "total_device_power": float(last_row["total_device_power"]),
        "total_device_lag_1": float(last_row["total_device_power"]),
        "total_device_lag_24": safe_iloc(df_features, -24, "total_device_power"),
        "total_device_lag_48": safe_iloc(df_features, -48, "total_device_power"),
        "total_device_roll_24": safe_mean(df_features["total_device_power"], 24),

        "house_lag_1": float(last_row["House overall [kW]"]),
        "house_lag_24": safe_iloc(df_features, -24, "House overall [kW]"),
        "house_lag_48": safe_iloc(df_features, -48, "House overall [kW]"),
        "house_roll_24": safe_mean(df_features["House overall [kW]"], 24),

        "hour": new_time.hour,
        "weekday": new_time.weekday(),
        "is_weekend": int(new_time.weekday() >= 5)
    }

    df_features = pd.concat(
        [df_features, pd.DataFrame([new_row])],
        ignore_index=True
    )
    df_features.to_csv('features_final.csv',index=False)

def predict_future(hours_ahead, device=None):
    temp_df = df_features.copy()
    predictions = []

    device_encoded = encode_device(device) if device else int(temp_df.iloc[-1]["device"])

    for _ in range(hours_ahead):
        X = get_model_input(temp_df)
        X_scaled = scaler.transform(X)
        X_input = np.expand_dims(X_scaled, axis=0)

        delta = lstm_model.predict(X_input, verbose=0)[0][0]
        last_house = float(temp_df.iloc[-1]["House overall [kW]"])
        next_house = last_house + float(delta)

        if np.isnan(next_house) or np.isinf(next_house):
            next_house = last_house

        predictions.append(float(next_house))

        new_time = temp_df.iloc[-1]["time"] + pd.Timedelta(hours=1)

        new_row = {
            "time": new_time,
            "device": device_encoded,
            "House overall [kW]": next_house,

            "total_device_power": float(temp_df.iloc[-1]["total_device_power"]),
            "total_device_lag_1": float(temp_df.iloc[-1]["total_device_power"]),
            "total_device_lag_24": safe_iloc(temp_df, -24, "total_device_power"),
            "total_device_lag_48": safe_iloc(temp_df, -48, "total_device_power"),
            "total_device_roll_24": safe_mean(temp_df["total_device_power"], 24),

            "house_lag_1": last_house,
            "house_lag_24": safe_iloc(temp_df, -24, "House overall [kW]"),
            "house_lag_48": safe_iloc(temp_df, -48, "House overall [kW]"),
            "house_roll_24": safe_mean(temp_df["House overall [kW]"], 24),

            "hour": new_time.hour,
            "weekday": new_time.weekday(),
            "is_weekend": int(new_time.weekday() >= 5)
        }

        temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)
        temp_df.to_csv('features_final.csv',index=False)
    return predictions

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/devices")
def get_devices():
    encoded_ids = sorted(df_features["device"].unique())
    device_names = device_encoder.inverse_transform(encoded_ids)

    clean_names = [
        name.replace(" [kW]", "").strip()
        for name in device_names
    ]

    return jsonify(clean_names)


@app.route("/dashboard")
def dashboard():
    generate_charts()
    tips = smart_suggestions(df_features)

    return render_template(
        "dashboard.html",
        suggestions=tips,
        hourly_avg_features="hourly_avg_features.png",
        daily_total_features ="daily_total_features.png ",
        weekly_total_features="weekly_total_features.png",
        monthly_total_features="monthly_total_features.png",
        device_pie="device_pie.png",
        device_bar="device_bar.png"
    )
@app.route("/forecast")
def forecast_page():
    return render_template("prediction.html")

@app.route("/predict_with_input", methods=["POST"])
def predict_with_input():
    data = request.get_json(silent=True) or {}

    if "house_power" not in data:
        return jsonify({"error": "Missing house_power"}), 400

    device = data.get("device")
    device= device.lower() if device is not None else 'total_house'
    device_dict={
        'barn':'Barn [kW]',
    'dishwasher':'Dishwasher [kW]',
    'fridge':'Fridge [kW]',
    'furnace 1':'Furnace 1 [kW]',
    'furnace 2':'Furnace 2 [kW]',
    'garage door':'Garage door [kW]',
    'home office':'Home office [kW]',
    'kitchen 12':'Kitchen 12 [kW]',
    'kitchen 14':'Kitchen 14 [kW]',
    'kitchen 38':'Kitchen 38 [kW]',
    'living room':'Living room [kW]',
    'microwave':'Microwave [kW]',
    'well':'Well [kW]',
    'wine cellar':'Wine cellar [kW]',
    'total_house':None
 }
    dev=device_dict[device]
    new_value = float(data["house_power"])

    append_user_reading(new_value, device=dev)
    prediction = predict_next_house_power_from_features()

    return jsonify({
        "entered_value": new_value,
        "next_power_kW": prediction
    })
@app.route("/predict_day", methods=["POST"])
def predict_day():
    data = request.get_json(silent=True) or {}
    device = data.get("device")
    device= device.lower() if device is not None else 'total_house'
    device_dict={
        'barn':'Barn [kW]',
    'dishwasher':'Dishwasher [kW]',
    'fridge':'Fridge [kW]',
    'furnace 1':'Furnace 1 [kW]',
    'furnace 2':'Furnace 2 [kW]',
    'garage door':'Garage door [kW]',
    'home office':'Home office [kW]',
    'kitchen 12':'Kitchen 12 [kW]',
    'kitchen 14':'Kitchen 14 [kW]',
    'kitchen 38':'Kitchen 38 [kW]',
    'living room':'Living room [kW]',
    'microwave':'Microwave [kW]',
    'well':'Well [kW]',
    'wine cellar':'Wine cellar [kW]',
    'total_house':None
 }
    dev=device_dict[device]
    preds = predict_future(24, device=dev)

    return jsonify({
        "device": device if device else "total_house",
        "avg_kW": sum(preds) / len(preds)
    })

@app.route("/predict_week", methods=["POST"])
def predict_week():
    data = request.get_json(silent=True) or {}
    device = data.get("device")
    device= device.lower() if device is not None else 'total_house'
    device_dict={
        'barn':'Barn [kW]',
    'dishwasher':'Dishwasher [kW]',
    'fridge':'Fridge [kW]',
    'furnace 1':'Furnace 1 [kW]',
    'furnace 2':'Furnace 2 [kW]',
    'garage door':'Garage door [kW]',
    'home office':'Home office [kW]',
    'kitchen 12':'Kitchen 12 [kW]',
    'kitchen 14':'Kitchen 14 [kW]',
    'kitchen 38':'Kitchen 38 [kW]',
    'living room':'Living room [kW]',
    'microwave':'Microwave [kW]',
    'well':'Well [kW]',
    'wine cellar':'Wine cellar [kW]',
    'total_house':None
 }
    dev=device_dict[device]
    preds = predict_future(168, device=dev)

    return jsonify({
        "device": device if device else "total_house",
        "avg_kW": sum(preds) / len(preds)
    })

@app.route("/predict_month", methods=["POST"])
def predict_month():
    data = request.get_json(silent=True) or {}
    device = data.get("device")
    device= device.lower() if device is not None else 'total_house'
    device_dict={
        'barn':'Barn [kW]',
    'dishwasher':'Dishwasher [kW]',
    'fridge':'Fridge [kW]',
    'furnace 1':'Furnace 1 [kW]',
    'furnace 2':'Furnace 2 [kW]',
    'garage door':'Garage door [kW]',
    'home office':'Home office [kW]',
    'kitchen 12':'Kitchen 12 [kW]',
    'kitchen 14':'Kitchen 14 [kW]',
    'kitchen 38':'Kitchen 38 [kW]',
    'living room':'Living room [kW]',
    'microwave':'Microwave [kW]',
    'well':'Well [kW]',
    'wine cellar':'Wine cellar [kW]',
    'total_house':None
 }
    dev=device_dict[device]
    preds = predict_future(720, device=dev)

    return jsonify({
        "device": device if device else "total_house",
        "avg_kW": sum(preds) / len(preds)
    })

app.run(debug=True, use_reloader=False)
