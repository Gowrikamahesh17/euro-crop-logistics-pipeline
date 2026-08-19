"""
Generates a synthetic replacement for the EuroCrop dataset.

The original Kaggle dataset (Step 1 audit, notebooks/01_eda_inspection.ipynb)
was found to be structurally unusable: 18 of 23 numeric columns were 47-100%
corrupted by float overflow (inf), with 0 fully-clean rows and no way to
recover the original signal. This script generates a same-schema replacement
with physically plausible feature distributions and targets computed as
formulas over those features plus noise, so the 4-target modeling system has
real, learnable relationships to find.

Run directly to (re)write data/raw/EuroCrop_agricultural_logistics_dataset.csv:
    python src/generate_synthetic_data.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

N_ROWS = 53_305
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))


def _clip(series: pd.Series, lo: float, hi: float) -> pd.Series:
    return series.clip(lower=lo, upper=hi)


def generate(n_rows: int = N_ROWS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Categorical features -------------------------------------------------
    crop_type = rng.choice(
        ["Wheat", "Corn", "Rice"], size=n_rows, p=[0.40, 0.40, 0.20]
    )

    # --- Timestamps -------------------------------------------------------------
    event_ts = pd.to_datetime("2018-06-01") + pd.to_timedelta(
        rng.integers(0, 365 * 6, size=n_rows), unit="D"
    ) + pd.to_timedelta(rng.integers(0, 24, size=n_rows), unit="h")

    harvest_date = pd.to_datetime("2017-01-01") + pd.to_timedelta(
        rng.integers(0, 365, size=n_rows), unit="D"
    )

    # --- Environmental sensors (correlated within each family) -----------------
    # Base "true" ambient temperature/humidity per row, sensors read it with noise.
    true_temp = rng.normal(loc=8, scale=10, size=n_rows)  # cold-chain-ish mean
    storage_temperature = _clip(pd.Series(true_temp + rng.normal(0, 1.5, n_rows)), -30, 50)
    temperature = _clip(pd.Series(true_temp + rng.normal(0, 1.5, n_rows)), -30, 50)
    iot_temperature = _clip(pd.Series(true_temp + rng.normal(0, 2.0, n_rows)), -30, 50)

    true_humidity = rng.normal(loc=55, scale=15, size=n_rows)
    storage_humidity = _clip(pd.Series(true_humidity + rng.normal(0, 4, n_rows)), 0, 100)
    humidity = _clip(pd.Series(true_humidity + rng.normal(0, 4, n_rows)), 0, 100)
    iot_humidity = _clip(pd.Series(true_humidity + rng.normal(0, 5, n_rows)), 0, 100)

    iot_light = _clip(pd.Series(rng.gamma(shape=2.0, scale=8000, size=n_rows)), 0, 120_000)

    vibration_level = _clip(pd.Series(rng.gamma(shape=2.0, scale=1.8, size=n_rows)), 0, 60)

    # --- Vehicle / logistics features ------------------------------------------
    route_distance = _clip(pd.Series(rng.gamma(shape=2.2, scale=180, size=n_rows)), 5, 4500)
    traffic_level = _clip(pd.Series(rng.normal(3, 1.2, n_rows)), 1, 5)
    weather_impact = _clip(pd.Series(rng.normal(2.5, 1.0, n_rows)), 1, 5)

    base_speed_kmh = 55 - traffic_level * 4 - weather_impact * 2
    base_speed_kmh = _clip(base_speed_kmh, 15, 60)
    delivery_time = _clip(route_distance / base_speed_kmh + rng.normal(0, 1.5, n_rows), 0.2, 200)

    queue_time = _clip(pd.Series(rng.gamma(shape=1.5, scale=1.2, size=n_rows)), 0, 40)

    fuel_consumption = _clip(
        pd.Series(route_distance * rng.normal(0.28, 0.05, n_rows) + rng.normal(0, 3, n_rows)),
        1, 1800,
    )
    fuel_costs = _clip(fuel_consumption * rng.normal(1.6, 0.15, n_rows), 1, 3000)

    vehicle_load_capacity = _clip(pd.Series(rng.normal(12_000, 5000, n_rows)), 300, 40_000)
    station_capacity = _clip(pd.Series(rng.normal(50_000, 15_000, n_rows)), 1_000, 150_000)
    inventory_levels = _clip(pd.Series(rng.normal(20_000, 8000, n_rows)), 0, 120_000)

    operational_cost = _clip(
        fuel_costs + route_distance * rng.normal(0.4, 0.1, n_rows) + rng.normal(0, 30, n_rows),
        5, 6000,
    )
    energy_consumption = _clip(
        pd.Series(fuel_consumption * rng.normal(3.2, 0.4, n_rows)), 5, 6000
    )

    warehouse_storage_time = _clip(pd.Series(rng.gamma(shape=2.0, scale=15, size=n_rows)), 0, 400)
    crop_yield = _clip(pd.Series(rng.normal(500, 180, n_rows)), 10, 2000)

    # --- Vehicle_Type: Target 4 (classification) --------------------------------
    # Correlated with load / distance / crop: bulk grain over long routes -> Truck,
    # smaller loads / shorter local routes -> Van or Motorbike.
    # Intercepts set the baseline class balance (~50/30/20 truck/van/moto);
    # feature terms are standardized (z-scored) before weighting so no single
    # feature's raw scale can silently dominate or erase the intercept.
    load_z = (vehicle_load_capacity - vehicle_load_capacity.mean()) / vehicle_load_capacity.std()
    dist_z = (route_distance - route_distance.mean()) / route_distance.std()

    van_score = 0.2 - 0.5 * load_z - 0.4 * dist_z + rng.normal(0, 1.0, n_rows)
    moto_score = -0.5 - 0.9 * load_z - 0.9 * dist_z + rng.normal(0, 1.0, n_rows)
    truck_score = np.zeros(n_rows)  # reference class

    scores = np.vstack([truck_score, van_score, moto_score]).T
    exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    vehicle_choices = np.array(["Truck", "Van", "Motorbike"])
    vehicle_type = np.array([
        rng.choice(vehicle_choices, p=probs[i]) for i in range(n_rows)
    ])

    # --- Targets 1-3: physically-motivated formulas + noise --------------------
    ideal_temp = 4.0  # deg C, typical cold-chain target for produce
    temp_deviation = (storage_temperature - ideal_temp).abs()
    humidity_deviation = (storage_humidity - 60).abs()

    spoilage_risk = (
        1.8 * temp_deviation
        + 0.35 * humidity_deviation
        + 0.9 * vibration_level
        + 0.05 * warehouse_storage_time
        + rng.normal(0, 5, n_rows)
    )
    spoilage_risk = _clip(spoilage_risk, 0, 100)

    efficiency_ratio = (
        100
        - 0.02 * fuel_consumption
        - 0.015 * operational_cost
        - 0.05 * delivery_time
        - 0.01 * route_distance / 10
        + rng.normal(0, 6, n_rows)
    )
    efficiency_ratio = _clip(efficiency_ratio, 0, 100)

    quality_maintenance_ratio = (
        100
        - 0.15 * warehouse_storage_time
        - 0.3 * temp_deviation
        - 0.2 * humidity_deviation
        + rng.normal(0, 5, n_rows)
    )
    quality_maintenance_ratio = _clip(quality_maintenance_ratio, 0, 100)

    df = pd.DataFrame({
        "": event_ts,
        "Vehicle_Type": vehicle_type,
        "Crop_Type": crop_type,
        "Harvest_Date": harvest_date,
        "Crop_Yield": crop_yield.round(2),
        "Storage_Temperature": storage_temperature.round(2),
        "Storage_Humidity": storage_humidity.round(2),
        "Fuel_Consumption": fuel_consumption.round(2),
        "Route_Distance": route_distance.round(2),
        "Delivery_Time": delivery_time.round(2),
        "Traffic_Level": traffic_level.round(2),
        "Temperature": temperature.round(2),
        "Humidity": humidity.round(2),
        "Vehicle_Load_Capacity": vehicle_load_capacity.round(2),
        "Vibration_Level": vibration_level.round(2),
        "Queue_Time": queue_time.round(2),
        "Weather_Impact": weather_impact.round(2),
        "Station_Capacity": station_capacity.round(2),
        "Operational_Cost": operational_cost.round(2),
        "Energy_Consumption": energy_consumption.round(2),
        "IoT_Sensor_Reading_Temperature": iot_temperature.round(2),
        "IoT_Sensor_Reading_Humidity": iot_humidity.round(2),
        "IoT_Sensor_Reading_Light": iot_light.round(2),
        "Warehouse_Storage_Time": warehouse_storage_time.round(2),
        "Inventory_Levels": inventory_levels.round(2),
        "Fuel_Costs": fuel_costs.round(2),
        "Spoilage_Risk": spoilage_risk.round(2),
        "Efficiency_Ratio": efficiency_ratio.round(2),
        "Quality_Maintenance_Ratio": quality_maintenance_ratio.round(2),
    })

    return df


def main():
    raw_path = os.getenv("DATA_RAW_PATH", "data/raw/EuroCrop_agricultural_logistics_dataset.csv")
    out_path = PROJECT_ROOT / raw_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate()
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
