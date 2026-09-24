import numpy as np


def test_shape_and_seed_reproducibility():
    from generate_synthetic_data import generate

    df1 = generate(n_rows=500, seed=1)
    df2 = generate(n_rows=500, seed=1)
    assert df1.shape == (500, 29)
    assert df1.equals(df2)


def test_no_infinite_or_null_values(raw_df):
    numeric = raw_df.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.to_numpy()).any()
    assert not raw_df.isna().any().any()


def test_target_columns_within_expected_bounds(raw_df):
    for col in ["Spoilage_Risk", "Efficiency_Ratio", "Quality_Maintenance_Ratio"]:
        assert raw_df[col].between(0, 100).all()


def test_vehicle_type_categories(raw_df):
    assert set(raw_df["Vehicle_Type"].unique()) <= {"Truck", "Van", "Motorbike"}
