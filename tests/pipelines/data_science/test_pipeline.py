import logging
import pandas as pd
import pytest
from kedro.io import DataCatalog
from kedro.runner import SequentialRunner
from project.pipelines.data_science import create_pipeline as create_ds_pipeline
from project.pipelines.data_science.nodes import split_data


@pytest.fixture
def dummy_data():
    return pd.DataFrame(
        {
            "engines": [1, 2, 3],
            "crew": [4, 5, 6],
            "passenger_capacity": [5, 6, 7],
            "price": [120, 290, 30],
        }
    )


@pytest.fixture
def dummy_parameters():
    parameters = {
        "model_options": {
            "test_size": 0.2,
            "random_state": 3,
            "features": ["engines", "passenger_capacity", "crew"],
        }
    }
    return parameters


def test_split_data(dummy_data, dummy_parameters):
    X_train, X_test, y_train, y_test = split_data(
        dummy_data, dummy_parameters["model_options"]
    )
    assert len(X_train) == 2
    assert len(y_train) == 2
    assert len(X_test) == 1
    assert len(y_test) == 1


def test_split_data_missing_price(dummy_data, dummy_parameters):
    dummy_data_missing_price = dummy_data.drop(columns="price")
    with pytest.raises(KeyError) as e_info:
        X_train, X_test, y_train, y_test = split_data(
            dummy_data_missing_price, dummy_parameters["model_options"]
        )

    assert "price" in str(e_info.value)


def test_data_science_pipeline(caplog, dummy_data, dummy_parameters):
    pipeline = (
        create_ds_pipeline().from_nodes("split_data_node").to_nodes("train_model_node")
    )
    import yaml
    conf_catalog = """
'{catch_all}':
    "type": "pickle.PickleDataset"
    "filepath": "data/{catch_all}.pkl"
"""
    conf_catalog = yaml.safe_load(conf_catalog)
    # catalog = DataCatalog()
    catalog = DataCatalog.from_config(conf_catalog)
    catalog.add_feed_dict(
        {
            "model_input_table": dummy_data,
            "params:model_options": dummy_parameters["model_options"],
        }
    )

    runner = SequentialRunner(
        extra_dataset_patterns={
            "catch_all": {
                "type": "pickle.PickleDataset",
                "filepath": "data/{catch_all}.pkl",
            }
        }
    )
    # Now a,b are empty. So adding pattern in yaml or runner create different result.
    a = SequentialRunner().run(pipeline, catalog)
    b = SequentialRunner().run(pipeline, catalog)
    print(f"{a=}")
    print(f"{b=}")
    raise

