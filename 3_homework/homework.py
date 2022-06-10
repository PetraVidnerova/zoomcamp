from datetime import timedelta
from prefect.flow_runners import SubprocessFlowRunner
from prefect.orion.schemas.schedules import CronSchedule
from prefect.deployments import DeploymentSpec
import pickle
#from prefect.deployments import DeploymentSpec
from datetime import timedelta, datetime
# from prefect.flow_runners import SubprocessFlowRunner
# from prefect.orion.schemas.schedules import IntervalSchedule
from prefect import flow, task, get_run_logger

import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from prefect.task_runners import SequentialTaskRunner

from dateutil.relativedelta import relativedelta


@task
def get_paths(date=None):
    if date is None:
        date = datetime.today()
    else:
        date = datetime.strptime(date, "%Y-%m-%d")

    date1 = date + relativedelta(months=-1)
    month1 = date1.month
    year1 = date1.year

    date2 = date + relativedelta(months=-2)
    month2 = date2.month
    year2 = date2.year

    return (
        f"https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year1}-{month1:02d}.parquet",
        f"https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year2}-{month2:02d}.parquet"
    )


@task
def read_data(path):
    logger = get_run_logger()
    logger.info(f"READING  {path}")
    df = pd.read_parquet(path)
    return df


@task
def prepare_features(df, categorical, train=True):

    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


@task
def train_model(df, categorical):
    logger = get_run_logger()

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv


@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()

    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return


@flow(task_runner=SequentialTaskRunner())
def main(date=None):

    train_path, val_path = get_paths(date).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)

    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    with open(f"model-{date}.bin", "wb") as f:
        pickle.dump(lr, f)

    with open(f"dv-{date}.bin", "wb") as f:
        pickle.dump(dv, f)


# main(date="2021-03-15")
# main(date="2021-08-15")


DeploymentSpec(
    flow=main,
    name="model_training",
    schedule=CronSchedule(
        cron="0 9 15 * *"
    ),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"]
)
