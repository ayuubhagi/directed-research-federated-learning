# Traffic Flow Prediction using Federated Learning
# Dataset: mugithi/down_sampled_file_traffic_dataset (Hugging Face)
# The goal is to predict how many vehicles pass a junction (vehicleCount)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# this class handles loading and cleaning the dataset
class DataLoader:

    def __init__(self):
        # dataset address on hugging face
        self.dataset_name = "mugithi/down_sampled_file_traffic_dataset"

    def load_data(self):
        print("loading dataset from hugging face...")
        data = load_dataset(self.dataset_name)

        # convert to pandas table
        df = data["train"].to_pandas()

    
        # take a random sample of 100k rows
        df = df.sample(100000, random_state=42)

        print(f"loaded {len(df)} rows")
        print("columns:", list(df.columns))
        return df

    def clean_data(self, df):
        # remove exact duplicate rows
        df = df.drop_duplicates()

        # drop any rows with missing values
        df = df.dropna()

        # the Day column has text like "Monday"
        # the model cant understand text so we convert to numbers
        # Monday = 0, Tuesday = 1 ... Sunday = 6
        day_order = ["Monday", "Tuesday", "Wednesday",
                     "Thursday", "Friday", "Saturday", "Sunday"]
        df["Day"] = df["Day"].map(
            {day: i for i, day in enumerate(day_order)}
        )

        return df


# this class does exploratory data analysis
class EDA:

    def __init__(self, df):
        self.df = df

    def run(self):
        # print basic stats like mean, min, max for each column
        print("\nbasic stats:")
        print(self.df.describe())

        # check if any columns have missing values
        print("\nmissing values:")
        print(self.df.isnull().sum())

        # plot how vehicle counts are distributed
        # most readings should be low (0-5 cars)
        plt.hist(self.df["vehicleCount"], bins=40, color="steelblue")
        plt.title("Vehicle Count Distribution")
        plt.xlabel("vehicle count")
        plt.ylabel("frequency")
        plt.savefig("vehicle_distribution.png")
        plt.close()
        print("graph saved as vehicle_distribution.png")


# this class is our machine learning model
# we use linear regression to predict vehicle count
class Model:

    def __init__(self):
        self.model = LinearRegression()

    def train(self, X, y):
        # fit the model on features X and target y
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_weights(self):
        # these are what the model learned
        # we share these with the server instead of raw data
        return self.model.coef_, self.model.intercept_

    def set_weights(self, coef, intercept):
        # server sends back averaged weights
        # we update our local model with them
        self.model.coef_ = coef
        self.model.intercept_ = intercept


# main - runs everything
if __name__ == "__main__":

    # step 1 - load and clean the data
    loader = DataLoader()
    df = loader.load_data()
    df = loader.clean_data(df)

    # step 2 - explore the data
    eda = EDA(df)
    eda.run()

    # step 3 - pick our features and target
    # we use speed, hour an day to predict vehicle count
    X = df[["avgSpeed", "Hour", "Day"]]
    y = df["vehicleCount"]

    # step 4 - federated learning
    # in real life each client would be a different city or road zone
    # here we simulate this by splitting the data into 3 parts
    # each part = one client that trains on their own local data
    print("\ntraining clients...")

    split1 = df[:33333]   # client 1 - first third of data
    split2 = df[33333:66666]  # client 2 - middle third
    split3 = df[66666:]   # client 3 - last third
    splits = [split1, split2, split3]

    coefs = []
    intercepts = []

    for i, part in enumerate(splits):
        print(f"client {i+1} training...")

        # each client only sees their own data - not the others
        X_part = part[["avgSpeed", "Hour", "Day"]]
        y_part = part["vehicleCount"]

        # train local model
        model = Model()
        model.train(X_part, y_part)

        # collect the weights this client learned
        coef, intercept = model.get_weights()
        coefs.append(coef)
        intercepts.append(intercept)

    # step 5 - federated averaging (FedAvg)
    # the server takes all client weights and averages them
    # this is the core idea of federated learning
    # no raw data is shared - only what each model learned
    avg_coef = np.mean(coefs, axis=0)
    avg_intercept = np.mean(intercepts)

    # step 6 - build the final global model with averaged weights
    final_model = Model()
    final_model.train(X, y)  # need to initialize first
    final_model.set_weights(avg_coef, avg_intercept)

    # step 7 - evaluate how good our global model is
    preds = final_model.predict(X)

    print("\nfinal results:")
    print("MAE:", round(mean_absolute_error(y, preds), 4))
    print("R2: ", round(r2_score(y, preds), 4))
    print("\nMAE = average error in vehicle count prediction")
    print("R2  = how well the model fits the data (1.0 = perfect)")