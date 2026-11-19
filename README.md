# Federated Learning for Traffic Flow Prediction

Starting point for my directed research at SPIN Lab: a federated learning baseline that predicts how many vehicles pass a junction, without any client sharing its raw traffic data.

## How it works

1. Loads a 100k-row sample of the [`mugithi/down_sampled_file_traffic_dataset`](https://huggingface.co/datasets/mugithi/down_sampled_file_traffic_dataset) dataset from Hugging Face, drops duplicates and missing rows, and encodes the day of week as a number.
2. Splits the data into 3 clients. In a real deployment each one would be a different city or road zone.
3. Each client trains its own linear regression on `avgSpeed`, `Hour` and `Day` to predict `vehicleCount`, using only its own data.
4. The server averages the clients' weights (FedAvg) into one global model. Only model weights leave a client.
5. Reports MAE and R² for the global model and saves a vehicle-count histogram.

## Running it

```bash
pip install pandas numpy matplotlib datasets scikit-learn
python traffic.py
```

`Machine Learning with Applications ppt.pptx` is the presentation for this stage of the project.

## Where it's going

This repo is the simple baseline. The ongoing work moves to sequence models on larger traffic datasets and looks at cutting the number of parameters clients send each round without losing accuracy.
