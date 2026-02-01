# ML-2026-Linear-Regression-by-hand — Housing Price Prediction

This is **my first machine learning “by hand” project**. I built a simple linear regression model from scratch to learn the fundamentals: building a design matrix, fitting weights with matrix math, calculating error (cost), and making predictions.

## What it does
- Loads a housing dataset from a CSV file
- Lets you choose X feature columns and a Y target column
- Fits linear regression weights using the normal equation (pseudoinverse)
- Prints the cost value and learned weights
- Predicts a target value from user-entered feature values

## Files
- `LinReg.py` — Linear regression implementation (builds X/Y, fits weights, predicts, computes cost).
- `main.py` — Command-line runner that loads the CSV, trains the model, and runs a prediction from input.
- `MoreHouseData.csv` — Example housing dataset used for training/testing.

## How to run
```bash
pip install numpy pandas
python main.py
