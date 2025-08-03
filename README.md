# NSAP Scheme Classification

This project aims to automate the classification of applicants into appropriate schemes under the **National Social Assistance Program (NSAP)** using machine learning. The NSAP is a flagship welfare program by the Government of India providing financial assistance to the elderly, widows, and persons with disabilities from below-poverty-line (BPL) households.

## Problem Statement

Manually verifying applications and assigning the correct scheme is time-consuming and error-prone. Incorrect allocation can delay or deny financial aid to deserving individuals.

### Objective

To design, build, and evaluate a **multi-class classification model** that predicts the most suitable NSAP scheme for an applicant based on their demographic and socio-economic data.

## Project Structure

- `NSAP.ipynb` – Jupyter Notebook with data exploration, preprocessing, model building, and evaluation.
- `Social Welfare Schemes.csv` – Dataset containing applicant details and scheme information.
- `Social_Welfare_Schemes.pkl` – Pickled version of the dataset for faster loading.
- `NSAP_Project_Description.pdf` – Project description in PDF format.
- `README.md` – You are here.

## Dataset

The dataset includes features such as:
- Age
- Gender
- Marital status
- Disability status
- Income level
- Region (urban/rural)
- Existing support

These features are used to classify applicants into various NSAP schemes such as:
- Indira Gandhi National Old Age Pension Scheme (IGNOAPS)
- Indira Gandhi National Widow Pension Scheme (IGNWPS)
- Indira Gandhi National Disability Pension Scheme (IGNDPS)

## Model

A multi-class classification model (e.g., Logistic Regression, Random Forest, or XGBoost) is trained to predict the most appropriate scheme.

### Key steps:
- Data Cleaning & Preprocessing
- Feature Encoding and Normalization
- Model Training & Cross-Validation
- Accuracy, Precision, Recall Evaluation

##  Usage

```bash
# Load the pickled dataset
import pickle

with open("Social_Welfare_Schemes.pkl", "rb") as file:
    df = pickle.load(file)

# Proceed with training or inference using the DataFrame
