# ML_challenge_proj

# Amazon ML Challenge 2026 – Business Entity Resolution

## Overview

This project was developed as part of the Amazon ML Challenge 2026. The objective was to solve a large-scale business entity resolution problem by identifying records that belong to the same real-world business across multiple datasets.

The task involved matching business records from two source datasets (S2 and S3) against a reference dataset (S1), despite differences in business names, addresses, contact details and formatting.

## Problem Statement

Business information is often stored across multiple databases, where the same entity may appear with slightly different names, addresses or contact information.

The goal was to identify matching business entities accurately while minimizing incorrect matches and handling records that did not have a corresponding entity in the reference dataset.

## Dataset

The challenge involved large-scale datasets:

- **S1:** Approximately 2.21 million reference entities.
- **S2 and S3:** More than 12 million records combined.
- Approximately 5.6% of records were unmatched.

The datasets contained business attributes such as names, addresses, phone numbers and email addresses.

## Approach and Methodology

Our solution followed a multi-stage entity matching pipeline.

### 1. Data Preprocessing

- Cleaned and normalized business names and addresses.
- Standardized phone numbers and email addresses.
- Handled inconsistent formatting and missing values.

### 2. Candidate Generation (Blocking)

Comparing every record against millions of reference entities would be computationally expensive. We used country-aware blocking to reduce the search space and generate relevant candidate matches.

Token-based and trigram-based techniques were also used to improve candidate identification.

### 3. Feature Engineering

Similarity features were generated between candidate business records, including:

- Business name similarity.
- Address similarity.
- Phone number similarity.
- Email similarity.
- Token and trigram-based similarity scores.

### 4. Machine Learning Models

We experimented with multiple approaches:

- Logistic Regression as a baseline model.
- XGBoost as the primary binary classification model.

The classifier estimated the probability that a candidate pair represented the same business entity.

### 5. Match Prediction

The predicted probabilities were compared against a selected threshold to determine whether records should be considered a match or remain unmatched.

The final predictions were exported in the required TSV format.

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Machine Learning
- Data Preprocessing and Feature Engineering
- Entity Resolution and Record Linkage

## Project Outputs

| File | Description |
|---|---|
| `matching_results.tsv` | Final predicted entity matches |
| `candidate_pairs.tsv` | Generated candidate pairs for matching and review |

**Note:** The complete output files are large, so they may be hosted externally rather than directly in this repository.

## Key Challenges

- Handling millions of business records efficiently.
- Reducing the computational cost of candidate matching.
- Dealing with inconsistent business information.
- Balancing false matches and missed matches.
- Managing unmatched records.

## What We Learned

This project provided practical experience with large-scale data processing, feature engineering, candidate generation and supervised machine learning. It also demonstrated the importance of efficient preprocessing and matching strategies when working with real-world datasets.

## Team Project

Developed collaboratively as part of the Amazon ML Challenge 2026.


