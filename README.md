# tslib - A Time Series Library
Time Series Forecasting and Imputation

Implementation based on: Model Agnostic Time Series Analysis via Matrix Estimation
(https://dl.acm.org/citation.cfm?id=3287319). 

This work has the following dependencies:

numpy
pandas
scipy
sklearn

Supported for Python 2.7 and 3+.

# Robust Synthetic Control

This library also has an implementation for RSC as detailed in http://www.jmlr.org/papers/volume19/17-777/17-777.pdf

# Multi-Dimensional Robust Synthetic Control

This library also has an implementation for mRSC as detailed in http://dna-pubs.cs.columbia.edu/citation/paperfile/230/mRSC.pdf


# Documentation:
Please see the test scripts under the tests/ folder which provide sample usage, often with real datasets.

1. testScriptSingleTimeseries.py: imputing and foreasting a single time series using synthetically generated data. The script uses both the SVD method and ALS.

2. testScriptMultipleTimeseries.py: imputing and foreasting a multiple time series using synthetically generated data. The script uses both the SVD method and ALS.

3. testScriptSynthControlSVD.py: two real case studies for Robust Synthetic Control. Case studies and data taken from the pioneering works on Synthetic Control by Abadie et. al. This script uses the SVD method.

4. testScriptSynthControlALS.py: two real case studies for Robust Synthetic Control. Case studies and data taken from the pioneering works on Synthetic Control by Abadie et. al. This script uses the ALS method.

5. testScriptMultiSynthControlSVD.py: sample usage of the mRSC method with synthetically generated data. This script uses the SVD method.
