
# Chemical Process Optimization with ML Models

This project applies machine learning models to predict the activity of chemical compounds based on SMILES data.

## Models Used:
- **Random Forest**
- **Decision Tree**
- **Neural Network**

## Results:
The Mean Squared Error (MSE) for each model is as follows:
- Random Forest: 1461.61
- Decision Tree: 1549.85
- Neural Network: 1217.16

Neural Networks performed the best with the lowest MSE.

## How to Run:
1. Install necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the main script:
   ```bash
   python main.py
   ```

## Visualization:
The MSE comparison of the models is visualized in the `mse_comparison_plot.png`.
