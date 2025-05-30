# Predictive Maintenance for Turbofan Engine Remaining Useful Life (RUL) Prediction

## Abstract
This project develops and evaluates machine learning models to predict the Remaining Useful Life (RUL) of turbofan engines using the NASA Turbofan Engine Degradation dataset. We compare a baseline Random Forest, a tuned Random Forest, LightGBM, and a GPU-accelerated XGBoost model. Our best model (GPU XGBoost) achieves MAE = 46.69 cycles and RMSE = 62.82 cycles, demonstrating promising accuracy for predictive maintenance applications.

## Rationale
Timely and accurate RUL estimation allows maintenance teams to plan interventions before critical failures occur, reducing unplanned downtime and maintenance costs. Traditional condition-based monitoring often relies on simple thresholds, which cannot capture complex degradation patterns. Machine learning offers a data-driven approach to learn those patterns directly from historical sensor readings.

## Research Question
> **Can we build and deploy a data-driven model that accurately forecasts the remaining useful life of turbofan engines, and how does GPU-accelerated XGBoost compare to tree-based baselines?**

## Data Sources
- **NASA Turbofan Engine Degradation Dataset** (FD001–FD004):
  - `train_FD00x.txt`: multivariate time series of 21 sensor measurements + 3 operational settings per engine cycle
  - `RUL_FD00x.txt`: true remaining life in cycles for each engine at its last recorded cycle
  - `test_FD00x.txt` & `RUL_FD00x.txt`: test inputs and ground-truth labels
- Total of ~160 K training rows across 100 engines.

## Methodology
1. **Data Preprocessing**  
   - Merged all FD00x subsets into `processed_all.csv`  
   - Computed per-engine sliding-window statistics (mean, std, min, max over last 20 cycles)  
   - Generated RUL labels by subtracting current cycle from engine’s max cycle  

2. **Feature Engineering**  
   - Added polynomial features of `cycle` (² and ³)  
   - First-order differences of sensors and operational settings to capture change rates  

3. **Model Training & Tuning**  
   - **Baseline Random Forest** (n_estimators=100)  
   - **Tuned Random Forest** via RandomizedSearchCV on subsampled data  
   - **LightGBM** with early stopping (200 rounds)  
   - **GPU XGBoost** hyperparameter search using `tree_method='gpu_hist'`  

4. **Evaluation**  
   - Metrics: MAE, RMSE, R²  
   - Visualizations:  
     - True vs. Predicted RUL scatter  
     - Residual histograms, residual vs. true RUL/cycle, and error CDF  
   - Interval analysis by RUL bands (≤20, 21–50, >50 cycles)

## Results
| Model               | MAE    | RMSE   | R²    |
|---------------------|--------|--------|-------|
| Baseline RF         | 48.10  | 63.79  | 0.426 |
| Tuned RF            | 47.20  | 63.52  | 0.435 |
| LightGBM            | 46.85  | 63.01  | 0.442 |
| GPU XGBoost         | 46.69  | 62.82  | 0.439 |

**Error by RUL Interval**  
| Interval     | Count  | MAE    | RMSE   |
|--------------|--------|--------|--------|
| RUL ≤ 20     |  2,820 | 36.63  | 46.66  |
| 20 < RUL ≤ 50|  4,230 | 37.00  | 47.20  |
| RUL > 50     | 24,864 | 50.13  | 67.47  |

- The model performs best in the low-RUL range, with errors rising sharply for long-horizon predictions.
- Residual analyses reveal a tendency to overestimate RUL at early cycles and underestimate at late cycles.

## Next Steps
- **Residual Correction**: Introduce a secondary calibration model for high-RUL predictions to reduce long-tail errors.  
- **Deep Sequence Models**: Compare against LSTM/Transformer architectures to capture long-term temporal dependencies.  
- **Online Monitoring**: Deploy the GPU XGBoost model with a real-time feature pipeline and threshold-based alerts.  
- **Periodic Retraining**: Schedule retraining as new engine data arrives to adapt to evolving degradation patterns.

## Conclusion
We have demonstrated a complete ML workflow—from data preprocessing to model deployment recommendations—for predictive maintenance of turbofan engines. The GPU-accelerated XGBoost model offers the best balance of accuracy and speed, making it the recommended choice for production use, with LightGBM and Random Forest as viable backups.
