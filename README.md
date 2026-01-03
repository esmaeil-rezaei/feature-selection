# Feature Selection Methods

## Overview

This repository provides a complete suite of feature selection implementations for machine learning and data science practitioners. Each method is implemented in Jupyter notebooks with detailed explanations, code examples, and industrial use cases.

Feature selection is critical for:
- Reducing model complexity and training time
- Improving model interpretability
- Preventing overfitting
- Handling high-dimensional data
- Reducing computational costs in production

---

## Repository Structure

```
.
├── 0 Introduction
│   └── introduction.ipynb
├── 01 Filtering Methods
│   ├── 1.1 quasi-constant-filtering-basic.ipynb
│   ├── 1.2 quasi-constant filtering by feature-engine.ipynb
│   ├── 1.3 filtering-by-corr-basic.ipynb
│   ├── 1.4 filtering-by-corr-using-feature-engine.ipynb
│   ├── 1.5 filtering-pipeline-quasi-const-corr-feature-engine.ipynb
│   ├── 1.6 filtering-statistical-metrics.ipynb
│   ├── 1.7 filtering-chi-square-fisher-test.ipynb
│   ├── 1.8 filtering-ANOVA.ipynb
│   └── 1.9 filtering-univariate-ml.ipynb
├── 02 Wrapper Methods
│   ├── 01 introduction-to-wrappers.ipynb
│   ├── 02 wrapper-stepwise-forward.ipynb
│   ├── 03 wrapper-step-backward-elimination.ipynb
│   └── 04 wrapper-exhaustive-feature-selection.ipynb
├── 03 Embedded Methods
│   ├── 01 embedded-logistic-regression.ipynb
│   ├── 02 embedded-linear-regression.ipynb
│   ├── 03 effect-of-regularization-on-FS.ipynb
│   ├── 04 embedded-lasso-feature-selection.ipynb
│   ├── 05 embedded-tree-based-methods.ipynb
│   └── 06 embedded-tree-recursive.ipynb
├── 04 Hybrid Methods
│   ├── 01 shuffling.ipynb
│   ├── 02 recursive-feature-elimination.ipynb
│   ├── 03 recursive-feature-addition.ipynb
│   └── 04 maximum-relevance-minimum-redundancy.ipynb
├── 05 Time Series Feature Selection
│   ├── 01 mutual-information.ipynb
│   ├── 02 distance-correlation.ipynb
│   └── 03 hilbert-schmidt-independence-criterion.ipynb
└── README.md
```

---

## Method Categories Explained

### 1. Filtering Methods
Fast, model-agnostic techniques that evaluate features based on statistical properties and their relationship with the target variable.

**Implemented Techniques:**
- Quasi-constant removal
- Correlation-based filtering
- Integrated pipeline combining quasi-constant and correlation methods
- Mutual information
- Chi-square test
- ANOVA F-test
- Univariate ML metrics (ROC-AUC, MSE)

**When to use:**
- Large datasets where computational efficiency is critical
- Initial feature screening phase
- When model-agnostic selection is preferred
- Exploratory data analysis
- Production systems requiring fast preprocessing

**Advantages:**
- Computationally efficient
- Scalable to high-dimensional data
- Independent of ML algorithm
- Easy to interpret and explain

### 2. Wrapper Methods
Iterative search methods that evaluate feature subsets by training models and measuring performance.

**Implemented Techniques:**
- Step forward selection
- Step backward elimination
- Exhaustive feature selection

**When to use:**
- Small to medium-sized datasets
- When optimal performance is critical
- Sufficient computational resources available
- Model-specific feature selection needed
- When feature interactions are important

**Advantages:**
- Considers feature interactions
- Optimizes for specific model and metric
- Can find optimal feature combinations
- Accounts for model-specific behavior

**Considerations:**
- Computationally expensive
- Risk of overfitting on small datasets
- Requires cross-validation

### 3. Embedded Methods
Methods that perform feature selection during model training, integrating selection into the learning algorithm.

**Implemented Techniques:**
- Logistic regression coefficients
- Linear regression coefficients
- Regularization effects (Ridge, Lasso, Elastic Net)
- Lasso feature selection
- Random Forest feature importance
- Tree-based recursive elimination

**When to use:**
- Training linear or tree-based models
- High-dimensional data requiring automatic selection
- Production pipelines needing efficient selection
- When regularization is already being applied
- Real-time model training scenarios

**Advantages:**
- Computationally efficient (single training)
- Built into model training
- Naturally handles feature interactions
- Produces sparse models (Lasso)
- Provides interpretable importance scores (trees)

### 4. Hybrid Methods
Advanced techniques combining multiple approaches for robust, comprehensive feature selection.

**Implemented Techniques:**
- Feature shuffling (permutation importance)
- Recursive Feature Elimination (RFE)
- Recursive Feature Addition (RFA)
- Maximum Relevance Minimum Redundancy (mRMR)

**When to use:**
- Research and development projects
- When seeking optimal performance
- Model validation and testing
- Complex datasets with intricate relationships
- When combining strengths of multiple methods
- After deployment is already done

**Advantages:**
- More robust than single methods
- Validates feature importance multiple ways
- Handles complex feature relationships
- Reduces false discoveries
- Suitable for critical applications


### 5. Time Series Feature Selection 🕐
Specialized methods designed for temporal data where traditional independence assumptions don't hold.

**Implemented Techniques:**
- Mutual Information (time series adapted)
- Distance Correlation
- Hilbert-Schmidt Independence Criterion (HSIC)

**When to use:**
- Financial forecasting (stock prices, trading signals)
- Demand forecasting (sales, inventory)
- Sensor data analysis (IoT, manufacturing)
- Healthcare monitoring (vital signs, patient data)
- Energy consumption prediction
- Weather and climate modeling
- Time series classification tasks
- Sequential pattern recognition

**Key Considerations:**
- **Temporal dependencies**: Features may have lagged relationships
- **Autocorrelation**: Variables correlated with their own past values
- **Non-stationarity**: Statistical properties change over time
- **Seasonality**: Periodic patterns must be preserved
- **Causality**: Direction of influence matters (Granger causality)

**Advantages:**
- Captures temporal dependencies and lag relationships
- Detects non-linear relationships in sequential data
- Handles autocorrelated features properly
- Preserves temporal structure during selection
- Works with multivariate time series
- Robust to non-stationarity

**Time Series Specific Challenges:**
- Must respect temporal order (no data leakage)
- Need to handle multiple lag features
- Seasonal patterns require special treatment
- Feature importance changes over time windows
- Cross-validation requires time-aware splits

**Industrial Applications:**
- **Finance**: Selecting technical indicators for algorithmic trading, risk factor identification
- **Retail**: Demand forecasting with promotional and calendar features
- **Manufacturing**: Predictive maintenance using sensor readings and operational metrics
- **Energy**: Load forecasting with weather and historical consumption patterns
- **Healthcare**: Patient monitoring with vital signs and treatment history
- **Marketing**: Campaign effectiveness with time-lagged response variables



## Getting Started

### Prerequisites
```bash
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
feature-engine>=1.6.0
mlxtend>=0.19.0
scipy>=1.7.0
statsmodels>=0.13.0  # For time series analysis
dcor>=0.5.3          # For distance correlation
```



## Best Practices

### General Guidelines
1. **Always start with basic filtering** to clean your data
2. **Use pipelines** for reproducible, production-ready workflows
3. **Validate with multiple methods** from different categories
4. **Document feature selection rationale** for compliance and reproducibility
5. **Monitor feature importance in production** for drift detection
6. **Use cross-validation** when applying wrapper methods
7. **Consider computational budget** when choosing methods
8. **Align method choice with model type** (linear vs tree-based)
9. **Test on hold-out data** to validate selection stability
10. **Retrain selection periodically** as data distributions evolve

### Time Series Specific Guidelines
1. **Use time-aware cross-validation** (TimeSeriesSplit, expanding/rolling windows)
2. **Respect temporal order** - never use future data to predict the past
3. **Handle lag features carefully** - consider multiple time horizons
4. **Check for stationarity** before applying standard methods
5. **Account for seasonality** in feature importance evaluation
6. **Test feature stability** across different time periods
7. **Consider forecast horizon** when selecting features (short vs long-term)
8. **Validate on out-of-sample periods** that represent future conditions
9. **Monitor feature drift** over time in production
10. **Document temporal relationships** between features and targets


## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Add comprehensive documentation** for new methods
2. **Include industrial use cases** with specific examples
3. **Provide working code examples** with sample datasets
4. **Follow naming conventions** matching existing notebooks
5. **Add methods to appropriate category** folder
6. **Update this README** with new methods and applications
7. **Include performance benchmarks** where applicable
8. **Add references** to academic papers or original sources


## Performance Comparison

| Method Category | Speed | Accuracy | Scalability | Interpretability | Use Case |
|----------------|-------|----------|-------------|------------------|----------|
| Filtering | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | ⬆️⬆️⬆️⬆️⬆️ | 📊📊📊📊 | Large datasets, initial screening |
| Wrapper | ⚡⚡ | ⭐⭐⭐⭐⭐ | ⬆️⬆️ | 📊📊📊 | Small datasets, optimal performance |
| Embedded | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | ⬆️⬆️⬆️⬆️ | 📊📊📊📊 | Production systems, automatic selection |
| Hybrid | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ⬆️⬆️⬆️ | 📊📊📊📊📊 | Research, validation, critical systems |
| Time Series | ⚡⚡⚡ | ⭐⭐⭐⭐ | ⬆️⬆️⬆️ | 📊📊📊 | Temporal data, forecasting applications |

---

## Troubleshooting

### Common Issues

**Issue:** Quasi-constant filtering removes too many features
- **Solution:** Adjust threshold in `1.1` or `1.2`, check variance distribution

**Issue:** Correlation filtering removes important features
- **Solution:** Review correlation threshold, use domain knowledge, try `1.6` mutual information

**Issue:** Wrapper methods take too long
- **Solution:** Pre-filter with `01 Filtering Methods`, reduce feature space, use embedded methods instead

**Issue:** Embedded methods give unstable results
- **Solution:** Use cross-validation, try regularization in `03`, ensemble multiple runs

**Issue:** Different methods give different results
- **Solution:** This is expected - combine results, use `04 Hybrid Methods` for consensus

### Time Series Specific Issues

**Issue:** Standard methods ignore temporal relationships
- **Solution:** Use `05 Time Series` methods specifically designed for temporal dependencies

**Issue:** Feature importance changes across time periods
- **Solution:** Perform rolling window feature selection, monitor stability metrics

**Issue:** Lag features showing spurious importance
- **Solution:** Check for autocorrelation, use distance correlation (`05/02`), test with different lag values

**Issue:** Data leakage in time series cross-validation
- **Solution:** Always use TimeSeriesSplit or forward-chaining validation, never shuffle temporal data


## FAQ

**Q: Which method should I start with?**
A: Always start with `01 Filtering Methods` for data cleaning, then choose based on dataset size and computational budget. For time series, add `05 Time Series` methods after basic filtering.

**Q: Can I combine methods from different categories?**
A: Yes! Sequential application is recommended: Filtering → Time Series (if applicable) → Embedded → Wrapper/Hybrid for validation.

**Q: How do I choose the right threshold?**
A: Use cross-validation to evaluate performance at different thresholds, consider business constraints and domain expertise.

**Q: Should I select features separately for train/test?**
A: No! Fit feature selection on training data only, then transform test data using the same selection.

**Q: How many features should I select?**
A: Depends on model complexity, interpretability needs, and performance goals. Start with top 20-30% and iterate.

**Q: Do I need to select features if using tree-based models?**
A: Trees handle irrelevant features naturally, but selection still helps with: interpretability, speed, preventing overfitting, and reducing storage.

**Q: How do time series methods differ from standard methods?**
A: Time series methods preserve temporal structure, handle autocorrelation properly, and can detect lagged relationships that standard methods miss.

**Q: Can I use standard feature selection on time series data?**
A: You can, but you'll miss temporal dependencies and lagged relationships. Always complement with `05 Time Series` methods for temporal data.

**Q: How do I handle seasonality in feature selection?**
A: Include seasonal features (month, quarter, day of week), test feature importance within each season separately, or use deseasonalized data for selection.


## References

### Academic Papers
- Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. JMLR
- Kohavi, R., & John, G. H. (1997). Wrappers for feature subset selection. Artificial Intelligence
- Ding, C., & Peng, H. (2005). Minimum redundancy feature selection from microarray gene expression data. JBI
- Székely, G. J., & Rizzo, M. L. (2009). Brownian distance covariance. The Annals of Applied Statistics
- Gretton, A., et al. (2005). Measuring statistical dependence with Hilbert-Schmidt norms. ALT

### Libraries
- Scikit-learn: https://scikit-learn.org
- Feature-Engine: https://feature-engine.readthedocs.io
- MLxtend: http://rasbt.github.io/mlxtend/
- Statsmodels: https://www.statsmodels.org
- dcor: https://dcor.readthedocs.io

## Roadmap

### Current Features ✅
- Complete filtering methods suite
- Wrapper methods implementation
- Embedded methods with regularization
- Hybrid methods for validation
- **Time series feature selection methods** 🆕

### Planned Additions
- [ ] Deep learning feature selection methods
- [ ] Automated feature selection with AutoML
- [ ] Time series specific feature selection
- [ ] Text data feature selection techniques
- [ ] Image feature selection methods
- [ ] Multi-objective feature selection
- [ ] Distributed feature selection for big data
- [ ] Feature selection for imbalanced datasets