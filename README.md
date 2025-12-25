# Feature Selection Methods: A Comprehensive Implementation Guide

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
│   ├── 1.7 filtering-chi-square.ipynb
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

---

## Notebook Descriptions

### 0 Introduction
| Notebook | Description |
|----------|-------------|
| `introduction.ipynb` | Comprehensive overview of feature selection paradigms, comparison of methods, decision framework for method selection, and best practices |

### 01 Filtering Methods
| Notebook | Description |
|----------|-------------|
| `1.1 quasi-constant-filtering-basic.ipynb` | Manual implementation of quasi-constant feature detection using variance thresholds |
| `1.2 quasi-constant filtering by feature-engine.ipynb` | Production-ready quasi-constant removal using Feature-Engine library |
| `1.3 filtering-by-corr-basic.ipynb` | Basic correlation matrix analysis and feature removal implementation |
| `1.4 filtering-by-corr-using-feature-engine.ipynb` | Scalable correlation filtering with Feature-Engine DropCorrelatedFeatures |
| `1.5 filtering-pipeline-quasi-const-corr-feature-engine.ipynb` | Complete preprocessing pipeline combining quasi-constant and correlation filtering |
| `1.6 filtering-statistical-metrics.ipynb` | Mutual information for feature ranking and selection |
| `1.7 filtering-chi-square.ipynb` | Chi-square test for categorical feature selection |
| `1.8 filtering-ANOVA.ipynb` | ANOVA F-test for continuous features in classification |
| `1.9 filtering-univariate-ml.ipynb` | Univariate metrics (ROC-AUC, MSE) for feature ranking |

### 02 Wrapper Methods
| Notebook | Description |
|----------|-------------|
| `01 introduction-to-wrappers.ipynb` | Overview of wrapper methods, computational complexity, and use case guidance |
| `02 wrapper-stepwise-forward.ipynb` | Forward selection algorithm with cross-validation |
| `03 wrapper-step-backward-elimination.ipynb` | Backward elimination starting from full feature set |
| `04 wrapper-exhaustive-feature-selection.ipynb` | Brute-force search across all feature combinations |

### 03 Embedded Methods
| Notebook | Description |
|----------|-------------|
| `01 embedded-logistic-regression.ipynb` | Using logistic regression coefficients for feature importance |
| `02 embedded-linear-regression.ipynb` | Linear regression coefficient analysis for feature selection |
| `03 effect-of-regularization-on-FS.ipynb` | Comparing Ridge, Lasso, and Elastic Net effects on feature selection |
| `04 embedded-lasso-feature-selection.ipynb` | L1 regularization for automatic feature selection |
| `05 embedded-tree-based-methods.ipynb` | Random Forest, Gradient Boosting feature importance metrics |
| `06 embedded-tree-recursive.ipynb` | Combining tree importance with recursive elimination |

### 04 Hybrid Methods
| Notebook | Description |
|----------|-------------|
| `01 shuffling.ipynb` | Permutation importance through feature shuffling |
| `02 recursive-feature-elimination.ipynb` | RFE algorithm with multiple estimators (SVM, RF, etc.) |
| `03 recursive-feature-addition.ipynb` | Forward-style recursive addition of features |
| `04 maximum-relevance-minimum-redundancy.ipynb` | Information theory-based mRMR algorithm |

---

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
```


---

## Best Practices

### General Guidelines
1. **Always start with basic filtering** (`1.1`, `1.2`, `1.3`, `1.4`) to clean your data
2. **Use pipelines** (`1.5`) for reproducible, production-ready workflows
3. **Validate with multiple methods** from different categories
4. **Document feature selection rationale** for compliance and reproducibility
5. **Monitor feature importance in production** for drift detection
6. **Use cross-validation** when applying wrapper methods
7. **Consider computational budget** when choosing methods
8. **Align method choice with model type** (linear vs tree-based)
9. **Test on hold-out data** to validate selection stability
10. **Retrain selection periodically** as data distributions evolve

### Method Selection Framework

**For Large Datasets (>100K rows, >100 features):**
- Start: `1.1`, `1.2` (quasi-constant removal)
- Then: `1.3`, `1.4` (correlation filtering)
- Apply: `1.6`, `1.7`, `1.8`, `1.9` (statistical filters)
- Use: `04` (Lasso) or `05` (tree importance) for final selection

**For Medium Datasets (10K-100K rows, 20-100 features):**
- Start: Basic filtering (`1.1`-`1.4`)
- Apply: `02` (wrapper methods) with cross-validation
- Or use: `03` (embedded methods) for efficiency
- Validate: `04` (hybrid methods)

**For Small Datasets (<10K rows, <20 features):**
- Clean: Basic filtering (`1.1`-`1.4`)
- Use: `02` (exhaustive search) if computationally feasible
- Apply: `01 shuffling.ipynb` for importance validation
- Consider: Domain expertise over pure algorithmic selection

---

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

---

## Performance Comparison

| Method Category | Speed | Accuracy | Scalability | Interpretability | Use Case |
|----------------|-------|----------|-------------|------------------|----------|
| Filtering | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | ⬆️⬆️⬆️⬆️⬆️ | 📊📊📊📊 | Large datasets, initial screening |
| Wrapper | ⚡⚡ | ⭐⭐⭐⭐⭐ | ⬆️⬆️ | 📊📊📊 | Small datasets, optimal performance |
| Embedded | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | ⬆️⬆️⬆️⬆️ | 📊📊📊📊 | Production systems, automatic selection |
| Hybrid | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ⬆️⬆️⬆️ | 📊📊📊📊📊 | Research, validation, critical systems |

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

---

## FAQ

**Q: Which method should I start with?**
A: Always start with `1. filtering` for data cleaning, then choose based on dataset size and computational budget.

**Q: Can I combine methods from different categories?**
A: Yes! Sequential application is recommended: Filtering → Embedded → Wrapper/Hybrid for validation.

**Q: How do I choose the right threshold?**
A: Use cross-validation to evaluate performance at different thresholds, consider business constraints and domain expertise.

**Q: Should I select features separately for train/test?**
A: No! Fit feature selection on training data only, then transform test data using the same selection.

**Q: How many features should I select?**
A: Depends on model complexity, interpretability needs, and performance goals. Start with top 20-30% and iterate.

**Q: Do I need to select features if using tree-based models?**
A: Trees handle irrelevant features naturally, but selection still helps with: interpretability, speed, preventing overfitting, and reducing storage.

---

## References

### Academic Papers
- Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. JMLR.
- Kohavi, R., & John, G. H. (1997). Wrappers for feature subset selection. Artificial Intelligence.
- Ding, C., & Peng, H. (2005). Minimum redundancy feature selection from microarray gene expression data. JBI.

### Libraries
- Scikit-learn: https://scikit-learn.org
- Feature-Engine: https://feature-engine.readthedocs.io
- MLxtend: http://rasbt.github.io/mlxtend/

---


### Planned Additions
- [ ] Deep learning feature selection methods
- [ ] Automated feature selection with AutoML
- [ ] Time series specific feature selection
- [ ] Text data feature selection techniques
- [ ] Image feature selection methods
- [ ] Multi-objective feature selection
- [ ] Distributed feature selection for big data
- [ ] Feature selection for imbalanced datasets