# Titanic Survival Analysis in R

A complete, end-to-end data analysis project built in R as part of a 4-week Virtual R Data Analyst Internship. The project takes the Titanic passenger dataset from raw, messy data through cleaning, exploratory visualization, statistical hypothesis testing, and predictive modeling.

## Dataset

- **Source:** [Titanic dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) (public domain, derived from the well-known Kaggle Titanic competition)
- **Size:** 891 passenger records, 12 original variables
- **Why this dataset:** it has meaningful missing data (Age, Cabin, Embarked), a mix of numerical and categorical variables, and a clear binary outcome (`Survived`) suitable for classification modeling.

## Project Structure

```
├── README.md
├── titanic.csv                              # raw dataset
├── titanic_cleaned.csv                      # cleaned dataset output from Week 1 (used by Weeks 2–3)
│
├── week1_analysis.R                         # Data cleaning & preliminary analysis
├── week2_visualization.R                    # ggplot2 visualizations
├── week3_modeling.R                         # Hypothesis testing & logistic regression
│
├── Week1_Data_Cleaning_Report.docx
├── Week2_Data_Visualization_Report.docx
├── Week3_Statistical_Modeling_Report.docx
└── Week4_Final_Comprehensive_Report.docx    # consolidated final report (Weeks 1–3 synthesis)
```

## Week-by-Week Summary

### Week 1 — Data Cleaning and Preliminary Analysis
- Identified missing values: `Age` (19.87%), `Cabin` (77.10%), `Embarked` (0.22%)
- Median imputation for `Age`, mode imputation for `Embarked`
- Engineered `HasCabin` binary feature instead of imputing the mostly-missing `Cabin` column
- Detected Fare outliers via IQR method (116 flagged); treated via winsorizing at the 99th percentile
- Min-max normalized `Age` and `Fare`; encoded `Sex` (binary), `Embarked` (one-hot), `Pclass` (labeled factor)
- Output: `titanic_cleaned.csv` (891 rows, 21 columns, zero missing values)

**Run:** `Rscript week1_analysis.R`

### Week 2 — Data Visualization and Insight Communication
Six ggplot2 visualizations, each chosen to match a specific analytical question:
1. Bar chart — survival counts by passenger class
2. Scatter plot — Age vs. Fare, colored by survival
3. Faceted histogram — age distribution by survival outcome
4. Line chart — survival rate trend across age groups
5. Grouped bar chart — survival rate by sex and class
6. Boxplot — fare distribution by class

**Key finding:** the female survival advantage holds within every class but widens sharply in Third class (50% vs 13.5% for men).

**Run:** `Rscript week2_visualization.R` (requires `ggplot2`)

### Week 3 — Statistical Analysis and Predictive Modeling
- **Hypothesis tests:** Shapiro-Wilk (normality), Wilcoxon rank-sum (Fare vs. survival), Chi-square (Sex vs. survival; Class vs. survival), Pearson correlation (Age vs. Fare) — all significant at p < 0.05 except the Age-Fare relationship, which is statistically significant but practically negligible (r = 0.097)
- **Model:** Logistic regression (`Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + HasCabin`) on a 70/30 train-test split, validated with 5-fold cross-validation
- **Performance:** 79.1% test accuracy, AUC 0.844, mean CV accuracy 80.1% (SD 4.1%)
- **Strongest predictor:** Sex (odds ratio ≈ 16.4 — being female multiplied survival odds by ~16x, controlling for other variables)

**Run:** `Rscript week3_modeling.R`

### Week 4 — Comprehensive Final Report
Consolidates all three weeks into a single narrative (Introduction, Methodology, Data Preparation, Visualization, Statistical Analysis & Modeling, Discussion, Challenges, Conclusion & Recommendations). No new analysis — see `Week4_Final_Comprehensive_Report.docx`.

## Requirements

- R (≥ 4.0)
- Base R only for Weeks 1 and 3
- `ggplot2` for Week 2:
  ```r
  install.packages("ggplot2")
  ```

## Reproducing the Analysis

```bash
# 1. Clean the data
Rscript week1_analysis.R

# 2. Generate visualizations (depends on titanic_cleaned.csv from step 1)
Rscript week2_visualization.R

# 3. Run statistical tests and build the predictive model (depends on titanic_cleaned.csv)
Rscript week3_modeling.R
```

All three scripts print full console output (structure, summaries, test results, model diagnostics) and save any generated plots to a local `plots/` subfolder.

## Key Insights

- Overall survival rate: 38.4%
- Survival by sex: 74.2% (female) vs. 18.9% (male)
- Survival by class: 63.0% (First) vs. 47.3% (Second) vs. 24.2% (Third)
- Children aged 12 and under survived at 58.0% vs. 36.7% for everyone older
- The strongest predictor of survival in the fitted model is Sex, followed by passenger class and cabin presence

## Author

Raju — Virtual R Data Analyst Intern, YuvaIntern
