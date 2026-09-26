# =========================================================
# Week 1 Task: Data Cleaning and Preliminary Analysis with R
# Dataset: Titanic passenger data (Kaggle / public domain)
# =========================================================

# ---- 0. Setup ----
df <- read.csv("/home/claude/titanic.csv", stringsAsFactors = FALSE)

cat("===== STEP 1: Initial structure of the raw dataset =====\n")
str(df)

cat("\n===== First 6 rows (raw) =====\n")
print(head(df))

cat("\n===== Dimensions =====\n")
print(dim(df))

# ---- 1. Missing value detection ----
cat("\n===== STEP 2: Missing values per column (before cleaning) =====\n")
# Empty strings in Cabin/Embarked also count as missing, so check both NA and ""
missing_summary <- sapply(df, function(x) sum(is.na(x) | x == ""))
print(missing_summary)

cat("\nPercentage missing:\n")
print(round(100 * missing_summary / nrow(df), 2))

# ---- 2. Handling missing values ----
cat("\n===== STEP 3: Handling missing values =====\n")

# Age: numeric, ~19.9% missing -> impute with median (robust to skew/outliers)
age_median <- median(df$Age, na.rm = TRUE)
cat("Median Age used for imputation:", age_median, "\n")
df$Age[is.na(df$Age)] <- age_median

# Embarked: categorical, only 2 missing -> impute with mode (most frequent port)
embarked_table <- table(df$Embarked[df$Embarked != ""])
print(embarked_table)
embarked_mode <- names(embarked_table)[which.max(embarked_table)]
cat("Mode of Embarked used for imputation:", embarked_mode, "\n")
df$Embarked[df$Embarked == ""] <- embarked_mode

# Cabin: ~77% missing -> too sparse to impute meaningfully.
# Instead of dropping the column entirely, engineer a binary feature: HasCabin
df$HasCabin <- ifelse(df$Cabin == "", 0, 1)

cat("\nMissing values after cleaning:\n")
print(sapply(df[, c("Age", "Embarked")], function(x) sum(is.na(x) | x == "")))

# ---- 3. Outlier detection ----
cat("\n===== STEP 4: Outlier detection (Fare and Age) =====\n")

detect_outliers_iqr <- function(x) {
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  lower <- q1 - 1.5 * iqr
  upper <- q3 + 1.5 * iqr
  sum(x < lower | x > upper, na.rm = TRUE)
}

cat("Fare summary:\n")
print(summary(df$Fare))
cat("Number of Fare outliers (IQR method):", detect_outliers_iqr(df$Fare), "\n")

cat("\nAge summary:\n")
print(summary(df$Age))
cat("Number of Age outliers (IQR method):", detect_outliers_iqr(df$Age), "\n")

# Cap extreme Fare outliers at the 99th percentile (winsorizing) rather than deleting rows,
# to preserve sample size while limiting the influence of extreme values
fare_cap <- quantile(df$Fare, 0.99, na.rm = TRUE)
cat("Capping Fare at 99th percentile:", round(fare_cap, 2), "\n")
df$Fare_capped <- ifelse(df$Fare > fare_cap, fare_cap, df$Fare)

# ---- 4. Normalization ----
cat("\n===== STEP 5: Normalization (min-max scaling) =====\n")

min_max_scale <- function(x) (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))

df$Age_scaled  <- min_max_scale(df$Age)
df$Fare_scaled <- min_max_scale(df$Fare_capped)

cat("Age_scaled summary:\n")
print(summary(df$Age_scaled))
cat("\nFare_scaled summary:\n")
print(summary(df$Fare_scaled))

# ---- 5. Encoding categorical variables ----
cat("\n===== STEP 6: Encoding categorical variables =====\n")

# Sex: binary label encoding
df$Sex_encoded <- ifelse(df$Sex == "male", 1, 0)

# Embarked: one-hot encoding
embarked_dummies <- model.matrix(~ Embarked - 1, data = df)
colnames(embarked_dummies) <- paste0("Embarked_", sub("Embarked", "", colnames(embarked_dummies)))
df <- cbind(df, embarked_dummies)

# Pclass: treat as ordered factor (already numeric 1/2/3, but label for clarity)
df$Pclass_label <- factor(df$Pclass, levels = c(1, 2, 3),
                           labels = c("First", "Second", "Third"))

cat("Encoded columns preview:\n")
print(head(df[, c("Sex", "Sex_encoded", "Embarked", "Embarked_C", "Embarked_Q", "Embarked_S", "Pclass_label")]))

# ---- 6. Exploratory Data Analysis ----
cat("\n\n===== STEP 7: Exploratory Data Analysis =====\n")

cat("\n--- str() of cleaned dataset ---\n")
str(df)

cat("\n--- summary() of key variables ---\n")
print(summary(df[, c("Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "HasCabin")]))

cat("\n--- Survival rate overall ---\n")
print(round(prop.table(table(df$Survived)) * 100, 1))

cat("\n--- Survival rate by Sex ---\n")
print(round(prop.table(table(df$Sex, df$Survived), margin = 1) * 100, 1))

cat("\n--- Survival rate by Passenger Class ---\n")
print(round(prop.table(table(df$Pclass_label, df$Survived), margin = 1) * 100, 1))

cat("\n--- Correlation matrix (numeric variables) ---\n")
num_vars <- df[, c("Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "HasCabin", "Sex_encoded")]
corr_matrix <- round(cor(num_vars, use = "complete.obs"), 2)
print(corr_matrix)

# ---- 7. Visualizations ----
cat("\n===== STEP 8: Generating visualizations =====\n")

plot_dir <- "/home/claude/week1/plots"

# Age distribution
png(file.path(plot_dir, "age_distribution.png"), width = 800, height = 600, res = 110)
hist(df$Age, breaks = 30, col = "#4C72B0", border = "white",
     main = "Distribution of Passenger Age (post-imputation)",
     xlab = "Age", ylab = "Frequency")
abline(v = age_median, col = "red", lwd = 2, lty = 2)
legend("topright", legend = paste("Median =", age_median), col = "red", lty = 2, lwd = 2, bty = "n")
dev.off()

# Fare boxplot (outlier visualization)
png(file.path(plot_dir, "fare_boxplot.png"), width = 800, height = 600, res = 110)
boxplot(df$Fare, main = "Boxplot of Fare (showing outliers)",
        ylab = "Fare", col = "#DD8452", horizontal = TRUE)
dev.off()

# Survival counts by sex
png(file.path(plot_dir, "survival_by_sex.png"), width = 800, height = 600, res = 110)
counts <- table(df$Sex, df$Survived)
barplot(counts, beside = TRUE, col = c("#4C72B0", "#DD8452"),
        main = "Survival Count by Sex", xlab = "Survived (0 = No, 1 = Yes)",
        ylab = "Count", legend.text = rownames(counts),
        names.arg = c("No", "Yes"))
dev.off()

# Survival rate by class
png(file.path(plot_dir, "survival_by_class.png"), width = 800, height = 600, res = 110)
class_surv <- prop.table(table(df$Pclass_label, df$Survived), margin = 1)[, "1"] * 100
barplot(class_surv, col = "#55A868",
        main = "Survival Rate (%) by Passenger Class",
        ylab = "Survival Rate (%)", ylim = c(0, 100))
dev.off()

# Correlation heatmap (base R)
png(file.path(plot_dir, "correlation_heatmap.png"), width = 800, height = 700, res = 110)
image(1:ncol(corr_matrix), 1:nrow(corr_matrix), t(corr_matrix)[, nrow(corr_matrix):1],
      axes = FALSE, xlab = "", ylab = "", main = "Correlation Heatmap",
      col = colorRampPalette(c("#4C72B0", "white", "#C44E52"))(50))
axis(1, at = 1:ncol(corr_matrix), labels = colnames(corr_matrix), las = 2, cex.axis = 0.7)
axis(2, at = 1:nrow(corr_matrix), labels = rev(rownames(corr_matrix)), las = 2, cex.axis = 0.7)
for (i in 1:nrow(corr_matrix)) {
  for (j in 1:ncol(corr_matrix)) {
    text(j, nrow(corr_matrix) - i + 1, corr_matrix[i, j], cex = 0.65)
  }
}
dev.off()

cat("\nAll plots saved to:", plot_dir, "\n")

# ---- 8. Save cleaned dataset ----
write.csv(df, "/home/claude/week1/titanic_cleaned.csv", row.names = FALSE)
cat("\nCleaned dataset saved as titanic_cleaned.csv\n")

cat("\n===== SCRIPT COMPLETE =====\n")
