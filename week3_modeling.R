# =========================================================
# Week 3 Task: Statistical Analysis and Predictive Modeling with R
# Dataset: Titanic passenger data (cleaned in Week 1)
# =========================================================

set.seed(42)  # reproducibility for train/test split

df <- read.csv("/home/claude/week1/titanic_cleaned.csv", stringsAsFactors = FALSE)
df$Pclass_label   <- factor(df$Pclass_label, levels = c("First", "Second", "Third"))
df$Sex            <- factor(df$Sex, levels = c("male", "female"))
df$Survived       <- as.integer(df$Survived)

cat("===== Dataset loaded =====\n")
cat("Rows:", nrow(df), " Columns:", ncol(df), "\n\n")

plot_dir <- "/home/claude/week3/plots"

# =========================================================
# PART 1: Exploratory statistical analysis / hypothesis testing
# =========================================================

cat("\n########## PART 1: HYPOTHESIS TESTING ##########\n")

# ---- 1a. Normality test on Age and Fare ----
cat("\n===== Shapiro-Wilk normality tests =====\n")
# Shapiro-Wilk max sample 5000; we have 891 so fine directly
age_shapiro <- shapiro.test(df$Age)
fare_shapiro <- shapiro.test(df$Fare)
cat("Age  -> W =", round(age_shapiro$statistic, 4), " p-value =", format.pval(age_shapiro$p.value, digits = 4), "\n")
cat("Fare -> W =", round(fare_shapiro$statistic, 4), " p-value =", format.pval(fare_shapiro$p.value, digits = 4), "\n")
cat("Interpretation: p < 0.05 for both => reject H0 of normality; neither Age nor Fare is normally distributed.\n")
cat("(Fare is heavily right-skewed; Age has the imputation spike at the median.)\n")

# ---- 1b. Hypothesis test: does Fare differ by Survival? (Wilcoxon, since non-normal) ----
cat("\n===== Wilcoxon rank-sum test: Fare by Survival status =====\n")
cat("H0: Fare distribution is the same for survivors and non-survivors\n")
cat("H1: Fare distribution differs between survivors and non-survivors\n")
wilcox_fare <- wilcox.test(Fare ~ Survived, data = df)
print(wilcox_fare)
cat("Interpretation: p-value", format.pval(wilcox_fare$p.value, digits = 4),
    "<< 0.05 => reject H0. Survivors paid significantly higher fares.\n")

# ---- 1c. Hypothesis test: is survival independent of Sex? (Chi-square) ----
cat("\n===== Chi-square test of independence: Survival vs Sex =====\n")
cat("H0: Survival is independent of Sex\n")
cat("H1: Survival is associated with Sex\n")
sex_table <- table(df$Sex, df$Survived)
chi_sex <- chisq.test(sex_table)
print(chi_sex)
cat("Interpretation: p-value", format.pval(chi_sex$p.value, digits = 4),
    "<< 0.05 => reject H0. Survival is strongly associated with sex.\n")

# ---- 1d. Hypothesis test: is survival independent of Passenger Class? (Chi-square) ----
cat("\n===== Chi-square test of independence: Survival vs Pclass =====\n")
cat("H0: Survival is independent of Passenger Class\n")
cat("H1: Survival is associated with Passenger Class\n")
class_table <- table(df$Pclass_label, df$Survived)
chi_class <- chisq.test(class_table)
print(chi_class)
cat("Interpretation: p-value", format.pval(chi_class$p.value, digits = 4),
    "<< 0.05 => reject H0. Survival is strongly associated with passenger class.\n")

# ---- 1e. Correlation significance test: Age vs Fare ----
cat("\n===== Pearson correlation test: Age vs Fare =====\n")
cor_test <- cor.test(df$Age, df$Fare)
print(cor_test)
cat("Interpretation: correlation is weak (r =", round(cor_test$estimate, 3),
    ") though statistically significant (p =", format.pval(cor_test$p.value, digits = 4),
    ") due to the large sample size; practical relationship is negligible.\n")

# =========================================================
# PART 2: Model building — Logistic Regression Classification
# =========================================================

cat("\n\n########## PART 2: PREDICTIVE MODEL BUILDING ##########\n")

# ---- 2a. Train/test split (70/30) ----
cat("\n===== Train/test split (70/30) =====\n")
n <- nrow(df)
train_idx <- sample(seq_len(n), size = 0.7 * n)
train <- df[train_idx, ]
test  <- df[-train_idx, ]
cat("Training set:", nrow(train), "rows\n")
cat("Test set:    ", nrow(test), "rows\n")

# ---- 2b. Fit logistic regression model ----
cat("\n===== Logistic regression model =====\n")
model <- glm(Survived ~ Pclass_label + Sex + Age + SibSp + Parch + Fare + HasCabin,
             data = train, family = binomial(link = "logit"))
print(summary(model))

# ---- 2c. 5-fold cross-validation (manual, base R) ----
cat("\n===== 5-fold cross-validation on training set =====\n")
k <- 5
folds <- sample(rep(1:k, length.out = nrow(train)))
cv_accuracies <- numeric(k)

for (i in 1:k) {
  cv_train <- train[folds != i, ]
  cv_valid <- train[folds == i, ]
  cv_model <- glm(Survived ~ Pclass_label + Sex + Age + SibSp + Parch + Fare + HasCabin,
                   data = cv_train, family = binomial(link = "logit"))
  cv_probs <- predict(cv_model, newdata = cv_valid, type = "response")
  cv_preds <- ifelse(cv_probs > 0.5, 1, 0)
  cv_accuracies[i] <- mean(cv_preds == cv_valid$Survived)
}
cat("Per-fold accuracy:", round(cv_accuracies, 4), "\n")
cat("Mean CV accuracy:", round(mean(cv_accuracies), 4), " (SD:", round(sd(cv_accuracies), 4), ")\n")

# ---- 2d. Evaluate on held-out test set ----
cat("\n===== Test set evaluation =====\n")
test_probs <- predict(model, newdata = test, type = "response")
test_preds <- ifelse(test_probs > 0.5, 1, 0)

conf_matrix <- table(Predicted = test_preds, Actual = test$Survived)
cat("Confusion Matrix:\n")
print(conf_matrix)

TP <- conf_matrix["1", "1"]
TN <- conf_matrix["0", "0"]
FP <- conf_matrix["1", "0"]
FN <- conf_matrix["0", "1"]

accuracy  <- (TP + TN) / sum(conf_matrix)
precision <- TP / (TP + FP)
recall    <- TP / (TP + FN)
f1        <- 2 * precision * recall / (precision + recall)
specificity <- TN / (TN + FP)

cat("\nPerformance metrics on test set:\n")
cat("Accuracy   :", round(accuracy, 4), "\n")
cat("Precision  :", round(precision, 4), "\n")
cat("Recall     :", round(recall, 4), "\n")
cat("Specificity:", round(specificity, 4), "\n")
cat("F1 Score   :", round(f1, 4), "\n")

# ---- 2e. Manual ROC curve + AUC (base R, no external package) ----
cat("\n===== ROC curve and AUC (manually computed) =====\n")
thresholds <- seq(0, 1, by = 0.01)
roc_points <- data.frame(threshold = thresholds, tpr = NA, fpr = NA)

for (t in seq_along(thresholds)) {
  pred_t <- ifelse(test_probs > thresholds[t], 1, 0)
  tp <- sum(pred_t == 1 & test$Survived == 1)
  fp <- sum(pred_t == 1 & test$Survived == 0)
  fn <- sum(pred_t == 0 & test$Survived == 1)
  tn <- sum(pred_t == 0 & test$Survived == 0)
  roc_points$tpr[t] <- tp / (tp + fn)
  roc_points$fpr[t] <- fp / (fp + tn)
}

# Order by FPR for correct AUC integration (trapezoidal rule)
roc_ordered <- roc_points[order(roc_points$fpr), ]
auc <- sum(diff(roc_ordered$fpr) * (head(roc_ordered$tpr, -1) + tail(roc_ordered$tpr, -1)) / 2)
cat("AUC (Area Under ROC Curve):", round(auc, 4), "\n")

# =========================================================
# PART 3: Diagnostic plots
# =========================================================

cat("\n\n########## PART 3: DIAGNOSTIC VISUALIZATIONS ##########\n")

# ---- Plot 1: ROC curve ----
png(file.path(plot_dir, "roc_curve.png"), width = 800, height = 700, res = 130)
plot(roc_points$fpr, roc_points$tpr, type = "l", col = "#4C72B0", lwd = 2.5,
     xlab = "False Positive Rate", ylab = "True Positive Rate",
     main = paste0("ROC Curve (AUC = ", round(auc, 3), ")"))
abline(0, 1, col = "gray60", lty = 2)
legend("bottomright", legend = c("Model", "Random guess"), col = c("#4C72B0", "gray60"), lty = c(1, 2), lwd = 2, bty = "n")
dev.off()

# ---- Plot 2: Confusion matrix heatmap ----
# Build an explicit data frame of the four cells: (Predicted, Actual, Count)
# so plotting position and label always come from the same row - no transpose risk.
cm_cells <- data.frame(
  Predicted = c("1 (Yes)", "1 (Yes)", "0 (No)", "0 (No)"),
  Actual    = c("0 (No)",  "1 (Yes)", "0 (No)", "1 (Yes)"),
  Count     = c(FP,        TP,        TN,       FN)
)
# x = Actual (columns: No, Yes), y = Predicted (rows, top to bottom: Yes, No)
cm_cells$x <- ifelse(cm_cells$Actual == "0 (No)", 1, 2)
cm_cells$y <- ifelse(cm_cells$Predicted == "1 (Yes)", 2, 1)  # Yes on top

png(file.path(plot_dir, "confusion_matrix.png"), width = 700, height = 650, res = 130)
par(mar = c(5, 6, 4, 2))
plot(NA, xlim = c(0.5, 2.5), ylim = c(0.5, 2.5), axes = FALSE,
     xlab = "Actual", ylab = "Predicted", main = "Confusion Matrix (Test Set)")
max_count <- max(cm_cells$Count)
for (i in 1:nrow(cm_cells)) {
  shade <- 0.15 + 0.65 * (cm_cells$Count[i] / max_count)
  rect(cm_cells$x[i] - 0.48, cm_cells$y[i] - 0.48, cm_cells$x[i] + 0.48, cm_cells$y[i] + 0.48,
       col = rgb(0.30, 0.45, 0.63, shade), border = "white", lwd = 2)
  text(cm_cells$x[i], cm_cells$y[i], labels = cm_cells$Count[i], cex = 2, font = 2)
}
axis(1, at = c(1, 2), labels = c("0 (No)", "1 (Yes)"))
axis(2, at = c(1, 2), labels = c("0 (No)", "1 (Yes)"))
box()
dev.off()

# ---- Plot 3: Residual diagnostics (deviance residuals vs fitted) ----
png(file.path(plot_dir, "residual_plot.png"), width = 800, height = 650, res = 130)
plot(fitted(model), residuals(model, type = "deviance"),
     xlab = "Fitted values (predicted probability)", ylab = "Deviance residuals",
     main = "Deviance Residuals vs Fitted Values", pch = 20, col = rgb(0.3, 0.4, 0.6, 0.5))
abline(h = 0, col = "red", lty = 2)
lines(lowess(fitted(model), residuals(model, type = "deviance")), col = "#C44E52", lwd = 2)
dev.off()

# ---- Plot 4: Predicted probability distribution by actual outcome ----
png(file.path(plot_dir, "predicted_prob_distribution.png"), width = 800, height = 650, res = 130)
boxplot(test_probs ~ test$Survived, col = c("#C44E52", "#4C8C4C"),
        names = c("Did not survive", "Survived"),
        main = "Predicted Survival Probability by Actual Outcome",
        ylab = "Predicted probability of survival", xlab = "Actual outcome")
dev.off()

# ---- Plot 5: Coefficient / odds ratio plot ----
coefs <- summary(model)$coefficients
or_df <- data.frame(
  term = rownames(coefs)[-1],
  estimate = coefs[-1, "Estimate"],
  se = coefs[-1, "Std. Error"]
)
or_df$or <- exp(or_df$estimate)
or_df$lower <- exp(or_df$estimate - 1.96 * or_df$se)
or_df$upper <- exp(or_df$estimate + 1.96 * or_df$se)

png(file.path(plot_dir, "odds_ratios.png"), width = 850, height = 650, res = 130)
par(mar = c(5, 12, 4, 2))
plot_order <- order(or_df$or)
or_df <- or_df[plot_order, ]
plot(or_df$or, 1:nrow(or_df), xlim = c(0, max(or_df$upper) * 1.1), yaxt = "n",
     xlab = "Odds Ratio (log scale reference line at 1)", ylab = "",
     main = "Predictor Odds Ratios (95% CI)", pch = 19, col = "#4C72B0", cex = 1.3)
segments(or_df$lower, 1:nrow(or_df), or_df$upper, 1:nrow(or_df), col = "#4C72B0", lwd = 2)
abline(v = 1, col = "red", lty = 2)
axis(2, at = 1:nrow(or_df), labels = or_df$term, las = 2, cex.axis = 0.8)
dev.off()

cat("\nAll diagnostic plots saved to:", plot_dir, "\n")

# Print odds ratios table
cat("\n===== Odds ratios table =====\n")
or_print <- or_df[, c("term", "or", "lower", "upper")]
or_print$or <- round(or_print$or, 3)
or_print$lower <- round(or_print$lower, 3)
or_print$upper <- round(or_print$upper, 3)
print(or_print)

cat("\n===== SCRIPT COMPLETE =====\n")
