# =========================================================
# Week 2 Task: Data Visualization and Insight Communication with R
# Dataset: Titanic passenger data (cleaned in Week 1)
# =========================================================

library(ggplot2)

# ---- 0. Load cleaned data from Week 1 ----
df <- read.csv("/home/claude/week1/titanic_cleaned.csv", stringsAsFactors = FALSE)
df$Pclass_label <- factor(df$Pclass_label, levels = c("First", "Second", "Third"))
df$Survived_label <- factor(df$Survived, levels = c(0, 1), labels = c("Did not survive", "Survived"))
df$Sex <- factor(df$Sex, levels = c("male", "female"))

cat("===== Data loaded from Week 1 =====\n")
cat("Rows:", nrow(df), " Columns:", ncol(df), "\n\n")
str(df[, c("Survived_label", "Pclass_label", "Sex", "Age", "Fare", "SibSp", "Parch", "HasCabin")])

plot_dir <- "/home/claude/week2/plots"

# A consistent theme for all charts
theme_report <- theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 15, hjust = 0),
    plot.subtitle = element_text(color = "gray30", size = 11),
    panel.grid.minor = element_blank(),
    legend.position = "bottom"
  )

survival_colors <- c("Did not survive" = "#C44E52", "Survived" = "#4C8C4C")

# =========================================================
# Chart 1: Bar chart — Survival counts by Passenger Class
# =========================================================
cat("\n===== Chart 1: Survival counts by Passenger Class (bar chart) =====\n")

p1 <- ggplot(df, aes(x = Pclass_label, fill = Survived_label)) +
  geom_bar(position = "dodge", width = 0.7) +
  scale_fill_manual(values = survival_colors) +
  labs(
    title = "Survival Counts by Passenger Class",
    subtitle = "Third-class passengers were the largest group, but had the fewest survivors",
    x = "Passenger Class", y = "Number of Passengers", fill = NULL
  ) +
  theme_report

ggsave(file.path(plot_dir, "chart1_survival_by_class_bar.png"), p1, width = 7.5, height = 5.5, dpi = 150)
cat("Saved chart1_survival_by_class_bar.png\n")

class_counts <- table(df$Pclass_label, df$Survived_label)
print(class_counts)

# =========================================================
# Chart 2: Scatter plot — Age vs Fare, colored by survival
# =========================================================
cat("\n===== Chart 2: Age vs Fare scatter plot =====\n")

p2 <- ggplot(df, aes(x = Age, y = Fare, color = Survived_label)) +
  geom_point(alpha = 0.6, size = 2) +
  scale_color_manual(values = survival_colors) +
  scale_y_continuous(limits = c(0, 300)) +
  labs(
    title = "Age vs. Fare, Colored by Survival Outcome",
    subtitle = "Higher-fare passengers (likely higher class) show a higher survival concentration",
    x = "Age (years)", y = "Fare (capped at 300 for readability)", color = NULL
  ) +
  theme_report

ggsave(file.path(plot_dir, "chart2_age_vs_fare_scatter.png"), p2, width = 7.5, height = 5.5, dpi = 150)
cat("Saved chart2_age_vs_fare_scatter.png\n")

cat("Correlation between Age and Fare:", round(cor(df$Age, df$Fare), 3), "\n")

# =========================================================
# Chart 3: Histogram — Age distribution split by survival
# =========================================================
cat("\n===== Chart 3: Age distribution histogram, faceted by survival =====\n")

p3 <- ggplot(df, aes(x = Age, fill = Survived_label)) +
  geom_histogram(binwidth = 5, color = "white", alpha = 0.85) +
  scale_fill_manual(values = survival_colors) +
  facet_wrap(~Survived_label, ncol = 1) +
  labs(
    title = "Age Distribution by Survival Outcome",
    subtitle = "Young children show a visibly higher survival rate; the imputed-median spike at 28 appears in both groups",
    x = "Age (years)", y = "Count", fill = NULL
  ) +
  theme_report +
  theme(legend.position = "none")

ggsave(file.path(plot_dir, "chart3_age_histogram_by_survival.png"), p3, width = 7.5, height = 6, dpi = 150)
cat("Saved chart3_age_histogram_by_survival.png\n")

# Quick check: survival rate for children (<=12)
child_surv <- mean(df$Survived[df$Age <= 12])
cat("Survival rate for passengers aged 12 or under:", round(child_surv * 100, 1), "%\n")
cat("Survival rate for passengers over 12:", round(mean(df$Survived[df$Age > 12]) * 100, 1), "%\n")

# =========================================================
# Chart 4: Line chart — Survival rate trend across age bins
# =========================================================
cat("\n===== Chart 4: Survival rate trend by age group (line chart) =====\n")

df$AgeBin <- cut(df$Age, breaks = c(0, 5, 12, 18, 30, 45, 60, 80),
                  labels = c("0-5", "6-12", "13-18", "19-30", "31-45", "46-60", "61-80"),
                  include.lowest = TRUE)

age_surv_rate <- aggregate(Survived ~ AgeBin, data = df, FUN = mean)
age_surv_rate$SurvivalPct <- age_surv_rate$Survived * 100
print(age_surv_rate)

p4 <- ggplot(age_surv_rate, aes(x = AgeBin, y = SurvivalPct, group = 1)) +
  geom_line(color = "#4C72B0", linewidth = 1.2) +
  geom_point(color = "#4C72B0", size = 3) +
  geom_text(aes(label = paste0(round(SurvivalPct, 0), "%")), vjust = -1, size = 3.5) +
  scale_y_continuous(limits = c(0, 80)) +
  labs(
    title = "Survival Rate Trend Across Age Groups",
    subtitle = "Youngest passengers (0-5) had the highest survival rate; rates decline through middle age",
    x = "Age Group", y = "Survival Rate (%)"
  ) +
  theme_report

ggsave(file.path(plot_dir, "chart4_survival_rate_by_age_line.png"), p4, width = 7.5, height = 5.5, dpi = 150)
cat("Saved chart4_survival_rate_by_age_line.png\n")

# =========================================================
# Chart 5: Stacked bar — Survival rate (%) by Sex and Class
# =========================================================
cat("\n===== Chart 5: Survival rate (%) by Sex and Class (grouped bar) =====\n")

sex_class_surv <- aggregate(Survived ~ Sex + Pclass_label, data = df, FUN = mean)
sex_class_surv$SurvivalPct <- sex_class_surv$Survived * 100
print(sex_class_surv)

p5 <- ggplot(sex_class_surv, aes(x = Pclass_label, y = SurvivalPct, fill = Sex)) +
  geom_col(position = "dodge", width = 0.65) +
  geom_text(aes(label = paste0(round(SurvivalPct, 0), "%")),
            position = position_dodge(width = 0.65), vjust = -0.5, size = 3.5) +
  scale_fill_manual(values = c("male" = "#4C72B0", "female" = "#DD8452")) +
  scale_y_continuous(limits = c(0, 105)) +
  labs(
    title = "Survival Rate (%) by Sex and Passenger Class",
    subtitle = "The sex gap in survival holds within every class, but is most extreme in Third class",
    x = "Passenger Class", y = "Survival Rate (%)", fill = NULL
  ) +
  theme_report

ggsave(file.path(plot_dir, "chart5_survival_by_sex_class.png"), p5, width = 7.5, height = 5.5, dpi = 150)
cat("Saved chart5_survival_by_sex_class.png\n")

# =========================================================
# Chart 6: Boxplot — Fare distribution by Class
# =========================================================
cat("\n===== Chart 6: Fare distribution by Class (boxplot) =====\n")

p6 <- ggplot(df, aes(x = Pclass_label, y = Fare, fill = Pclass_label)) +
  geom_boxplot(alpha = 0.8, outlier.color = "#C44E52", outlier.alpha = 0.5) +
  scale_fill_manual(values = c("First" = "#55A868", "Second" = "#4C72B0", "Third" = "#DD8452")) +
  scale_y_continuous(limits = c(0, 300)) +
  labs(
    title = "Fare Distribution by Passenger Class",
    subtitle = "First class shows the widest spread and highest median fare, as expected",
    x = "Passenger Class", y = "Fare (capped at 300 for readability)"
  ) +
  theme_report +
  theme(legend.position = "none")

ggsave(file.path(plot_dir, "chart6_fare_boxplot_by_class.png"), p6, width = 7.5, height = 5.5, dpi = 150)
cat("Saved chart6_fare_boxplot_by_class.png\n")

fare_by_class <- aggregate(Fare ~ Pclass_label, data = df, FUN = median)
print(fare_by_class)

cat("\n===== ALL VISUALIZATIONS COMPLETE =====\n")
cat("Plots saved to:", plot_dir, "\n")
