# Packages
library(readxl)
library(dplyr)
library(tidyverse)
library(vegan)
library(patchwork)
library(pls)
library(ggplot2)
install.packages("plsVarSel")
library(plsVarSel)

# ─────────────────────────────────────────────────────────────
# Load and clean data
# ─────────────────────────────────────────────────────────────
df <- read_excel("D:\\PhD Uni South Bohemia\\Dissertation\\Europe Invasion Database\\EU predictors_Snapshot.xlsx")
names(df) <- make.names(names(df))  # clean column names

df_clean <- na.omit(df)

# Quick checks
colSums(is.na(df_clean))
sapply(df, function(x) sum(x == "", na.rm = TRUE))
sapply(df, function(x) sum(x == "NA", na.rm = TRUE))
str(df_clean)

# Ensure predictors are numeric (exclude identifiers/groups)
df_clean <- df_clean %>%
  mutate(across(-c(Country, Group_region, Group_culture),
                ~ suppressWarnings(as.numeric(.))))

# ─────────────────────────────────────────────────────────────
# Create predictor blocks (scaled)
# ─────────────────────────────────────────────────────────────
eco <- df_clean %>%
  select(protected_land_cover_., Agricultural_land, Surface_area,
         Air_quality_index_AQI, Ecological_footprint_hectat_pp, Nat_Biodiv,
         Pop_dens, urban_pop_perc, n_border_countries) %>%
  mutate(across(everything(), as.numeric)) %>%
  scale()

econ <- df_clean %>%
  select(GDP2021_b, GNI_per_capita_in_thousands,
         Research_expenditure_perc_GDP, Foreign_Direct_Investment_FDI_2021,
         Government_Debt_to_GDP_Ratio_., Innovation_Index_Ranking_score,
         unemployment_rate_._2021, n_airports, n_airports_intl,
         ports_traffic, imports_CIF_in_million, tourists_m, Human_development_Index_HDI,
         Country_Growth, Sust_development) %>%
  mutate(across(everything(), as.numeric)) %>%
  scale()

cult <- df_clean %>%
  select(Trompenaar_1, Trompenaar_7, tau,
         Literacy_Rate_., Migration_Rate_.migrants.1000population., WVS_prot_envi_vs_econ, WVS_envi_prot_mov) %>%
  mutate(across(everything(), as.numeric)) %>%
  scale()

# Predictor matrix
X <- cbind(eco, econ, cult)

# Response matrix (scaled)
Y_multi <- df_clean %>%
  select(Elzas_DB_._terrestrial_species,
         Elzas_DB_._aquatic_species,
         Elzas_DB_._semiaquatic_species) %>%
  scale()
Y <- as.matrix(Y_multi)

# Optional: correlation of responses
cor(Y)

# Define block names once (used by multiple VIP summaries)
eco_vars  <- colnames(eco)
econ_vars <- colnames(econ)
cult_vars <- colnames(cult)
blocks <- list(Ecology = eco_vars, Economics = econ_vars, Culture = cult_vars)

# ─────────────────────────────────────────────────────────────
# Train/Test split
# ─────────────────────────────────────────────────────────────
set.seed(123)
n <- nrow(df_clean)
test_idx  <- sample(1:n, size = round(0.3 * n))   # 30% test set
train_idx <- setdiff(1:n, test_idx)

X_train <- X[train_idx, ]
Y_train <- Y[train_idx, ]
X_test  <- X[test_idx, ]
Y_test  <- Y[test_idx, ]

# ─────────────────────────────────────────────────────────────
# Fit PLS2 with CV on training set, select components
# ─────────────────────────────────────────────────────────────
pls2_fit <- plsr(Y_train ~ ., data = as.data.frame(X_train),
                 ncomp = 10, validation = "CV", scale = FALSE)

plot(RMSEP(pls2_fit), legendpos = "topright")
opt_comp <- which.min(RMSEP(pls2_fit)$val["CV", , ])

# Refit with optimal components (training model)
pls2_opt <- plsr(Y_train ~ ., data = as.data.frame(X_train),
                 ncomp = opt_comp, scale = FALSE)

summary(pls2_opt)

# ─────────────────────────────────────────────────────────────
# Predict on TEST set and unbiased metrics
# ─────────────────────────────────────────────────────────────
Y_pred_test <- predict(pls2_opt, newdata = as.data.frame(X_test), ncomp = opt_comp)[,,1]

r2 <- function(obs, pred) cor(obs, pred)^2
rmse <- function(obs, pred) sqrt(mean((obs - pred)^2))

r2_terrestrial <- r2(Y_test[,1], Y_pred_test[,1])
r2_aquatic     <- r2(Y_test[,2], Y_pred_test[,2])
r2_semiaquatic <- r2(Y_test[,3], Y_pred_test[,3])

rmse_terrestrial <- rmse(Y_test[,1], Y_pred_test[,1])
rmse_aquatic     <- rmse(Y_test[,2], Y_pred_test[,2])
rmse_semiaquatic <- rmse(Y_test[,3], Y_pred_test[,3])

r2_terrestrial; r2_aquatic; r2_semiaquatic
rmse_terrestrial; rmse_aquatic; rmse_semiaquatic


# Predict on TRAIN set
Y_pred_train <- predict(pls2_opt, newdata = as.data.frame(X_train), ncomp = opt_comp)[,,1]

# Define helper functions
r2 <- function(obs, pred) cor(obs, pred)^2
rmse <- function(obs, pred) sqrt(mean((obs - pred)^2))

# --- Training set metrics ---
r2_train_terrestrial <- r2(Y_train[,1], Y_pred_train[,1])
r2_train_aquatic     <- r2(Y_train[,2], Y_pred_train[,2])
r2_train_semiaquatic <- r2(Y_train[,3], Y_pred_train[,3])

rmse_train_terrestrial <- rmse(Y_train[,1], Y_pred_train[,1])
rmse_train_aquatic     <- rmse(Y_train[,2], Y_pred_train[,2])
rmse_train_semiaquatic <- rmse(Y_train[,3], Y_pred_train[,3])

# --- Test set metrics ---
r2_test_terrestrial <- r2(Y_test[,1], Y_pred_test[,1])
r2_test_aquatic     <- r2(Y_test[,2], Y_pred_test[,2])
r2_test_semiaquatic <- r2(Y_test[,3], Y_pred_test[,3])

rmse_test_terrestrial <- rmse(Y_test[,1], Y_pred_test[,1])
rmse_test_aquatic     <- rmse(Y_test[,2], Y_pred_test[,2])
rmse_test_semiaquatic <- rmse(Y_test[,3], Y_pred_test[,3])


# ─────────────────────────────────────────────────────────────
# VIP on training model (evaluation perspective)
# ─────────────────────────────────────────────────────────────
vip_scores <- VIP(pls2_opt, opt.comp = opt_comp)
vip_df <- data.frame(Predictor = names(vip_scores), VIP = vip_scores)
vip_df_rounded <- vip_df %>% mutate(VIP = round(VIP, 3))

# Global VIP by block (training model)
block_summary <- sapply(blocks, function(vars) {
  block_scores <- vip_scores[vars]
  c(
    mean_VIP    = mean(block_scores, na.rm = TRUE),
    median_VIP  = median(block_scores, na.rm = TRUE),
    prop_above1 = mean(block_scores > 1, na.rm = TRUE)
  )
})
block_summary <- t(block_summary)
rownames(block_summary) <- names(blocks)
print(round(block_summary, 3))

# Univariate PLS per habitat (training model)
pls_terrestrial <- plsr(Y_train[,1] ~ ., data = as.data.frame(X_train), opt.comp = opt_comp, scale = FALSE)
pls_aquatic     <- plsr(Y_train[,2] ~ ., data = as.data.frame(X_train), opt.comp = opt_comp, scale = FALSE)
pls_semiaquatic <- plsr(Y_train[,3] ~ ., data = as.data.frame(X_train), opt.comp = opt_comp, scale = FALSE)

vip_terrestrial <- VIP(pls_terrestrial, opt.comp = opt_comp)
vip_aquatic     <- VIP(pls_aquatic,     opt.comp = opt_comp)
vip_semiaquatic <- VIP(pls_semiaquatic, opt.comp = opt_comp)

vip_df_habitats <- data.frame(
  Predictor    = names(vip_terrestrial),
  Terrestrial  = vip_terrestrial,
  Aquatic      = vip_aquatic,
  Semi_aquatic = vip_semiaquatic
) %>%
  mutate(across(-Predictor, ~ round(., 3)))

print(vip_df_habitats)

# Plots for VIP > 1 (training model, by habitat)
vip_terrestrial_df <- vip_df_habitats %>% filter(Terrestrial  > 1) %>% arrange(desc(Terrestrial))
vip_aquatic_df     <- vip_df_habitats %>% filter(Aquatic      > 1) %>% arrange(desc(Aquatic))
vip_semiaquatic_df <- vip_df_habitats %>% filter(Semi_aquatic > 1) %>% arrange(desc(Semi_aquatic))

ggplot(vip_terrestrial_df, aes(x = reorder(Predictor, Terrestrial), y = Terrestrial)) +
  geom_col(fill = "#1b9e77") + coord_flip() +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  labs(title = "VIP Scores – Terrestrial Species (training)", x = "Predictor", y = "VIP Score") +
  theme_minimal()

ggplot(vip_aquatic_df, aes(x = reorder(Predictor, Aquatic), y = Aquatic)) +
  geom_col(fill = "#7570b3") + coord_flip() +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  labs(title = "VIP Scores – Aquatic Species (training)", x = "Predictor", y = "VIP Score") +
  theme_minimal()

ggplot(vip_semiaquatic_df, aes(x = reorder(Predictor, Semi_aquatic), y = Semi_aquatic)) +
  geom_col(fill = "#d95f02") + coord_flip() +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  labs(title = "VIP Scores – Semi-aquatic Species (training)", x = "Predictor", y = "VIP Score") +
  theme_minimal()

# Summarize VIP by block and habitat (training model)
block_vip_summary <- sapply(blocks, function(vars) {
  block_scores <- vip_df_habitats[vip_df_habitats$Predictor %in% vars, ]
  c(
    mean_terrestrial = mean(block_scores$Terrestrial,  na.rm = TRUE),
    mean_aquatic     = mean(block_scores$Aquatic,      na.rm = TRUE),
    mean_semiaquatic = mean(block_scores$Semi_aquatic, na.rm = TRUE)
  )
})
block_vip_summary <- t(block_vip_summary)
print(round(block_vip_summary, 3))

# Save global VIP plot (training model)
vip_plot <- ggplot(vip_df, aes(x = reorder(Predictor, VIP), y = VIP)) +
  geom_col(fill = "darkgreen") + coord_flip() +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  labs(x = "Predictor", y = "VIP score", title = "Variable Importance (VIP) – training model") +
  theme_minimal()
print(vip_plot)
ggsave("vip_plot_training.svg", plot = vip_plot, width = 8, height = 6, units = "in")

# ─────────────────────────────────────────────────────────────
# Observed vs Predicted: TEST set (evaluation)
# ─────────────────────────────────────────────────────────────
svg("terre_obs_vs_pred.svg", width = 8, height = 6)
plot(Y_test[,1], Y_pred_test[,1], xlab = "Observed terrestrial (scaled)", ylab = "Predicted terrestrial (scaled)")
abline(0, 1, col = "red"); dev.off()

svg("aquat_obs_vs_pred.svg", width = 8, height = 6)
plot(Y_test[,2], Y_pred_test[,2], xlab = "Observed aquatic (scaled)", ylab = "Predicted aquatic (scaled)")
abline(0, 1, col = "red"); dev.off()

svg("semiaqua_obs_vs_pred.svg", width = 8, height = 6)
plot(Y_test[,3], Y_pred_test[,3], xlab = "Observed semi-aquatic (scaled)", ylab = "Predicted semi-aquatic (scaled)")
abline(0, 1, col = "red"); dev.off()
##--------------------
#
##--------------------
# Open SVG device
svg("obs_vs_pred_grid.svg", width = 10, height = 12)

# Set up a 3x2 grid layout
par(mfrow = c(3, 2), mar = c(4, 4, 2, 1))

# Terrestrial
plot(Y_train[,1], Y_pred_train[,1],
     xlab = "Observed terrestrial (scaled)", ylab = "Predicted terrestrial (scaled)",
     main = "Training - Terrestrial")
abline(0, 1, col = "red")

plot(Y_test[,1], Y_pred_test[,1],
     xlab = "Observed terrestrial (scaled)", ylab = "Predicted terrestrial (scaled)",
     main = "Test - Terrestrial")
abline(0, 1, col = "red")

# Aquatic
plot(Y_train[,2], Y_pred_train[,2],
     xlab = "Observed aquatic (scaled)", ylab = "Predicted aquatic (scaled)",
     main = "Training - Aquatic")
abline(0, 1, col = "red")

plot(Y_test[,2], Y_pred_test[,2],
     xlab = "Observed aquatic (scaled)", ylab = "Predicted aquatic (scaled)",
     main = "Test - Aquatic")
abline(0, 1, col = "red")

# Semi-aquatic
plot(Y_train[,3], Y_pred_train[,3],
     xlab = "Observed semi-aquatic (scaled)", ylab = "Predicted semi-aquatic (scaled)",
     main = "Training - Semi-aquatic")
abline(0, 1, col = "red")

plot(Y_test[,3], Y_pred_test[,3],
     xlab = "Observed semi-aquatic (scaled)", ylab = "Predicted semi-aquatic (scaled)",
     main = "Test - Semi-aquatic")
abline(0, 1, col = "red")

# Close SVG device
dev.off()
# ─────────────────────────────────────────────────────────────
# Residuals and country-level bias (TEST set)
# ─────────────────────────────────────────────────────────────
residuals_mat <- Y_test - Y_pred_test
country_names_test <- df_clean$Country[test_idx]

residuals_df <- data.frame(
  Country           = country_names_test,
  Terrestrial_resid = residuals_mat[,1],
  Aquatic_resid     = residuals_mat[,2],
  Semi_resid        = residuals_mat[,3],
  Mean_bias         = rowMeans(residuals_mat)
)

residuals_df$BiasType <- ifelse(residuals_df$Mean_bias > 0, "Over-reporting", "Under-reporting")

country_bias <- ggplot(residuals_df, aes(x = reorder(Country, Mean_bias), y = Mean_bias, fill = BiasType)) +
  geom_col() + coord_flip() +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(x = "Country", y = "Mean residual (Observed - Predicted)",
       title = "Country-level reporting bias (TEST set)") +
  scale_fill_manual(values = c("Over-reporting" = "firebrick", "Under-reporting" = "steelblue")) +
  theme_minimal()
ggsave("country_bias.svg", plot = country_bias, width = 8, height = 6, units = "in")

# Cultural/regional bias (TEST set)
residuals_grouped <- residuals_df %>%
  left_join(df_clean %>% select(Country, Group_culture, Group_region), by = "Country")

cultural_summary <- residuals_grouped %>%
  group_by(Group_culture) %>%
  summarise(Mean_bias = mean(Mean_bias, na.rm = TRUE), .groups = "drop") %>%
  mutate(BiasType = ifelse(Mean_bias > 0, "Over-reporting", "Under-reporting"))

regional_summary <- residuals_grouped %>%
  group_by(Group_region) %>%
  summarise(Mean_bias = mean(Mean_bias, na.rm = TRUE), .groups = "drop") %>%
  mutate(BiasType = ifelse(Mean_bias > 0, "Over-reporting", "Under-reporting"))

cultgrp_bias <- ggplot(cultural_summary, aes(x = reorder(Group_culture, Mean_bias), y = Mean_bias, fill = BiasType)) +
  geom_col() + coord_flip() +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(x = "Cultural group", y = "Mean residual", title = "Reporting bias by cultural group (TEST set)") +
  scale_fill_manual(values = c("Over-reporting" = "firebrick", "Under-reporting" = "steelblue")) +
  theme_minimal()
ggsave("cultgrp_bias.svg", plot = cultgrp_bias, width = 8, height = 6, units = "in")

regiogrp_bias <- ggplot(regional_summary, aes(x = reorder(Group_region, Mean_bias), y = Mean_bias, fill = BiasType)) +
  geom_col() + coord_flip() +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(x = "Region", y = "Mean residual", title = "Reporting bias by region (TEST set)") +
  scale_fill_manual(values = c("Over-reporting" = "firebrick", "Under-reporting" = "steelblue")) +
  theme_minimal()
ggsave("regiogrp_bias.svg", plot = regiogrp_bias, width = 8, height = 6, units = "in")

# ─────────────────────────────────────────────────────────────
# Refit on ALL countries and predict EU-wide
# ─────────────────────────────────────────────────────────────
# Refit on ALL countries
pls2_full <- plsr(Y ~ ., data = as.data.frame(X), ncomp = opt_comp, scale = FALSE)

# VIP on full model
vip_scores_full <- VIP(pls2_full, opt.comp = opt_comp)
vip_full_df <- data.frame(Predictor = names(vip_scores_full), VIP = round(vip_scores_full, 3))

vip_plot_full <- ggplot(vip_full_df, aes(x = reorder(Predictor, VIP), y = VIP)) +
  geom_col(fill = "darkgreen") + coord_flip() +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  labs(x = "Predictor", y = "VIP score", title = "Variable Importance (VIP) – full model") +
  theme_minimal()

print(vip_plot_full)

ggsave("vip_plot_full.svg", plot = vip_plot_full, width = 8, height = 6, units = "in")

# Predict for ALL EU countries
Y_pred_all <- predict(pls2_full, newdata = as.data.frame(X), ncomp = opt_comp)[,,1]

# Predictions table
predictions_df <- data.frame(
  Country      = df_clean$Country,
  Terrestrial  = Y_pred_all[,1],
  Aquatic      = Y_pred_all[,2],
  Semi_aquatic = Y_pred_all[,3]
)

# Residuals and bias
residuals_mat_full <- Y - Y_pred_all
residuals_df_full <- data.frame(
  Country           = df_clean$Country,
  Terrestrial_resid = residuals_mat_full[,1],
  Aquatic_resid     = residuals_mat_full[,2],
  Semi_resid        = residuals_mat_full[,3],
  Mean_bias         = rowMeans(residuals_mat_full)
)
residuals_df_full$BiasType <- ifelse(residuals_df_full$Mean_bias > 0,
                                     "Over-reporting", "Under-reporting")

# Country bias plot
country_bias_full <- ggplot(residuals_df_full, aes(x = reorder(Country, Mean_bias),
                                                   y = Mean_bias, fill = BiasType)) +
  geom_col() + coord_flip() +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(x = "Country", y = "Mean residual (Observed - Predicted)",
       title = "Country-level reporting bias (ALL EU countries)") +
  scale_fill_manual(values = c("Over-reporting" = "firebrick",
                               "Under-reporting" = "steelblue")) +
  theme_minimal()
ggsave("country_bias_full.svg", plot = country_bias_full, width = 10, height = 6, units = "in")

# Cultural/regional bias summaries
residuals_grouped_full <- residuals_df_full %>%
  left_join(df_clean %>% select(Country, Group_culture, Group_region), by = "Country")

cultural_summary_full <- residuals_grouped_full %>%
  group_by(Group_culture) %>%
  summarise(Mean_bias = mean(Mean_bias, na.rm = TRUE), .groups = "drop") %>%
  mutate(BiasType = ifelse(Mean_bias > 0, "Over-reporting", "Under-reporting"))

regional_summary_full <- residuals_grouped_full %>%
  group_by(Group_region) %>%
  summarise(Mean_bias = mean(Mean_bias, na.rm = TRUE), .groups = "drop") %>%
  mutate(BiasType = ifelse(Mean_bias > 0, "Over-reporting", "Under-reporting"))

print(cultural_summary_full)
print(regional_summary_full)

# Combine predictions and residuals into one master table
results_df <- predictions_df %>%
  left_join(residuals_df_full %>% select(Country, Terrestrial_resid, Aquatic_resid, Semi_resid, Mean_bias, BiasType),
            by = "Country")

head(results_df)

# --- Cultural bias barplot ---
cultural_bias_plot <- ggplot(cultural_summary_full,
                             aes(x = reorder(Group_culture, Mean_bias),
                                 y = Mean_bias, fill = BiasType)) +
  geom_col() +
  coord_flip() +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(x = "Cultural grouping", y = "Mean residual (Observed - Predicted)",
       title = "Cultural group reporting bias") +
  scale_fill_manual(values = c("Over-reporting" = "firebrick",
                               "Under-reporting" = "steelblue")) +
  theme_minimal()

ggsave("cultural_bias_plot.svg", plot = cultural_bias_plot,
       width = 8, height = 6, units = "in")

# --- Regional bias barplot ---
regional_bias_plot <- ggplot(regional_summary_full,
                             aes(x = reorder(Group_region, Mean_bias),
                                 y = Mean_bias, fill = BiasType)) +
  geom_col() +
  coord_flip() +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(x = "Regional grouping", y = "Mean residual (Observed - Predicted)",
       title = "Regional group reporting bias") +
  scale_fill_manual(values = c("Over-reporting" = "firebrick",
                               "Under-reporting" = "steelblue")) +
  theme_minimal()

ggsave("regional_bias_plot.svg", plot = regional_bias_plot,
       width = 8, height = 6, units = "in")

##---------------------------------------
#Visualizations
##------------------------------------

library(giscoR)
library(sf)
library(dplyr)
library(ggplot2)
library(countrycode)

# Load GISCO shapefile
world_map <- gisco_get_countries(resolution = "10", epsg = "4326")

# Define Europe ISO3 codes
europe_iso3 <- c("FRA","DEU","ESP","ITA","POL","UKR","SWE","NOR","FIN","CHE","AUT",
                 "BEL","NLD","PRT","GRC","CZE","SVK","HUN","ROU","BGR","SRB","HRV",
                 "SVN","BIH","MNE","MKD","ALB","EST","LVA","LTU","IRL","GBR",
                 "BLR","CYP","DNK","ISL")

# Filter shapefile
europe_map <- world_map %>% filter(ISO3_CODE %in% europe_iso3)

# Add ISO3 codes to results_df
results_df$ISO3_CODE <- countrycode(results_df$Country, "country.name", "iso3c")

# Manual fixes if needed (e.g. Germany/Deutschland)
results_df$ISO3_CODE[results_df$Country %in% c("Germany","Deutschland")] <- "DEU"

# Merge shapefile with results_df
map_data <- europe_map %>% left_join(results_df, by = "ISO3_CODE")

# Plot mean bias
bias_map <- ggplot(map_data) +
  geom_sf(aes(fill = Mean_bias), color = "grey40", size = 0.2) +
  scale_fill_gradient2(low = "steelblue", mid = "white", high = "firebrick",
                       midpoint = 0, name = "Mean residual") +
  labs(title = "Country-level reporting bias in Europe",
       subtitle = "Observed – Predicted (positive = over-reporting, negative = under-reporting)") +
  theme_minimal()
ggsave("europe_bias_map.svg", plot = bias_map, width = 10, height = 8, units = "in")

#Terrestrial Bias
Terr_bias <- ggplot(map_data) +
  geom_sf(aes(fill = Terrestrial_resid), color = "grey40", size = 0.2) +
  scale_fill_gradient2(low = "steelblue", mid = "white", high = "firebrick",
                       midpoint = 0, name = "Terrestrial residual") +
  labs(title = "Terrestrial reporting bias in Europe") +
  theme_minimal()
print(Terr_bias)

#Aquatic Bias
aquatic_bias_map <- ggplot(map_data) +
  geom_sf(aes(fill = Aquatic_resid), color = "grey40", size = 0.2) +
  scale_fill_gradient2(low = "steelblue", mid = "white", high = "firebrick",
                       midpoint = 0, name = "Aquatic residual") +
  labs(title = "Aquatic reporting bias in Europe") +
  theme_minimal()
print(aquatic_bias_map)

#semiaquatic bias

# Semi-aquatic reporting bias map
semi_bias_map <- ggplot(map_data) +
  geom_sf(aes(fill = Semi_resid), color = "grey40", size = 0.2) +
  scale_fill_gradient2(
    low = "steelblue", mid = "white", high = "firebrick",
    midpoint = 0, name = "Semi-aquatic residual"
  ) +
  labs(title = "Semi-aquatic reporting bias in Europe") +
       theme_minimal()

# Display
print(semi_bias_map)


# Plot predicted terrestrial richness
terrestrial_map <- ggplot(map_data) +
  geom_sf(aes(fill = Terrestrial), color = "grey40", size = 0.2) +
  scale_fill_viridis_c(name = "Predicted terrestrial") +
  labs(title = "Predicted terrestrial species richness in Europe") +
  theme_minimal()
ggsave("europe_terrestrial_map.svg", plot = terrestrial_map, width = 10, height = 8, units = "in")

# Predicted aquatic richness
aquatic_map <- ggplot(map_data) +
  geom_sf(aes(fill = Aquatic), color = "grey40", size = 0.2) +
  scale_fill_viridis_c(name = "Predicted aquatic") +
  labs(title = "Predicted aquatic species richness in Europe") +
  theme_minimal()

print(aquatic_map)

# Predicted semi-aquatic richness
semi_map <- ggplot(map_data) +
  geom_sf(aes(fill = Semi_aquatic), color = "grey40", size = 0.2) +
  scale_fill_viridis_c(name = "Predicted semi-aquatic") +
  labs(title = "Predicted semi-aquatic species richness in Europe") +
  theme_minimal()

print(semi_map)

###------------------------------------------------
#VIP visuals for full model
##-------------------------------------------------

vip_full_df <- data.frame(Predictor = names(vip_scores_full), VIP = vip_scores_full)

vip_plot_full <- ggplot(vip_full_df, aes(x = reorder(Predictor, VIP), y = VIP)) +
  geom_col(fill = "darkblue") +
  coord_flip() +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  labs(x = "Predictor", y = "VIP score", title = "Global VIP – Full model") +
  theme_minimal()

print(vip_plot_full)

ggsave("vip_full.svg", plot = vip_plot_full, width = 8, height = 6, units = "in")


# Terrestrial-only model (full dataset)
pls_terrestrial <- plsr(Y[,1] ~ ., data = as.data.frame(X), opt.comp = opt_comp, scale = FALSE)

# Compute VIP scores
vip_terrestrial_full <- VIP(pls_terrestrial,opt.comp = opt_comp)

# Convert to data frame
vip_terrestrial_full_df <- data.frame(
  Predictor = names(vip_terrestrial_full),
  VIP = round(vip_terrestrial_full, 3)
)

# Plot
vip_plot_terrestrial_full <- ggplot(vip_terrestrial_full_df,
                                    aes(x = reorder(Predictor, VIP), y = VIP)) +
  geom_col(fill = "#1b9e77") +
  coord_flip() +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  labs(x = "Predictor", y = "VIP score",
       title = "VIP – Terrestrial species (Full model)") +
  theme_minimal()

print(vip_plot_terrestrial_full)

ggsave("terre_vip_full.svg", plot = vip_plot_terrestrial_full, width = 8, height = 6, units = "in")

# Aquatic-only model (full dataset)
pls_aquatic <- plsr(Y[,2] ~ ., data = as.data.frame(X), opt.comp = opt_comp, scale = FALSE)

# Compute VIP scores
vip_aquatic_full <- VIP(pls_aquatic, opt.comp = opt_comp)

# Convert to data frame
vip_aquatic_full_df <- data.frame(
  Predictor = names(vip_aquatic_full),
  VIP = round(vip_aquatic_full, 3)
)

# Plot
vip_plot_aquatic_full <- ggplot(vip_aquatic_full_df,
                                aes(x = reorder(Predictor, VIP), y = VIP)) +
  geom_col(fill = "#7570b3") +
  coord_flip() +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  labs(x = "Predictor", y = "VIP score",
       title = "VIP – Aquatic species (Full model)") +
  theme_minimal()

print(vip_plot_aquatic_full)

ggsave("aqua_vip_full.svg", plot = vip_plot_aquatic_full, width = 8, height = 6, units = "in")

# Semi-aquatic-only model (full dataset)
pls_semiaquatic <- plsr(Y[,3] ~ ., data = as.data.frame(X), opt.comp = opt_comp, scale = FALSE)

# Compute VIP scores
vip_semiaquatic_full <- VIP(pls_semiaquatic, opt.comp = opt_comp)

# Convert to data frame
vip_semiaquatic_full_df <- data.frame(
  Predictor = names(vip_semiaquatic_full),
  VIP = round(vip_semiaquatic_full, 3)
)

# Plot
vip_plot_semiaquatic_full <- ggplot(vip_semiaquatic_full_df,
                                    aes(x = reorder(Predictor, VIP), y = VIP)) +
  geom_col(fill = "#d95f02") +
  coord_flip() +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  labs(x = "Predictor", y = "VIP score",
       title = "VIP – Semi-aquatic species (Full model)") +
  theme_minimal()

print(vip_plot_semiaquatic_full)

ggsave("semi_vip_full.svg", plot = vip_plot_semiaquatic_full, width = 8, height = 6, units = "in")




# Long format residuals
resid_long <- map_data %>%
  st_drop_geometry() %>%   # drop spatial geometry
  select(Country, Terrestrial_resid, Aquatic_resid, Semi_resid) %>%
  tidyr::pivot_longer(cols = ends_with("resid"),
                      names_to = "Habitat", values_to = "Residual")

# Raincloud plot
rain_fig <- ggplot(resid_long, aes(x = Habitat, y = Residual, fill = Habitat)) +
  ggdist::stat_halfeye(adjust = .5, width = .6, .width = 0, justification = -.3,
                       point_colour = NA) +
  geom_boxplot(width = .15, outlier.shape = NA, alpha = 0.3) +
  geom_jitter(width = .05, alpha = 0.5) +
  coord_flip() +
  labs(title = "Raincloud plot of reporting bias by habitat",
       y = "Residual (Observed – Predicted)",
       x = "Habitat") +
  theme_minimal()

print(rain_fig)
ggsave("raincloud_reporting.svg", plot = rain_fig,width = 10, height = 8, units = "in" )

#########################################################################################

summary(resid_long$Residual)
# Check which rows are NA
which(is.na(resid_long$Residual))

# See the countries with missing residuals
resid_long %>% filter(is.na(Residual)) %>% select(Country, Habitat)

map_data %>% 
  filter(is.na(Terrestrial_resid) | is.na(Aquatic_resid) | is.na(Semi_resid)) %>% 
  select(Country)

# Plot the polygons to see where they are
plot(map_data %>% filter(is.na(Terrestrial_resid) | is.na(Aquatic_resid) | is.na(Semi_resid)))


# Compare counts
nrow(Y)        # number of response rows
nrow(pls2_full$fitted.values[[opt_comp]])  # number of fitted values



########################################################################################33

permute_block <- function(block_vars, X_train, Y_train, X_test, Y_test, opt_comp, nrep = 100) {
  drops <- matrix(NA, nrow = nrep, ncol = ncol(Y_test))
  colnames(drops) <- colnames(Y_test)
  
  for (i in 1:nrep) {
    X_train_perm <- X_train
    X_test_perm  <- X_test
    
    for (var in block_vars) {
      X_train_perm[, var] <- sample(X_train_perm[, var])
      X_test_perm[, var]  <- sample(X_test_perm[, var])
    }
    
    pls_perm <- plsr(Y_train ~ ., data = as.data.frame(X_train_perm),
                     ncomp = opt_comp, scale = FALSE)
    
    Y_pred_perm <- predict(pls_perm, newdata = as.data.frame(X_test_perm), ncomp = opt_comp)[,,1]
    
    for (j in 1:ncol(Y_test)) {
      rmse_orig <- sqrt(mean((Y_test[,j] - Y_pred_test[,j])^2))
      rmse_perm <- sqrt(mean((Y_test[,j] - Y_pred_perm[,j])^2))
      drops[i,j] <- rmse_perm - rmse_orig
    }
  }
  return(drops)
}

eco_drops  <- permute_block(eco_vars,  X_train, Y_train, X_test, Y_test, opt_comp)
econ_drops <- permute_block(econ_vars, X_train, Y_train, X_test, Y_test, opt_comp)
cult_drops <- permute_block(cult_vars, X_train, Y_train, X_test, Y_test, opt_comp)

# Summarize by habitat
apply(eco_drops, 2, mean)
apply(econ_drops, 2, mean)
apply(cult_drops, 2, mean)


#####################################
#VIP using R package
#####################################

# Calculate VIP scores using 5 components
vip_scores <- VIP(pls2_opt, opt.comp = 5)

# Inspect results
print(vip_scores)

# Identify influential predictors (VIP > 1)
influential_predictors <- names(vip_scores[vip_scores > 1])
print(influential_predictors)



# Assuming you already fit these earlier:
pls_terr <- plsr(Elzas_DB_._terrestrial_species ~ ., data = your_data, ncomp = 5, validation = "CV")
pls_aqua <- plsr(Elzas_DB_._aquatic_species ~ ., data = your_data, ncomp = 5, validation = "CV")
pls_semi <- plsr(Elzas_DB_._semiaquatic_species ~ ., data = your_data, ncomp = 5, validation = "CV")

# Now calculate VIP scores for each habitat
vip_terr <- VIP(pls_terr, opt.comp = 5)
vip_aqua <- VIP(pls_aqua, opt.comp = 5)
vip_semi <- VIP(pls_semi, opt.comp = 5)

# Inspect influential predictors for each habitat
names(vip_terr[vip_terr > 1])
names(vip_aqua[vip_aqua > 1])
names(vip_semi[vip_semi > 1])



print(vip_full_df)
