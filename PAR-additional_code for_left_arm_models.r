raw_df <- read.csv("D:/activities/dataset_full.csv")
df <- raw_df[, -1]

df$activity <- as.factor(df$activity)
df$activity <- make.names(df$activity)

LAindex <-
  grepl('LA', colnames(df)) #index of just the left arm variables
df_new <- cbind(df[, c(1:3)], df[, LAindex])
df <- df_new

trainIndex <- createDataPartition(df$activity,
                                  p = 0.7,
                                  list = FALSE,
                                  times = 1)

training_df <- df[trainIndex, ]
testing_df <- df[-trainIndex, ]

pca <- prcomp(training_df[, c(4:237)], center = TRUE, scale. = TRUE)
summary(pca)[["importance"]][, 1:17] #17 principal components explain 50.567% of the variability
training_pca <-
  predict(pca, training_df)[, 1:17] #taking first 17 principal components
training_df <-
  as.data.frame(cbind(training_df[, 1], training_pca)) #also removing person and segment
names(training_df)[1] <- "activity"
training_df[, 1] <- as.factor(training_df[, 1])
training_df[, 2:18] <- apply(training_df[, 2:18], 2, as.numeric)

testing_pca <-
  predict(pca, testing_df)[, 1:17] #taking first 17 principal components
testing_df <-
  as.data.frame(cbind(testing_df[, 1], testing_pca)) #also removing person and segment
names(testing_df)[1] <- "activity"
testing_df[, 1] <- as.factor(testing_df[, 1])
testing_df[, 2:18] <- apply(testing_df[, 2:18], 2, as.numeric)

resamplingINDEX <- createFolds(training_df$activity, k = 3)
resamplingCTRL <- trainControl(
  method = "cv",
  number = 3,
  verboseIter = TRUE,
  index = resamplingINDEX,
  allowParallel = TRUE
)

rm(raw_df,
   training_pca,
   testing_pca,
   LAindex,
   trainIndex,
   resamplingINDEX,
   pca)
save.image("D:/activities/models_code_SAVED_allACTIVITIES_PCA_leftarm_BASICS.RData")
