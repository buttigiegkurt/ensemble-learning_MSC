rm(list = ls())
library(readxl)
library(caret)
library(caretEnsemble)
library(rpart)
library(rpart.plot)
library(randomForest)
library(RRF)
library(adabag)
library(caTools)
library(h2o)
library(xgboost)

#library(doParallel)
#cl <- makePSOCKcluster(4)
#registerDoParallel(cl)

#stopCluster(cl)

####BASICS####

##ALL ACTIVITIES

raw_df <- read.csv("D:/activities/dataset_full.csv")
df <- raw_df[, -1]

df$activity <- as.factor(df$activity)
df$activity <- make.names(df$activity)

trainIndex <- createDataPartition(df$activity,
                                  p = 0.7,
                                  list = FALSE,
                                  times = 1)

training_df <- df[trainIndex, ]
testing_df <- df[-trainIndex, ]

pca <- prcomp(training_df[, c(4:1173)], center = TRUE, scale. = TRUE)
summary(pca)[["importance"]][, 1:30] #30 principal components explain 50.08% of the variability
training_pca <-
  predict(pca, training_df)[, 1:30] #taking first 30 principal components
training_df <-
  as.data.frame(cbind(training_df[, 1], training_pca)) #also removing person and segment
names(training_df)[1] <- "activity"
training_df[, 1] <- as.factor(training_df[, 1])
training_df[, 2:31] <- apply(training_df[, 2:31], 2, as.numeric)

testing_pca <-
  predict(pca, testing_df)[, 1:30] #taking first 30 principal components
testing_df <-
  as.data.frame(cbind(testing_df[, 1], testing_pca)) #also removing person and segment
names(testing_df)[1] <- "activity"
testing_df[, 1] <- as.factor(testing_df[, 1])
testing_df[, 2:31] <- apply(testing_df[, 2:31], 2, as.numeric)

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
   trainIndex,
   resamplingINDEX,
   pca)
save.image("D:/activities/models_code_SAVED_allACTIVITIES_PCA_BASICS.RData")

####BASICS TO LOAD####

load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_BASICS.RData")

####A - CART####

rm(list = ls())
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_BASICS.RData")

cartGRID <- expand.grid(
  cp = seq(0, 0.1, 0.005),
  #complexity parameter
  maxdepth = seq(3, 18, 3),
  minbucket = c(3, 6, 9, 12)
)

results <- NULL

for (i in 1:nrow(cartGRID)) {
  mdValue <- expand.grid(maxdepth = cartGRID[i, 2])
  tempcartMODEL <- train(
    activity ~ .,
    data = training_df,
    method = "rpart2",
    metric = "Kappa",
    trControl = resamplingCTRL,
    tuneGrid = mdValue,
    control = rpart.control(
      cp = cartGRID[i, 1],
      maxdepth = cartGRID[i, 2],
      minbucket = cartGRID[i, 3],
      xval = 0
    )
  )
  results <-
    rbind(results, cbind(cartGRID[i, ], tempcartMODEL[["results"]][1, c(2:5)]))
}

optimal <- results[which(results[, 5] == max(results[, 5])), ]
optimal
#optimal cp is 0
#optimal maxdepth is 15 or 18
#optimal minbucket is 3

cartMODEL <- train(
  activity ~ .,
  data = training_df,
  method = "rpart2",
  metric = "Kappa",
  trControl = trainControl(method = "none"),
  tuneGrid = expand.grid(maxdepth = 7),
  control = rpart.control(
    cp = 0,
    maxdepth = 18,
    minbucket = 3,
    xval = 0
  )
)
cartMODEL[["results"]] <- results
rm(tempcartMODEL, mdValue, results)

cartPREDICT <- predict(cartMODEL, testing_df)
cartprobPREDICT <- predict(cartMODEL, testing_df, type = "prob")
cartCM <- confusionMatrix(cartPREDICT, testing_df$activity)

cartPREDICT_training <- predict(cartMODEL, training_df)
cartprobPREDICT_training <-
  predict(cartMODEL, training_df, type = "prob")
cartCM_training <-
  confusionMatrix(cartPREDICT_training, training_df$activity)

rpart.plot(cartMODEL$finalModel)

save.image("D:/activities/models_code_SAVED_allACTIVITIES_PCA_1_AtoC.RData")

####B - Random Forest####

rfcustom <- list(
  label = "CUSTOM Random Forest",
  library = "randomForest",
  loop = NULL,
  type = c("Classification", "Regression"),
  parameters = data.frame(
    parameter = c("mtry", "nodesize"),
    #AMENDED
    class = c("numeric", "numeric"),
    #AMENDED
    label = c("#Randomly Selected Predictors", "Minimum size of terminal nodes")
  ),
  #AMENDED
  grid = function(x, y, len = NULL, search = "grid") {
    if (search == "grid") {
      out <-
        expand.grid(
          mtry = caret::var_seq(
            p = ncol(x),
            classification = is.factor(y),
            len = len
          ),
          #AMENDED
          nodesize = 1
        ) #ADDED
    } else {
      out <-
        data.frame(mtry = unique(sample(
          1:ncol(x), size = len, replace = TRUE
        )),
        nodesize = sample(1:10, replace = TRUE, size = len)) #AMENDED
    }
  },
  fit = function(x, y, wts, param, lev, last, classProbs, ...)
    randomForest::randomForest(
      x,
      y,
      mtry = min(param$mtry, ncol(x)),
      nodesize = param$nodesize,
      ...
    ),
  #AMENDED
  predict = function(modelFit, newdata, submodels = NULL)
    if (!is.null(newdata))
      predict(modelFit, newdata)
  else
    predict(modelFit),
  prob = function(modelFit, newdata, submodels = NULL)
    if (!is.null(newdata))
      predict(modelFit, newdata, type = "prob")
  else
    predict(modelFit, type = "prob"),
  predictors = function(x, ...) {
    varIndex <- as.numeric(names(table(x$forest$bestvar)))
    varIndex <- varIndex[varIndex > 0]
    varsUsed <- names(x$forest$ncat)[varIndex]
    varsUsed
  },
  varImp = function(object, ...) {
    varImp <- randomForest::importance(object, ...)
    if (object$type == "regression") {
      if ("%IncMSE" %in% colnames(varImp)) {
        varImp <- as.data.frame(varImp[, "%IncMSE", drop = FALSE])
        colnames(varImp) <- "Overall"
      } else {
        varImp <- as.data.frame(varImp[, 1, drop = FALSE])
        colnames(varImp) <- "Overall"
      }
    }
    else {
      retainNames <- levels(object$y)
      if (all(retainNames %in% colnames(varImp))) {
        varImp <- varImp[, retainNames, drop = FALSE]
      } else {
        varImp <- as.data.frame(varImp[, 1, drop = FALSE])
        colnames(varImp) <- "Overall"
      }
    }
    
    out <- as.data.frame(varImp, stringsAsFactors = TRUE)
    if (dim(out)[2] == 2) {
      tmp <- apply(out, 1, mean)
      out[, 1] <- out[, 2] <- tmp
    }
    out
  },
  levels = function(x)
    x$classes,
  tags = c(
    "Random Forest",
    "Ensemble Model",
    "Bagging",
    "Implicit Feature Selection"
  ),
  sort = function(x)
    x[order(x[, 1]), ],
  oob = function(x) {
    out <- switch(
      x$type,
      regression = c(sqrt(max(x$mse[length(x$mse)], 0)), x$rsq[length(x$rsq)]),
      classification = c(1 - x$err.rate[x$ntree, "OOB"],
                         e1071::classAgreement(x$confusion[, -dim(x$confusion)[2]])[["kappa"]])
    )
    names(out) <-
      if (x$type == "regression")
        c("RMSE", "Rsquared")
    else
      c("Accuracy", "Kappa")
    out
  }
)

rfGRID <-
  expand.grid(mtry = c(1, 2, 3, 4, seq(5, ncol(training_df), 5)),
              #number of variables randomly sampled as candidates at each split
              nodesize = c(1, 3, 6, 9, 12)) #default is 1
#default is mtry=floor(ncol(training_df)/3)
#maximum mtry is ncol(training_df)-1
#mtry is p_tk in the random forest algorithm in the dissertation, and is kept constant across all nodes across all trees

rfMODEL <- train(
  activity ~ .,
  data = training_df,
  method = rfcustom,
  metric = "Kappa",
  trControl = resamplingCTRL,
  tuneGrid = rfGRID,
  importance = TRUE
)

rfPREDICT_training <- predict(rfMODEL, training_df)
rfprobPREDICT_training <- predict(rfMODEL, training_df, type = "prob")
rfCM_training <-
  confusionMatrix(rfPREDICT_training, as.factor(training_df$activity))

rfPREDICT <- predict(rfMODEL, testing_df)
rfprobPREDICT <- predict(rfMODEL, testing_df, type = "prob")
rfCM <- confusionMatrix(rfPREDICT, as.factor(testing_df$activity))

#plotting the errors
plot(rfMODEL$finalModel, main = "Prediction Error of the Random Forest")

save.image("D:/activities/models_code_SAVED_allACTIVITIES_PCA_1_AtoC.RData")

####C - RRF and GRRF and GRF####

#TO REDO CARET MODEL FUNCTION TO GRAB IMPORTANCE FROM PREVIOUSLY TRAINED RANDOM FOREST
RRFcustom <- list(
  label = "CUSTOM Regularized Random Forest",
  library = c("randomForest", "RRF"),
  loop = NULL,
  type = c('Regression', 'Classification'),
  parameters = data.frame(
    parameter = c('mtry', 'coefReg', 'coefImp'),
    class = c('numeric', 'numeric', 'numeric'),
    label = c(
      '#Randomly Selected Predictors',
      'Regularization Value',
      'Importance Coefficient'
    )
  ),
  grid = function(x, y, len = NULL, search = "grid") {
    if (search == "grid") {
      out <- expand.grid(
        mtry = caret::var_seq(
          p = ncol(x),
          classification = is.factor(y),
          len = len
        ),
        coefReg = seq(0.01, 1, length = len),
        coefImp = seq(0, 1, length = len)
      )
    } else {
      out <-
        data.frame(
          mtry = sample(1:ncol(x), size = len, replace = TRUE),
          coefReg = runif(len, min = 0, max = 1),
          coefImp = runif(len, min = 0, max = 1)
        )
    }
    out
  },
  fit = function(x, y, wts, param, lev, last, classProbs, ...) {
    theDots <- list(...)
    theDots$importance <- TRUE
    args <- list(x = x,
                 y = y,
                 mtry = min(param$mtry, ncol(x)))
    args <- c(args, theDots)
    #firstFit <- do.call(randomForest::randomForest, args) #REMOVED
    #firstImp <- randomForest:::importance(firstFit) #REMOVED
    impRF <- randomForest:::importance(rfMODEL$finalModel) #ADDED
    firstImp <- impRF / (max(impRF)) #ADDED
    if (is.factor(y))
    {
      firstImp <-
        firstImp[, "MeanDecreaseGini"] / max(firstImp[, "MeanDecreaseGini"])
    } else
      firstImp <- firstImp[, "%IncMSE"] / max(firstImp[, "%IncMSE"])
    firstImp <-
      ((1 - param$coefImp) * param$coefReg) + (param$coefImp * firstImp)
    
    RRF::RRF(x,
             y,
             mtry = min(param$mtry, ncol(x)),
             coefReg = firstImp,
             ...)
  },
  predict = function(modelFit, newdata, submodels = NULL)
    predict(modelFit, newdata),
  prob = function(modelFit, newdata, submodels = NULL)
    predict(modelFit, newdata, type = "prob"),
  varImp = function(object, ...) {
    varImp <- RRF::importance(object, ...)
    if (object$type == "regression")
      varImp <- data.frame(Overall = varImp[, "%IncMSE"])
    else {
      retainNames <- levels(object$y)
      if (all(retainNames %in% colnames(varImp))) {
        varImp <- varImp[, retainNames]
      } else {
        varImp <- data.frame(Overall = varImp[, 1])
      }
    }
    out <- as.data.frame(varImp, stringsAsFactors = TRUE)
    if (dim(out)[2] == 2) {
      tmp <- apply(out, 1, mean)
      out[, 1] <- out[, 2] <- tmp
    }
    out
  },
  levels = function(x)
    x$obsLevels,
  tags = c(
    "Random Forest",
    "Ensemble Model",
    "Bagging",
    "Implicit Feature Selection",
    "Regularization"
  ),
  sort = function(x)
    x[order(x$coefReg), ]
)

#regularized random forest
rrfGRID <- expand.grid(
  mtry = c(seq(5, ncol(training_df), 5)),
  coefReg = seq(0.1, 1, 0.1),
  coefImp = 0
)
rrfMODEL <- train(
  activity ~ .,
  data = training_df,
  method = RRFcustom,
  metric = "Kappa",
  trControl = resamplingCTRL,
  tuneGrid = rrfGRID,
  flagReg = 1
)

rrfPREDICT_training <- predict(rrfMODEL, training_df)
rrfprobPREDICT_training <- predict(rrfMODEL, training_df, type = "prob")
rrfCM_training <-
  confusionMatrix(rrfPREDICT_training, as.factor(training_df$activity))

rrfPREDICT <- predict(rrfMODEL, testing_df)
rrfprobPREDICT <- predict(rrfMODEL, testing_df, type = "prob")
rrfCM <- confusionMatrix(rrfPREDICT, as.factor(testing_df$activity))

#guided regularized random forest
grrfGRID <- expand.grid(
  mtry = c(seq(5, ncol(training_df), 5)),
  coefReg = seq(0, 1, 0.1),
  coefImp = seq(0, 1, 0.1)
)
grrfGRID <-
  grrfGRID[-which(grrfGRID[, 2] == 0 &
                    grrfGRID[, 3] == 0), ] #excluding both 0

grrfMODEL <- train(
  activity ~ .,
  data = training_df,
  method = RRFcustom,
  metric = "Kappa",
  trControl = resamplingCTRL,
  tuneGrid = grrfGRID,
  flagReg = 1
)

grrfPREDICT_training <- predict(grrfMODEL, training_df)
grrfprobPREDICT_training <-
  predict(grrfMODEL, training_df, type = "prob")
grrfCM_training <-
  confusionMatrix(grrfPREDICT_training, as.factor(training_df$activity))

grrfPREDICT <- predict(grrfMODEL, testing_df)
grrfprobPREDICT <- predict(grrfMODEL, testing_df, type = "prob")
grrfCM <- confusionMatrix(grrfPREDICT, as.factor(testing_df$activity))

#guided random forest
grfGRID <- expand.grid(
  mtry = c(1, 2, 3, 4, seq(5, ncol(training_df), 5)),
  coefReg = 1,
  coefImp = seq(0, 1, 0.1)
)
grfMODEL <- train(
  activity ~ .,
  data = training_df,
  method = RRFcustom,
  metric = "Kappa",
  trControl = resamplingCTRL,
  tuneGrid = grfGRID,
  flagReg = 0
)

grfPREDICT_training <- predict(grfMODEL, training_df)
grfprobPREDICT_training <- predict(grfMODEL, training_df, type = "prob")
grfCM_training <-
  confusionMatrix(grfPREDICT_training, as.factor(training_df$activity))

grfPREDICT <- predict(grfMODEL, testing_df)
grfprobPREDICT <- predict(grfMODEL, testing_df, type = "prob")
grfCM <- confusionMatrix(grfPREDICT, as.factor(testing_df$activity))

save.image("D:/activities/models_code_SAVED_allACTIVITIES_PCA_1_AtoC.RData")

####D - AdaBoost####

rm(list = ls())
gc(reset = TRUE)
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_BASICS.RData")

adaGRID <- expand.grid(
  coeflearn = c("Breiman"),
  maxdepth = seq(3, 18, 3),
  mfinal = seq(100, 1000, 100)
) #iterations

adaMODEL <- train(
  activity ~ .,
  data = training_df,
  method = "AdaBoost.M1",
  metric = "Kappa",
  trControl = resamplingCTRL,
  tuneGrid = adaGRID,
  control = rpart.control(minsplit = 5, cp = 0)
)

adaPREDICT_training <- predict(adaMODEL, training_df)
adaprobPREDICT_training <- predict(adaMODEL, training_df, type = "prob")
adaCM_training <-
  confusionMatrix(adaPREDICT_training, as.factor(training_df$activity))

adaPREDICT <- predict(adaMODEL, testing_df)
adaprobPREDICT <- predict(adaMODEL, testing_df, type = "prob")
adaCM <- confusionMatrix(adaPREDICT, as.factor(testing_df$activity))

save.image("D:/activities/models_code_SAVED_allACTIVITIES_PCA_2_D.RData")

####E - AdaBoost - WEAK CLASSIFIERS####

rm(list = ls())
gc(reset = TRUE)
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_BASICS.RData")

adaWEAKGRID <- expand.grid(
  coeflearn = c("Breiman"),
  maxdepth = 1,
  mfinal = seq(100, 1000, 100)
) #iterations

adaWEAKMODEL <- train(
  activity ~ .,
  data = training_df,
  method = "AdaBoost.M1",
  metric = "Kappa",
  trControl = resamplingCTRL,
  tuneGrid = adaWEAKGRID,
  control = rpart.control(
    maxdepth = 1,
    cp = -1,
    minsplit = 0,
    xval = 0
  ) #as suggested by author for stumps
)

adaWEAKPREDICT_training <- predict(adaWEAKMODEL, training_df)
adaWEAKprobPREDICT_training <-
  predict(adaWEAKMODEL, training_df, type = "prob")
adaWEAKCM_training <-
  confusionMatrix(adaWEAKPREDICT_training, as.factor(training_df$activity))

adaWEAKPREDICT <- predict(adaWEAKMODEL, testing_df)
adaWEAKprobPREDICT <- predict(adaWEAKMODEL, testing_df, type = "prob")
adaWEAKCM <-
  confusionMatrix(adaWEAKPREDICT, as.factor(testing_df$activity))

save.image("D:/activities/models_code_SAVED_allACTIVITIES_PCA_3_EtoF.RData")

####F - LogitBoost - WEAK CLASSIFIERS####

logitWEAKGRID <- expand.grid(nIter = seq(100, 5000, 100))

logitWEAKMODEL <- train(
  activity ~ .,
  data = training_df,
  method = "LogitBoost",
  #stumps are used automatically by the package
  metric = "Kappa",
  trControl = resamplingCTRL,
  tuneGrid = logitWEAKGRID
)

logitWEAKPREDICT_training <- predict(logitWEAKMODEL, training_df)
logitWEAKprobPREDICT_training <-
  predict(logitWEAKMODEL, training_df, type = "prob")
logitWEAKCM_training <-
  confusionMatrix(logitWEAKPREDICT_training, as.factor(training_df$activity))

logitWEAKPREDICT <- predict(logitWEAKMODEL, testing_df)
logitWEAKprobPREDICT <-
  predict(logitWEAKMODEL, testing_df, type = "prob")
logitWEAKCM <-
  confusionMatrix(logitWEAKPREDICT, as.factor(testing_df$activity))

save.image("D:/activities/models_code_SAVED_allACTIVITIES_PCA_3_EtoF.RData")

####G - Gradient Boosting multinomial logistic loss function####

rm(list = ls())
gc(reset = TRUE)
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_BASICS.RData")

gbm_h2ocustom <- list(
  label = "CUSTOM Gradient Boosting Machines",
  library = "h2o",
  type = c("Regression", "Classification"),
  parameters = data.frame(
    parameter = c(
      'ntrees',
      'max_depth',
      'min_rows',
      'learn_rate',
      'col_sample_rate_per_tree',
      'sample_rate'
    ),
    class = rep("numeric", 6),
    label = c(
      '# Boosting Iterations',
      'Max Tree Depth',
      'Min. Terminal Node Size',
      'Shrinkage',
      '#Randomly Selected Predictors',
      '#Randomly Selected Observations'
    )
  ),
  grid = function(x, y, len = NULL, search = "grid") {
    if (search == "grid") {
      out <- expand.grid(
        max_depth = seq(1, len),
        ntrees = floor((1:len) * 50),
        learn_rate = .1,
        min_rows = 10,
        sample_rate = 1,
        col_sample_rate_per_tree = 1
      )
    } else {
      out <- data.frame(
        ntrees = floor(runif(len, min = 1, max = 5000)),
        max_depth = sample(1:10, replace = TRUE, size = len),
        learn_rate = runif(len, min = .001, max = .6),
        min_rows = sample(5:25, replace = TRUE, size = len),
        sample_rate = runif(len),
        col_sample_rate_per_tree = runif(len)
      )
    }
    out
  },
  loop = NULL,
  fit = function(x, y, wts, param, lev, last, classProbs, ...) {
    lvs <- length(levels(y))
    fam <- "gaussian"
    if (lvs == 2)
      fam <- "bernoulli"
    if (lvs > 2)
      fam <- "multinomial" ## intercept ... for family arg
    
    dat <-
      if (!is.data.frame(x))
        as.data.frame(x, stringsAsFactors = TRUE)
    else
      x
    dat$.outcome <- y
    frame_name <- paste0("tmp_gbm_dat_", sample.int(100000, 1))
    tmp_train_dat = h2o::as.h2o(dat, destination_frame = frame_name)
    
    out <- h2o::h2o.gbm(
      x = colnames(x),
      y = ".outcome",
      training_frame = tmp_train_dat,
      distribution = fam,
      ntrees = param$ntrees,
      max_depth = param$max_depth,
      learn_rate = param$learn_rate,
      min_rows = param$min_rows,
      col_sample_rate_per_tree = param$col_sample_rate_per_tree,
      sample_rate = param$sample_rate,
      ...
    )
    h2o::h2o.getModel(out@model_id)
  },
  predict = function(modelFit, newdata, submodels = NULL) {
    frame_name <- paste0("new_gbm_dat_", sample.int(100000, 1))
    newdata <- h2o::as.h2o(newdata, destination_frame = frame_name)
    as.data.frame(predict(modelFit, newdata), stringsAsFactors = TRUE)[, 1]
  },
  prob = function(modelFit, newdata, submodels = NULL) {
    frame_name <- paste0("new_gbm_dat_", sample.int(100000, 1))
    newdata <- h2o::as.h2o(newdata, destination_frame = frame_name)
    as.data.frame(predict(modelFit, newdata), stringsAsFactors = TRUE)[, -1]
  },
  predictors = function(x, ...) {
    out <- as.data.frame(h2o::h2o.varimp(x), stringsAsFactors = TRUE)
    out <- subset(out, relative_importance > 0)
    as.character(out$variable)
  },
  varImp = function(object, ...) {
    out <-
      as.data.frame(h2o::h2o.varimp(object), stringsAsFactors = TRUE)
    colnames(out)[colnames(out) == "relative_importance"] <- "Overall"
    rownames(out) <- out$variable
    out[, c("Overall"), drop = FALSE]
  },
  levels = function(x)
    x@model$training_metrics@metrics$domain,
  tags = c(
    "Tree-Based Model",
    "Boosting",
    "Ensemble Model",
    "Implicit Feature Selection"
  ),
  sort = function(x)
    x[order(x$ntrees, x$max_depth, x$learn_rate), ],
  trim = NULL
)

h2o.init()

gbmGRID <- expand.grid(
  max_depth = c(3, 5, 7),
  ntrees = c(25, 50, 100, 150, 200),
  min_rows = c(3, 5, 7),
  #minimum number of observations in a node
  learn_rate = c(0.05, 0.1, 0.25, 0.3, 0.35),
  #shrinkage
  sample_rate = c(0.5, 0.75, 1),
  #percentage of training data to be sampled per tree, <1 implies stochastic Gradient Boosting
  col_sample_rate_per_tree = c(0.5, 0.75, 1)
) #percentage of predictors to be sampled per tree


gbmmultiMODEL <- train(
  activity ~ .,
  data = training_df,
  method = gbm_h2ocustom,
  metric = "Kappa",
  trControl = resamplingCTRL,
  tuneGrid = gbmGRID
)

gbmmultiMODEL_2 <- train(
  activity ~ .,
  data = training_df,
  method = gbm_h2ocustom,
  metric = "Kappa",
  trControl = resamplingCTRL,
  tuneGrid = gbmmultiMODEL[["bestTune"]]
)

gbmmultiPREDICT_training <- predict(gbmmultiMODEL, training_df)
gbmmultiprobPREDICT_training <-
  predict(gbmmultiMODEL, training_df, type = "prob")
gbmmultiCM_training <-
  confusionMatrix(gbmmultiPREDICT_training, as.factor(training_df$activity))

gbmmultiPREDICT <- predict(gbmmultiMODEL, testing_df)
gbmmultiprobPREDICT <- predict(gbmmultiMODEL, testing_df, type = "prob")
gbmmultiCM <-
  confusionMatrix(gbmmultiPREDICT, as.factor(testing_df$activity))

save.image("D:/activities/models_code_SAVED_allACTIVITIES_PCA_4_G.RData")

####H - EXTREME GRADIENT BOOSTING - TREES####

rm(list = ls())
gc(reset = TRUE)
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_BASICS.RData")

xgbcustom <- list(
  label = "CUSTOM Extreme Gradient Boosting",
  library = c("xgboost", "plyr"),
  type = c("Regression", "Classification"),
  parameters = data.frame(
    parameter = c(
      'nrounds',
      'max_depth',
      'eta',
      'gamma',
      'colsample_bytree',
      'lambda',
      'subsample'
    ),
    class = rep("numeric", 7),
    label = c(
      '# Boosting Iterations',
      'Max Tree Depth',
      'Shrinkage',
      "Minimum Loss Reduction",
      'Subsample Ratio of Columns',
      'L2 Regularization',
      'Subsample Percentage'
    )
  ),
  grid = function(x, y, len = NULL, search = "grid") {
    if (search == "grid") {
      out <- expand.grid(
        max_depth = seq(1, len),
        nrounds = floor((1:len) * 50),
        eta = c(.3, .4),
        gamma = 0,
        colsample_bytree = c(.6, .8),
        lambda = c(0.5, 1, 2),
        subsample = seq(.5, 1, length = len)
      )
    } else {
      out <-
        data.frame(
          nrounds = sample(1:1000, size = len, replace = TRUE),
          max_depth = sample(1:10, replace = TRUE, size = len),
          eta = runif(len, min = .001, max = .6),
          gamma = runif(len, min = 0, max = 10),
          colsample_bytree = runif(len, min = .3, max = .7),
          lambda = runif(len, min = 0.25, max = 2.00),
          subsample = runif(len, min = .25, max = 1)
        )
      out$nrounds <- floor(out$nrounds)
    }
    out
  },
  loop = function(grid) {
    loop <- plyr::ddply(grid, c(
      "eta",
      "max_depth",
      "gamma",
      "colsample_bytree",
      "lambda",
      "subsample"
    ),
    function(x)
      c(nrounds = max(x$nrounds)))
    submodels <- vector(mode = "list", length = nrow(loop))
    for (i in seq(along = loop$nrounds)) {
      index <- which(
        grid$max_depth == loop$max_depth[i] &
          grid$eta == loop$eta[i] &
          grid$gamma == loop$gamma[i] &
          grid$colsample_bytree == loop$colsample_bytree[i] &
          grid$lambda == loop$lambda[i] &
          grid$subsample == loop$subsample[i]
      )
      trees <- grid[index, "nrounds"]
      submodels[[i]] <-
        data.frame(nrounds = trees[trees != loop$nrounds[i]])
    }
    list(loop = loop, submodels = submodels)
  },
  fit = function(x, y, wts, param, lev, last, classProbs, ...) {
    if (!inherits(x, "xgb.DMatrix"))
      x <- as.matrix(x)
    
    if (is.factor(y)) {
      if (length(lev) == 2) {
        y <- ifelse(y == lev[1], 1, 0)
        
        if (!inherits(x, "xgb.DMatrix"))
          x <- xgboost::xgb.DMatrix(x, label = y, missing = NA)
        else
          xgboost::setinfo(x, "label", y)
        
        if (!is.null(wts))
          xgboost::setinfo(x, 'weight', wts)
        
        out <- xgboost::xgb.train(
          list(
            eta = param$eta,
            max_depth = param$max_depth,
            gamma = param$gamma,
            colsample_bytree = param$colsample_bytree,
            lambda = param$lambda,
            subsample = param$subsample
          ),
          data = x,
          nrounds = param$nrounds,
          objective = "binary:logistic",
          ...
        )
      } else {
        y <- as.numeric(y) - 1
        
        if (!inherits(x, "xgb.DMatrix"))
          x <- xgboost::xgb.DMatrix(x, label = y, missing = NA)
        else
          xgboost::setinfo(x, "label", y)
        
        if (!is.null(wts))
          xgboost::setinfo(x, 'weight', wts)
        
        out <- xgboost::xgb.train(
          list(
            eta = param$eta,
            max_depth = param$max_depth,
            gamma = param$gamma,
            colsample_bytree = param$colsample_bytree,
            lambda = param$lambda,
            subsample = param$subsample
          ),
          data = x,
          num_class = length(lev),
          nrounds = param$nrounds,
          objective = "multi:softprob",
          ...
        )
      }
    } else {
      if (!inherits(x, "xgb.DMatrix"))
        x <- xgboost::xgb.DMatrix(x, label = y, missing = NA)
      else
        xgboost::setinfo(x, "label", y)
      
      if (!is.null(wts))
        xgboost::setinfo(x, 'weight', wts)
      
      out <- xgboost::xgb.train(
        list(
          eta = param$eta,
          max_depth = param$max_depth,
          gamma = param$gamma,
          colsample_bytree = param$colsample_bytree,
          lambda = param$lambda,
          subsample = param$subsample
        ),
        data = x,
        nrounds = param$nrounds,
        objective = "reg:squarederror",
        ...
      )
    }
    out
    
    
  },
  predict = function(modelFit, newdata, submodels = NULL) {
    if (!inherits(newdata, "xgb.DMatrix")) {
      newdata <- as.matrix(newdata)
      newdata <- xgboost::xgb.DMatrix(data = newdata, missing = NA)
    }
    out <- predict(modelFit, newdata)
    if (modelFit$problemType == "Classification") {
      if (length(modelFit$obsLevels) == 2) {
        out <- ifelse(out >= .5,
                      modelFit$obsLevels[1],
                      modelFit$obsLevels[2])
      } else {
        out <- matrix(out,
                      ncol = length(modelFit$obsLevels),
                      byrow = TRUE)
        out <- modelFit$obsLevels[apply(out, 1, which.max)]
      }
    }
    
    if (!is.null(submodels)) {
      tmp <- vector(mode = "list", length = nrow(submodels) + 1)
      tmp[[1]] <- out
      for (j in seq(along = submodels$nrounds)) {
        tmp_pred <-
          predict(modelFit, newdata, ntreelimit = submodels$nrounds[j])
        if (modelFit$problemType == "Classification") {
          if (length(modelFit$obsLevels) == 2) {
            tmp_pred <- ifelse(tmp_pred >= .5,
                               modelFit$obsLevels[1],
                               modelFit$obsLevels[2])
          } else {
            tmp_pred <-
              matrix(tmp_pred,
                     ncol = length(modelFit$obsLevels),
                     byrow = TRUE)
            tmp_pred <- modelFit$obsLevels[apply(tmp_pred, 1, which.max)]
          }
        }
        tmp[[j + 1]] <- tmp_pred
      }
      out <- tmp
    }
    out
  },
  prob = function(modelFit, newdata, submodels = NULL) {
    if (!inherits(newdata, "xgb.DMatrix")) {
      newdata <- as.matrix(newdata)
      newdata <- xgboost::xgb.DMatrix(data = newdata, missing = NA)
    }
    
    if (!is.null(modelFit$param$objective) &&
        modelFit$param$objective == 'binary:logitraw') {
      p <- predict(modelFit, newdata)
      out <- binomial()$linkinv(p) # exp(p)/(1+exp(p))
    } else {
      out <- predict(modelFit, newdata)
    }
    if (length(modelFit$obsLevels) == 2) {
      out <- cbind(out, 1 - out)
      colnames(out) <- modelFit$obsLevels
    } else {
      out <- matrix(out,
                    ncol = length(modelFit$obsLevels),
                    byrow = TRUE)
      colnames(out) <- modelFit$obsLevels
    }
    out <- as.data.frame(out, stringsAsFactors = TRUE)
    
    if (!is.null(submodels)) {
      tmp <- vector(mode = "list", length = nrow(submodels) + 1)
      tmp[[1]] <- out
      for (j in seq(along = submodels$nrounds)) {
        tmp_pred <-
          predict(modelFit, newdata, ntreelimit = submodels$nrounds[j])
        if (length(modelFit$obsLevels) == 2) {
          tmp_pred <- cbind(tmp_pred, 1 - tmp_pred)
          colnames(tmp_pred) <- modelFit$obsLevels
        } else {
          tmp_pred <-
            matrix(tmp_pred,
                   ncol = length(modelFit$obsLevels),
                   byrow = TRUE)
          colnames(tmp_pred) <- modelFit$obsLevels
        }
        tmp_pred <- as.data.frame(tmp_pred, stringsAsFactors = TRUE)
        tmp[[j + 1]] <- tmp_pred
      }
      out <- tmp
    }
    out
  },
  predictors = function(x, ...) {
    imp <- xgboost::xgb.importance(x$xNames, model = x)
    x$xNames[x$xNames %in% imp$Feature]
  },
  varImp = function(object, numTrees = NULL, ...) {
    imp <- xgboost::xgb.importance(object$xNames, model = object)
    imp <- as.data.frame(imp, stringsAsFactors = TRUE)[, 1:2]
    rownames(imp) <- as.character(imp[, 1])
    imp <- imp[, 2, drop = FALSE]
    colnames(imp) <- "Overall"
    
    missing <- object$xNames[!(object$xNames %in% rownames(imp))]
    missing_imp <- data.frame(Overall = rep(0, times = length(missing)))
    rownames(missing_imp) <- missing
    imp <- rbind(imp, missing_imp)
    
    imp
  },
  levels = function(x)
    x$obsLevels,
  tags = c(
    "Tree-Based Model",
    "Boosting",
    "Ensemble Model",
    "Implicit Feature Selection",
    "Accepts Case Weights"
  ),
  sort = function(x) {
    x[order(x$nrounds,
            x$max_depth,
            x$eta,
            x$gamma,
            x$colsample_bytree,
            x$lambda), ]
  }
)

xgbTreeGRID <-
  expand.grid(
    nrounds = c(50, 100, 150, 200, 300),
    #max boosting iterations - no default
    max_depth = c(3, 5, 7),
    #max depth of a tree - default 6
    eta = c(0.05, 0.1, 0.25),
    #shrinkage (scale contribution of each tree) - default 0.3
    gamma = c(0, 1),
    #coefficient of M_b - default 0
    lambda = c(0.5, 1, 2),
    #L2 regularization - default 1
    colsample_bytree = c(0.5, 0.75, 1),
    #column sample - default 1
    subsample = c(0.5, 0.75, 1)
  ) #observation sample - default 1)

xgbTreeMODEL <- train(
  activity ~ .,
  data = training_df,
  method = xgbcustom,
  metric = "Kappa",
  trControl = resamplingCTRL,
  tuneGrid = xgbTreeGRID,
  verbosity = 0
)

xgbTreePREDICT_training <- predict(xgbTreeMODEL, training_df)
xgbTreeprobPREDICT_training <-
  predict(xgbTreeMODEL, training_df, type = "prob")
xgbTreeCM_training <-
  confusionMatrix(xgbTreePREDICT_training, as.factor(training_df$activity))

xgbTreePREDICT <- predict(xgbTreeMODEL, testing_df)
xgbTreeprobPREDICT <- predict(xgbTreeMODEL, testing_df, type = "prob")
xgbTreeCM <-
  confusionMatrix(xgbTreePREDICT, as.factor(testing_df$activity))

save.image("D:/activities/models_code_SAVED_allACTIVITIES_PCA_5_H.RData")

####I - EXTREME GRADIENT BOOSTING - TREES WITH DROPOUT####

rm(list = ls()[!(ls() %in% c("xgbTreeMODEL"))])
gc(reset = TRUE)
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_BASICS.RData")

xgbdartcustom <-
  list(
    label = "CUSTOM DART Extreme Gradient Boosting",
    library = c("xgboost", "plyr"),
    type = c("Regression", "Classification"),
    parameters = data.frame(
      parameter = c(
        'nrounds',
        'max_depth',
        'eta',
        'gamma',
        'colsample_bytree',
        'lambda',
        'subsample',
        'rate_drop',
        'skip_drop'
      ),
      class = rep("numeric", 9),
      label = c(
        '# Boosting Iterations',
        'Max Tree Depth',
        'Shrinkage',
        "Minimum Loss Reduction",
        'Subsample Ratio of Columns',
        'L2 Regularization',
        'Subsample Percentage',
        "Fraction of Trees Dropped",
        "Prob. of Skipping Drop-out"
      )
    ),
    grid = function(x, y, len = NULL, search = "grid") {
      if (search == "grid") {
        out <- expand.grid(
          max_depth = seq(1, len),
          nrounds = floor((1:len) * 50),
          eta = c(.3, .4),
          gamma = 0,
          colsample_bytree = c(.6, .8),
          lambda = c(0.5, 1, 2),
          subsample = seq(.5, 1, length = len),
          rate_drop = c(0.01, 0.50),
          skip_drop = c(0.05, 0.95)
        )
      } else {
        out <-
          data.frame(
            nrounds = sample(1:1000, size = len, replace = TRUE),
            max_depth = sample(1:10, replace = TRUE, size = len),
            eta = runif(len, min = .001, max = .6),
            gamma = runif(len, min = 0, max = 10),
            colsample_bytree = runif(len, min = .3, max = .7),
            lambda = runif(len, min = 0.25, max = 2.00),
            subsample = runif(len, min = .25, max = 1),
            rate_drop = runif(len, min = 0.01, max = 0.50),
            skip_drop = runif(len, min = 0.05, max = 0.95)
          )
        out$nrounds <- floor(out$nrounds)
      }
      out
    },
    loop = function(grid) {
      loop <- plyr::ddply(grid, c(
        "eta",
        "max_depth",
        "gamma",
        "colsample_bytree",
        "lambda",
        "subsample",
        "rate_drop",
        "skip_drop"
      ),
      function(x)
        c(nrounds = max(x$nrounds)))
      submodels <- vector(mode = "list", length = nrow(loop))
      for (i in seq(along = loop$nrounds)) {
        index <- which(
          grid$max_depth == loop$max_depth[i] &
            grid$eta == loop$eta[i] &
            grid$gamma == loop$gamma[i] &
            grid$colsample_bytree == loop$colsample_bytree[i] &
            grid$lambda == loop$lambda[i] &
            grid$subsample == loop$subsample[i] &
            grid$rate_drop == loop$rate_drop[i] &
            grid$skip_drop == loop$skip_drop[i]
        )
        trees <- grid[index, "nrounds"]
        submodels[[i]] <-
          data.frame(nrounds = trees[trees != loop$nrounds[i]])
      }
      list(loop = loop, submodels = submodels)
    },
    fit = function(x, y, wts, param, lev, last, classProbs, ...) {
      if (!inherits(x, "xgb.DMatrix"))
        x <- as.matrix(x)
      
      if (is.factor(y)) {
        if (length(lev) == 2) {
          y <- ifelse(y == lev[1], 1, 0)
          
          if (!inherits(x, "xgb.DMatrix"))
            x <- xgboost::xgb.DMatrix(x, label = y, missing = NA)
          else
            xgboost::setinfo(x, "label", y)
          
          if (!is.null(wts))
            xgboost::setinfo(x, 'weight', wts)
          
          out <- xgboost::xgb.train(
            list(
              eta = param$eta,
              max_depth = param$max_depth,
              gamma = param$gamma,
              colsample_bytree = param$colsample_bytree,
              lambda = param$lambda,
              subsample = param$subsample,
              rate_drop = param$rate_drop,
              skip_drop = param$skip_drop
            ),
            data = x,
            nrounds = param$nrounds,
            objective = "binary:logistic",
            booster = "dart",
            ...
          )
        } else {
          y <- as.numeric(y) - 1
          
          if (!inherits(x, "xgb.DMatrix"))
            x <- xgboost::xgb.DMatrix(x, label = y, missing = NA)
          else
            xgboost::setinfo(x, "label", y)
          
          if (!is.null(wts))
            xgboost::setinfo(x, 'weight', wts)
          
          out <- xgboost::xgb.train(
            list(
              eta = param$eta,
              max_depth = param$max_depth,
              gamma = param$gamma,
              colsample_bytree = param$colsample_bytree,
              lambda = param$lambda,
              subsample = param$subsample,
              rate_drop = param$rate_drop,
              skip_drop = param$skip_drop
            ),
            data = x,
            num_class = length(lev),
            nrounds = param$nrounds,
            objective = "multi:softprob",
            booster = "dart",
            ...
          )
        }
      } else {
        if (!inherits(x, "xgb.DMatrix"))
          x <- xgboost::xgb.DMatrix(x, label = y, missing = NA)
        else
          xgboost::setinfo(x, "label", y)
        
        if (!is.null(wts))
          xgboost::setinfo(x, 'weight', wts)
        
        out <- xgboost::xgb.train(
          list(
            eta = param$eta,
            max_depth = param$max_depth,
            gamma = param$gamma,
            colsample_bytree = param$colsample_bytree,
            lambda = param$lambda,
            subsample = param$subsample,
            rate_drop = param$rate_drop,
            skip_drop = param$skip_drop
          ),
          data = x,
          nrounds = param$nrounds,
          objective = "reg:squarederror",
          booster = "dart",
          ...
        )
      }
      out
      
      
    },
    predict = function(modelFit, newdata, submodels = NULL) {
      if (!inherits(newdata, "xgb.DMatrix")) {
        newdata <- as.matrix(newdata)
        newdata <- xgboost::xgb.DMatrix(data = newdata, missing = NA)
      }
      out <- predict(modelFit, newdata)
      if (modelFit$problemType == "Classification") {
        if (length(modelFit$obsLevels) == 2) {
          out <- ifelse(out >= .5,
                        modelFit$obsLevels[1],
                        modelFit$obsLevels[2])
        } else {
          out <- matrix(out,
                        ncol = length(modelFit$obsLevels),
                        byrow = TRUE)
          out <- modelFit$obsLevels[apply(out, 1, which.max)]
        }
      }
      
      if (!is.null(submodels)) {
        tmp <- vector(mode = "list", length = nrow(submodels) + 1)
        tmp[[1]] <- out
        for (j in seq(along = submodels$nrounds)) {
          tmp_pred <-
            predict(modelFit, newdata, ntreelimit = submodels$nrounds[j])
          if (modelFit$problemType == "Classification") {
            if (length(modelFit$obsLevels) == 2) {
              tmp_pred <- ifelse(tmp_pred >= .5,
                                 modelFit$obsLevels[1],
                                 modelFit$obsLevels[2])
            } else {
              tmp_pred <-
                matrix(tmp_pred,
                       ncol = length(modelFit$obsLevels),
                       byrow = TRUE)
              tmp_pred <- modelFit$obsLevels[apply(tmp_pred, 1, which.max)]
            }
          }
          tmp[[j + 1]] <- tmp_pred
        }
        out <- tmp
      }
      out
    },
    prob = function(modelFit, newdata, submodels = NULL) {
      if (!inherits(newdata, "xgb.DMatrix")) {
        newdata <- as.matrix(newdata)
        newdata <- xgboost::xgb.DMatrix(data = newdata, missing = NA)
      }
      
      if (!is.null(modelFit$param$objective) &&
          modelFit$param$objective == 'binary:logitraw') {
        p <- predict(modelFit, newdata)
        out <- binomial()$linkinv(p) # exp(p)/(1+exp(p))
      } else {
        out <- predict(modelFit, newdata)
      }
      if (length(modelFit$obsLevels) == 2) {
        out <- cbind(out, 1 - out)
        colnames(out) <- modelFit$obsLevels
      } else {
        out <- matrix(out,
                      ncol = length(modelFit$obsLevels),
                      byrow = TRUE)
        colnames(out) <- modelFit$obsLevels
      }
      out <- as.data.frame(out, stringsAsFactors = TRUE)
      
      if (!is.null(submodels)) {
        tmp <- vector(mode = "list", length = nrow(submodels) + 1)
        tmp[[1]] <- out
        for (j in seq(along = submodels$nrounds)) {
          tmp_pred <-
            predict(modelFit, newdata, ntreelimit = submodels$nrounds[j])
          if (length(modelFit$obsLevels) == 2) {
            tmp_pred <- cbind(tmp_pred, 1 - tmp_pred)
            colnames(tmp_pred) <- modelFit$obsLevels
          } else {
            tmp_pred <-
              matrix(tmp_pred,
                     ncol = length(modelFit$obsLevels),
                     byrow = TRUE)
            colnames(tmp_pred) <- modelFit$obsLevels
          }
          tmp_pred <- as.data.frame(tmp_pred, stringsAsFactors = TRUE)
          tmp[[j + 1]] <- tmp_pred
        }
        out <- tmp
      }
      out
    },
    predictors = function(x, ...) {
      imp <- xgboost::xgb.importance(x$xNames, model = x)
      x$xNames[x$xNames %in% imp$Feature]
    },
    varImp = function(object, numTrees = NULL, ...) {
      imp <- xgboost::xgb.importance(object$xNames, model = object)
      imp <- as.data.frame(imp, stringsAsFactors = TRUE)[, 1:2]
      rownames(imp) <- as.character(imp[, 1])
      imp <- imp[, 2, drop = FALSE]
      colnames(imp) <- "Overall"
      
      missing <- object$xNames[!(object$xNames %in% rownames(imp))]
      missing_imp <- data.frame(Overall = rep(0, times = length(missing)))
      rownames(missing_imp) <- missing
      imp <- rbind(imp, missing_imp)
      
      imp
    },
    levels = function(x)
      x$obsLevels,
    tags = c(
      "Tree-Based Model",
      "Boosting",
      "Ensemble Model",
      "Implicit Feature Selection",
      "Accepts Case Weights"
    ),
    sort = function(x) {
      x[order(
        x$nrounds,
        x$max_depth,
        x$eta,
        x$gamma,
        x$colsample_bytree,
        x$lambda,
        x$rate_drop,
        x$skip_drop
      ), ]
    }
  )

xgbDARTGRID_min <-
  expand.grid(
    nrounds = xgbTreeMODEL[["bestTune"]][["nrounds"]],
    max_depth = xgbTreeMODEL[["bestTune"]][["max_depth"]],
    eta = xgbTreeMODEL[["bestTune"]][["eta"]],
    gamma = xgbTreeMODEL[["bestTune"]][["gamma"]],
    colsample_bytree = xgbTreeMODEL[["bestTune"]][["colsample_bytree"]],
    subsample = xgbTreeMODEL[["bestTune"]][["subsample"]],
    lambda = xgbTreeMODEL[["bestTune"]][["lambda"]],
    rate_drop = seq(0, 1, 0.1),
    skip_drop = seq(0, 1, 0.1)
  )

xgbDARTMODEL_min <- train(
  activity ~ .,
  data = training_df,
  method = xgbdartcustom,
  metric = "Kappa",
  trControl = resamplingCTRL,
  tuneGrid = xgbDARTGRID_min,
  verbosity = 0
)

xgbDARTPREDICT_min_training <- predict(xgbDARTMODEL_min, training_df)
xgbDARTprobPREDICT_min_training <-
  predict(xgbDARTMODEL_min, training_df, type = "prob")
xgbDARTCM_min_training <-
  confusionMatrix(xgbDARTPREDICT_min_training,
                  as.factor(training_df$activity))

xgbDARTPREDICT_min <- predict(xgbDARTMODEL_min, testing_df)
xgbDARTprobPREDICT_min <-
  predict(xgbDARTMODEL_min, testing_df, type = "prob")
xgbDARTCM_min <-
  confusionMatrix(xgbDARTPREDICT_min, as.factor(testing_df$activity))

rm(xgbTreeMODEL)
save.image("D:/activities/models_code_SAVED_allACTIVITIES_PCA_6_I.RData")

xgbDARTGRID <-
  expand.grid(
    nrounds = c(25, 50, 100, 150, 200),
    #max boosting iterations - no default
    max_depth = c(3, 5, 7),
    #max depth of a tree - default 6
    eta = c(0.05, 0.1, 0.25, 0.3, 0.35),
    #shrinkage (scale contribution of each tree) - default 0.3
    gamma = c(0, 1),
    #coefficient of M_b - default 0
    lambda = c(0.5, 1, 2),
    #L2 regularization - default 1
    colsample_bytree = c(0.5, 0.75, 1),
    #column sample - default 1
    subsample = c(0.5, 0.75, 1),
    #observation sample - default 1
    rate_drop = c(0.25, 0.5),
    skip_drop = c(0.25, 0.5)
  )

xgbDARTMODEL <- train(
  activity ~ .,
  data = training_df,
  method = xgbdartcustom,
  metric = "Kappa",
  trControl = resamplingCTRL,
  tuneGrid = xgbDARTGRID,
  verbosity = 0
)

xgbDARTPREDICT_training <- predict(xgbDARTMODEL, training_df)
xgbDARTprobPREDICT_training <-
  predict(xgbDARTMODEL, training_df, type = "prob")
xgbDARTCM_training <-
  confusionMatrix(xgbDARTPREDICT_training, as.factor(training_df$activity))

xgbDARTPREDICT <- predict(xgbDARTMODEL, testing_df)
xgbDARTprobPREDICT <- predict(xgbDARTMODEL, testing_df, type = "prob")
xgbDARTCM <-
  confusionMatrix(xgbDARTPREDICT, as.factor(testing_df$activity))

save.image("D:/activities/models_code_SAVED_allACTIVITIES_PCA_6_I.RData")

####SUMMARY - TESTING####

rm(list = ls())

modelNAMES <- c(
  "cartCM",
  "rfCM",
  "rrfCM",
  "grrfCM",
  "grfCM",
  "adaCM",
  "adaWEAKCM",
  "logitWEAKCM",
  "gbmmultiCM",
  "xgbTreeCM",
  "xgbDARTCM",
  "xgbDARTCM_min"
)

modelNAMESPREDICT <- c(
  "cartPREDICT",
  "rfPREDICT",
  "rrfPREDICT",
  "grrfPREDICT",
  "grfPREDICT",
  "adaPREDICT",
  "adaWEAKPREDICT",
  "logitWEAKPREDICT",
  "gbmmultiPREDICT",
  "xgbTreePREDICT",
  "xgbDARTPREDICT"
)

modelNAMESprobPREDICT <- c(
  "cartprobPREDICT",
  "rfprobPREDICT",
  "rrfprobPREDICT",
  "grrfprobPREDICT",
  "grfprobPREDICT",
  "adaprobPREDICT",
  "adaWEAKprobPREDICT",
  "logitWEAKprobPREDICT",
  "gbmmultiprobPREDICT",
  "xgbTreeprobPREDICT",
  "xgbDARTprobPREDICT"
)

load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_1_AtoC.RData")
rm(list = ls()[!(
  ls() %in% c(
    modelNAMES,
    modelNAMESPREDICT,
    modelNAMESprobPREDICT,
    "modelNAMES",
    "modelNAMESPREDICT",
    "modelNAMESprobPREDICT"
  )
)])
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_2_D.RData")
rm(list = ls()[!(
  ls() %in% c(
    modelNAMES,
    modelNAMESPREDICT,
    modelNAMESprobPREDICT,
    "modelNAMES",
    "modelNAMESPREDICT",
    "modelNAMESprobPREDICT"
  )
)])
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_3_EtoF.RData")
rm(list = ls()[!(
  ls() %in% c(
    modelNAMES,
    modelNAMESPREDICT,
    modelNAMESprobPREDICT,
    "modelNAMES",
    "modelNAMESPREDICT",
    "modelNAMESprobPREDICT"
  )
)])
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_4_G.RData")
rm(list = ls()[!(
  ls() %in% c(
    modelNAMES,
    modelNAMESPREDICT,
    modelNAMESprobPREDICT,
    "modelNAMES",
    "modelNAMESPREDICT",
    "modelNAMESprobPREDICT"
  )
)])
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_5_H.RData")
rm(list = ls()[!(
  ls() %in% c(
    modelNAMES,
    modelNAMESPREDICT,
    modelNAMESprobPREDICT,
    "modelNAMES",
    "modelNAMESPREDICT",
    "modelNAMESprobPREDICT"
  )
)])
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_6_I.RData")
rm(list = ls()[!(
  ls() %in% c(
    modelNAMES,
    modelNAMESPREDICT,
    modelNAMESprobPREDICT,
    "modelNAMES",
    "modelNAMESPREDICT",
    "modelNAMESprobPREDICT"
  )
)])
gc(reset = TRUE)

summaryMETRICS <-
  mat.or.vec(length(modelNAMES) - 1, 2) #-1 to exclude xgbDARTCM_min
colnames(summaryMETRICS) <- c("Accuracy", "Kappa")
row.names(summaryMETRICS) <- c(
  "Decision Tree",
  "Random Forest",
  "Regularized Random Forests",
  "Guided Regularized Random Forests",
  "Guided Random Forests",
  "AdaBoost.M1 (NOT WEAK)",
  "AdaBoost.M1 (WEAK CLASSIFIERS)",
  "LogitBoost (WEAK CLASSIFIERS)",
  "(Stochastic) Gradient Boosting using MULTINOMIAL Loss",
  "EXTREME GRADIENT BOOSTING - TREES",
  "EXTREME GRADIENT BOOSTING - TREES WITH DROPOUT"
)

for (i in 1:nrow(summaryMETRICS)) {
  tryCatch({
    #this is to ignore errors like MODEL not found
    summaryMETRICS[i, 1] = as.numeric(eval(parse(
      text = paste(modelNAMES[i], "$overall[1]", sep = "")
    ))) #Accuracy
    summaryMETRICS[i, 2] = as.numeric(eval(parse(
      text = paste(modelNAMES[i], "$overall[2]", sep = "")
    ))) #Kappa
  }, error = function(e) {
  }) #this is to ignore errors like MODEL not found
}

summaryMETRICS[order(summaryMETRICS[, 2], decreasing = TRUE), ]

save.image("D:/activities/models_code_SAVED_allACTIVITIES_PCA_RESULTS_ONLY.RData")

####SUMMARY - TRAINING####

rm(list = ls())

modelNAMES_training <- c(
  "cartCM_training",
  "rfCM_training",
  "rrfCM_training",
  "grrfCM_training",
  "grfCM_training",
  "adaCM_training",
  "adaWEAKCM_training",
  "logitWEAKCM_training",
  "gbmmultiCM_training",
  "xgbTreeCM_training",
  "xgbDARTCM_training"
)

modelNAMESPREDICT_training <- c(
  "cartPREDICT_training",
  "rfPREDICT_training",
  "rrfPREDICT_training",
  "grrfPREDICT_training",
  "grfPREDICT_training",
  "adaPREDICT_training",
  "adaWEAKPREDICT_training",
  "logitWEAKPREDICT_training",
  "gbmmultiPREDICT_training",
  "xgbTreePREDICT_training",
  "xgbDARTPREDICT_training"
)

modelNAMESprobPREDICT_training <- c(
  "cartprobPREDICT_training",
  "rfprobPREDICT_training",
  "rrfprobPREDICT_training",
  "grrfprobPREDICT_training",
  "grfprobPREDICT_training",
  "adaprobPREDICT_training",
  "adaWEAKprobPREDICT_training",
  "logitWEAKprobPREDICT_training",
  "gbmmultiprobPREDICT_training",
  "xgbTreeprobPREDICT_training",
  "xgbDARTprobPREDICT_training"
)

load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_1_AtoC.RData")
rm(list = ls()[!(
  ls() %in% c(
    modelNAMES_training,
    modelNAMESPREDICT_training,
    modelNAMESprobPREDICT_training,
    "modelNAMES_training",
    "modelNAMESPREDICT_training",
    "modelNAMESprobPREDICT_training"
  )
)])
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_2_D.RData")
rm(list = ls()[!(
  ls() %in% c(
    modelNAMES_training,
    modelNAMESPREDICT_training,
    modelNAMESprobPREDICT_training,
    "modelNAMES_training",
    "modelNAMESPREDICT_training",
    "modelNAMESprobPREDICT_training"
  )
)])
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_3_EtoF.RData")
rm(list = ls()[!(
  ls() %in% c(
    modelNAMES_training,
    modelNAMESPREDICT_training,
    modelNAMESprobPREDICT_training,
    "modelNAMES_training",
    "modelNAMESPREDICT_training",
    "modelNAMESprobPREDICT_training"
  )
)])
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_4_G.RData")
rm(list = ls()[!(
  ls() %in% c(
    modelNAMES_training,
    modelNAMESPREDICT_training,
    modelNAMESprobPREDICT_training,
    "modelNAMES_training",
    "modelNAMESPREDICT_training",
    "modelNAMESprobPREDICT_training"
  )
)])
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_5_H.RData")
rm(list = ls()[!(
  ls() %in% c(
    modelNAMES_training,
    modelNAMESPREDICT_training,
    modelNAMESprobPREDICT_training,
    "modelNAMES_training",
    "modelNAMESPREDICT_training",
    "modelNAMESprobPREDICT_training"
  )
)])
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_6_I.RData")
rm(list = ls()[!(
  ls() %in% c(
    modelNAMES_training,
    modelNAMESPREDICT_training,
    modelNAMESprobPREDICT_training,
    "modelNAMES_training",
    "modelNAMESPREDICT_training",
    "modelNAMESprobPREDICT_training"
  )
)])
gc(reset = TRUE)

load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_RESULTS_ONLY.RData")

summaryMETRICS_training <- mat.or.vec(length(modelNAMES_training), 2)
colnames(summaryMETRICS_training) <- c("Accuracy", "Kappa")
row.names(summaryMETRICS_training) <- c(
  "Decision Tree",
  "Random Forest",
  "Regularized Random Forests",
  "Guided Regularized Random Forests",
  "Guided Random Forests",
  "AdaBoost.M1 (NOT WEAK)",
  "AdaBoost.M1 (WEAK CLASSIFIERS)",
  "LogitBoost (WEAK CLASSIFIERS)",
  "(Stochastic) Gradient Boosting using MULTINOMIAL Loss",
  "EXTREME GRADIENT BOOSTING - TREES",
  "EXTREME GRADIENT BOOSTING - TREES WITH DROPOUT"
)

for (i in 1:nrow(summaryMETRICS_training)) {
  summaryMETRICS_training[i, 1] = as.numeric(eval(parse(
    text = paste(modelNAMES_training[i], "$overall[1]", sep = "")
  ))) #Accuracy
  summaryMETRICS_training[i, 2] = as.numeric(eval(parse(
    text = paste(modelNAMES_training[i], "$overall[2]", sep = "")
  ))) #Kappa
}

summaryMETRICS_training[order(summaryMETRICS_training[, 2], decreasing = TRUE), ]

save.image("D:/activities/models_code_SAVED_allACTIVITIES_PCA_RESULTS_ONLY.RData")

####SUMMARY - VALIDATION####

rm(list = ls())

modelNAMESMODEL <- c(
  "cartMODEL",
  "rfMODEL",
  "rrfMODEL",
  "grrfMODEL",
  "grfMODEL",
  "adaMODEL",
  "adaWEAKMODEL",
  "logitWEAKMODEL",
  "gbmmultiMODEL",
  "xgbTreeMODEL",
  "xgbDARTMODEL"
)

summaryMETRICS_validation <- mat.or.vec(length(modelNAMESMODEL), 2)
colnames(summaryMETRICS_validation) <- c("Accuracy", "Kappa")
row.names(summaryMETRICS_validation) <- c(
  "Decision Tree",
  "Random Forest",
  "Regularized Random Forests",
  "Guided Regularized Random Forests",
  "Guided Random Forests",
  "AdaBoost.M1 (NOT WEAK)",
  "AdaBoost.M1 (WEAK CLASSIFIERS)",
  "LogitBoost (WEAK CLASSIFIERS)",
  "(Stochastic) Gradient Boosting using MULTINOMIAL Loss",
  "EXTREME GRADIENT BOOSTING - TREES",
  "EXTREME GRADIENT BOOSTING - TREES WITH DROPOUT"
)

load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_1_AtoC.RData")
for (i in 1:nrow(summaryMETRICS_validation)) {
  tryCatch({
    #this is to ignore errors like MODEL not found
    summaryMETRICS_validation[i, 1] = as.numeric(eval(parse(
      text = paste(
        modelNAMESMODEL[i],
        "[['results']][['Accuracy']][order(",
        modelNAMESMODEL[i],
        "[['results']][['Accuracy']],decreasing = TRUE)][1]",
        sep = ""
      )
    ))) #Accuracy
    summaryMETRICS_validation[i, 2] = as.numeric(eval(parse(
      text = paste(
        modelNAMESMODEL[i],
        "[['results']][['Kappa']][order(",
        modelNAMESMODEL[i],
        "[['results']][['Accuracy']],decreasing = TRUE)][1]",
        sep = ""
      )
    ))) #Kappa
  }, error = function(e) {
  }) #this is to ignore errors like MODEL not found
}
rm(list = ls()[!(
  ls() %in% c(
    summaryMETRICS_validation,
    "modelNAMESMODEL",
    "summaryMETRICS_validation"
  )
)])
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_2_D.RData")
for (i in 1:nrow(summaryMETRICS_validation)) {
  tryCatch({
    #this is to ignore errors like _MODEL not found
    summaryMETRICS_validation[i, 1] = as.numeric(eval(parse(
      text = paste(
        modelNAMESMODEL[i],
        "[['results']][['Accuracy']][order(",
        modelNAMESMODEL[i],
        "[['results']][['Accuracy']],decreasing = TRUE)][1]",
        sep = ""
      )
    ))) #Accuracy
    summaryMETRICS_validation[i, 2] = as.numeric(eval(parse(
      text = paste(
        modelNAMESMODEL[i],
        "[['results']][['Kappa']][order(",
        modelNAMESMODEL[i],
        "[['results']][['Accuracy']],decreasing = TRUE)][1]",
        sep = ""
      )
    ))) #Kappa
  }, error = function(e) {
  }) #this is to ignore errors like _MODEL not found
}
rm(list = ls()[!(
  ls() %in% c(
    summaryMETRICS_validation,
    "modelNAMESMODEL",
    "summaryMETRICS_validation"
  )
)])
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_3_EtoF.RData")
for (i in 1:nrow(summaryMETRICS_validation)) {
  tryCatch({
    #this is to ignore errors like _MODEL not found
    summaryMETRICS_validation[i, 1] = as.numeric(eval(parse(
      text = paste(
        modelNAMESMODEL[i],
        "[['results']][['Accuracy']][order(",
        modelNAMESMODEL[i],
        "[['results']][['Accuracy']],decreasing = TRUE)][1]",
        sep = ""
      )
    ))) #Accuracy
    summaryMETRICS_validation[i, 2] = as.numeric(eval(parse(
      text = paste(
        modelNAMESMODEL[i],
        "[['results']][['Kappa']][order(",
        modelNAMESMODEL[i],
        "[['results']][['Accuracy']],decreasing = TRUE)][1]",
        sep = ""
      )
    ))) #Kappa
  }, error = function(e) {
  }) #this is to ignore errors like _MODEL not found
}
rm(list = ls()[!(
  ls() %in% c(
    summaryMETRICS_validation,
    "modelNAMESMODEL",
    "summaryMETRICS_validation"
  )
)])
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_4_G.RData")
for (i in 1:nrow(summaryMETRICS_validation)) {
  tryCatch({
    #this is to ignore errors like _MODEL not found
    summaryMETRICS_validation[i, 1] = as.numeric(eval(parse(
      text = paste(
        modelNAMESMODEL[i],
        "[['results']][['Accuracy']][order(",
        modelNAMESMODEL[i],
        "[['results']][['Accuracy']],decreasing = TRUE)][1]",
        sep = ""
      )
    ))) #Accuracy
    summaryMETRICS_validation[i, 2] = as.numeric(eval(parse(
      text = paste(
        modelNAMESMODEL[i],
        "[['results']][['Kappa']][order(",
        modelNAMESMODEL[i],
        "[['results']][['Accuracy']],decreasing = TRUE)][1]",
        sep = ""
      )
    ))) #Kappa
  }, error = function(e) {
  }) #this is to ignore errors like _MODEL not found
}
rm(list = ls()[!(
  ls() %in% c(
    summaryMETRICS_validation,
    "modelNAMESMODEL",
    "summaryMETRICS_validation"
  )
)])
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_5_H.RData")
for (i in 1:nrow(summaryMETRICS_validation)) {
  tryCatch({
    #this is to ignore errors like _MODEL not found
    summaryMETRICS_validation[i, 1] = as.numeric(eval(parse(
      text = paste(
        modelNAMESMODEL[i],
        "[['results']][['Accuracy']][order(",
        modelNAMESMODEL[i],
        "[['results']][['Accuracy']],decreasing = TRUE)][1]",
        sep = ""
      )
    ))) #Accuracy
    summaryMETRICS_validation[i, 2] = as.numeric(eval(parse(
      text = paste(
        modelNAMESMODEL[i],
        "[['results']][['Kappa']][order(",
        modelNAMESMODEL[i],
        "[['results']][['Accuracy']],decreasing = TRUE)][1]",
        sep = ""
      )
    ))) #Kappa
  }, error = function(e) {
  }) #this is to ignore errors like _MODEL not found
}
rm(list = ls()[!(
  ls() %in% c(
    summaryMETRICS_validation,
    "modelNAMESMODEL",
    "summaryMETRICS_validation"
  )
)])
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_6_I.RData")
for (i in 1:nrow(summaryMETRICS_validation)) {
  tryCatch({
    #this is to ignore errors like _MODEL not found
    summaryMETRICS_validation[i, 1] = as.numeric(eval(parse(
      text = paste(
        modelNAMESMODEL[i],
        "[['results']][['Accuracy']][order(",
        modelNAMESMODEL[i],
        "[['results']][['Accuracy']],decreasing = TRUE)][1]",
        sep = ""
      )
    ))) #Accuracy
    summaryMETRICS_validation[i, 2] = as.numeric(eval(parse(
      text = paste(
        modelNAMESMODEL[i],
        "[['results']][['Kappa']][order(",
        modelNAMESMODEL[i],
        "[['results']][['Accuracy']],decreasing = TRUE)][1]",
        sep = ""
      )
    ))) #Kappa
  }, error = function(e) {
  }) #this is to ignore errors like _MODEL not found
}
rm(list = ls()[!(
  ls() %in% c(
    summaryMETRICS_validation,
    "modelNAMESMODEL",
    "summaryMETRICS_validation"
  )
)])
gc(reset = TRUE)

summaryMETRICS_validation[order(summaryMETRICS_validation[, 2], decreasing = TRUE), ]

load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_RESULTS_ONLY.RData")

save.image("D:/activities/models_code_SAVED_allACTIVITIES_PCA_RESULTS_ONLY.RData")
####MAJORITY VOTING (HARD)#####

rm(list = ls())
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_BASICS.RData")
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_RESULTS_ONLY.RData")

top <- seq(2, 10, 1) #taking all top models, MAXIMUM IS 10
majorityHARDsummaryMETRICS <- mat.or.vec(length(top), 3)
colnames(majorityHARDsummaryMETRICS) <- c("TOP", "Accuracy", "Kappa")
k <- 1

for (j in top) {
  topmodelhardPREDICT <-
    modelNAMESPREDICT[order(summaryMETRICS_validation[, 2], decreasing = TRUE)][1:j] #taking PREDICT of top models
  
  #creating the MAJORITY column
  tophardPREDICT <-
    do.call("cbind.data.frame", mget(topmodelhardPREDICT))
  MAJORITY <- integer(nrow(tophardPREDICT))
  
  for (i in 1:nrow(tophardPREDICT)) {
    #calculating the majority
    temp_table <- sort(table(t(tophardPREDICT[i, ])), decreasing = TRUE)
    if (temp_table[1] == temp_table[2] & !is.na(temp_table[2])) {
      MAJORITY[i] <-
        names(sort(table(t(
          tophardPREDICT[i, 1]
        )), decreasing = TRUE)[1]) #if there is a draw, choose the one of the best performing model
    } else {
      MAJORITY[i] <-
        names(sort(table(t(
          tophardPREDICT[i, ]
        )), decreasing = TRUE)[1]) #if there is NOT a draw, choose the majority one
    }
  }
  MAJORITY <- as.factor(MAJORITY)
  tophardPREDICT <- cbind(tophardPREDICT, MAJORITY)
  
  temp <- paste0("top", j, "majorityHARDCM")
  assign(temp,
         confusionMatrix(tophardPREDICT[, "MAJORITY"], testing_df$activity))
  
  majorityHARDsummaryMETRICS[k, 1] = j
  majorityHARDsummaryMETRICS[k, 2] = as.numeric(eval(parse(text = paste(
    temp, "$overall[1]", sep = ""
  )))) #Accuracy
  majorityHARDsummaryMETRICS[k, 3] = as.numeric(eval(parse(text = paste(
    temp, "$overall[2]", sep = ""
  )))) #Kappa
  k <- k + 1
}

majorityHARDsummaryMETRICS

library(tidyr)
majorityHARDsummaryMETRICS <-
  as.data.frame(majorityHARDsummaryMETRICS)
majorityHARDsummaryMETRICS_pivot <-
  majorityHARDsummaryMETRICS %>% pivot_longer(!TOP, names_to = "metric", values_to = "value")

save.image("D:/activities/models_code_SAVED_allACTIVITIES_PCA_RESULTS_ONLY.RData")

####MAJORITY VOTING (SOFT)#####

rm(list = ls())
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_BASICS.RData")
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_RESULTS_ONLY.RData")

top <- seq(2, 10, 1) #taking all top models, MAXIMUM IS 10
majoritySOFTsummaryMETRICS <- mat.or.vec(length(top), 3)
colnames(majoritySOFTsummaryMETRICS) <- c("TOP", "Accuracy", "Kappa")
k <- 1

for (j in top) {
  topmodelsoftPREDICT <-
    modelNAMESprobPREDICT[order(summaryMETRICS_validation[, 2], decreasing = TRUE)][1:j] #taking PREDICT of top models
  
  #creating the AVERAGEPROB column
  topsoftPREDICT <-
    mget(topmodelsoftPREDICT) #grabbing the probabilities of activities for all models into a list
  
  temp <- as.data.frame(topsoftPREDICT[1])
  for (i in 2:j) {
    temp <- temp + as.data.frame(topsoftPREDICT[j])
  }
  AVERAGEPROBS <- temp / j
  names(AVERAGEPROBS) <-
    names(topsoftPREDICT[[1]]) #any one would suffice
  MAJORITYAVERAGE <- integer(nrow(AVERAGEPROBS))
  
  for (i in 1:nrow(AVERAGEPROBS)) {
    MAJORITYAVERAGE[i] <- names(which.max(AVERAGEPROBS[i, ]))
  }
  
  MAJORITYAVERAGE <- as.factor(MAJORITYAVERAGE)
  
  temp2 <- paste0("top", j, "majoritySOFTCM")
  assign(temp2, confusionMatrix(MAJORITYAVERAGE, as.factor(testing_df$activity)))
  
  majoritySOFTsummaryMETRICS[k, 1] = j
  majoritySOFTsummaryMETRICS[k, 2] = as.numeric(eval(parse(text = paste(
    temp2, "$overall[1]", sep = ""
  )))) #Accuracy
  majoritySOFTsummaryMETRICS[k, 3] = as.numeric(eval(parse(text = paste(
    temp2, "$overall[2]", sep = ""
  )))) #Kappa
  k <- k + 1
}

majoritySOFTsummaryMETRICS

library(tidyr)
majoritySOFTsummaryMETRICS <-
  as.data.frame(majoritySOFTsummaryMETRICS)
majoritySOFTsummaryMETRICS_pivot <-
  majoritySOFTsummaryMETRICS %>% pivot_longer(!TOP, names_to = "metric", values_to = "value")

save.image("D:/activities/models_code_SAVED_allACTIVITIES_PCA_RESULTS_ONLY.RData")

####Printing k-fold cross-validation results####

rm(list = ls())

library(writexl)

modelNAMESMODEL <- c(
  "cartMODEL",
  "rfMODEL",
  "rrfMODEL",
  "grrfMODEL",
  "grfMODEL",
  "adaMODEL",
  ##"adaWEAKMODEL", skipped
  "logitWEAKMODEL",
  "gbmmultiMODEL",
  "xgbTreeMODEL",
  "xgbDARTMODEL",
  "xgbDARTMODEL_min"
)

load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_1_AtoC.RData")
for (i in 1:length(modelNAMESMODEL)) {
  tryCatch({
    #this is to ignore errors like MODEL not found
    tempresults <-
      as.data.frame(eval(parse(
        text = paste(modelNAMESMODEL[i], "$results", sep = "")
      )))
    temppath <-
      paste(
        "D:/activities/allsensors_cross_val_results/",
        modelNAMESMODEL[i],
        "_results.xlsx",
        sep = ""
      )
    write_xlsx(tempresults, temppath)
  }, error = function(e) {
  }) #this is to ignore errors like MODEL not found
}
rm(list = ls()[!(ls() %in% c("modelNAMESMODEL"))])
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_2_D.RData")
for (i in 1:length(modelNAMESMODEL)) {
  tryCatch({
    #this is to ignore errors like MODEL not found
    tempresults <-
      as.data.frame(eval(parse(
        text = paste(modelNAMESMODEL[i], "$results", sep = "")
      )))
    temppath <-
      paste(
        "D:/activities/allsensors_cross_val_results/",
        modelNAMESMODEL[i],
        "_results.xlsx",
        sep = ""
      )
    write_xlsx(tempresults, temppath)
  }, error = function(e) {
  }) #this is to ignore errors like MODEL not found
}
rm(list = ls()[!(ls() %in% c("modelNAMESMODEL"))])
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_3_EtoF.RData")
for (i in 1:length(modelNAMESMODEL)) {
  tryCatch({
    #this is to ignore errors like MODEL not found
    tempresults <-
      as.data.frame(eval(parse(
        text = paste(modelNAMESMODEL[i], "$results", sep = "")
      )))
    temppath <-
      paste(
        "D:/activities/allsensors_cross_val_results/",
        modelNAMESMODEL[i],
        "_results.xlsx",
        sep = ""
      )
    write_xlsx(tempresults, temppath)
  }, error = function(e) {
  }) #this is to ignore errors like MODEL not found
}
rm(list = ls()[!(ls() %in% c("modelNAMESMODEL"))])
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_4_G.RData")
for (i in 1:length(modelNAMESMODEL)) {
  tryCatch({
    #this is to ignore errors like MODEL not found
    tempresults <-
      as.data.frame(eval(parse(
        text = paste(modelNAMESMODEL[i], "$results", sep = "")
      )))
    temppath <-
      paste(
        "D:/activities/allsensors_cross_val_results/",
        modelNAMESMODEL[i],
        "_results.xlsx",
        sep = ""
      )
    write_xlsx(tempresults, temppath)
  }, error = function(e) {
  }) #this is to ignore errors like MODEL not found
}
rm(list = ls()[!(ls() %in% c("modelNAMESMODEL"))])
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_5_H.RData")
for (i in 1:length(modelNAMESMODEL)) {
  tryCatch({
    #this is to ignore errors like MODEL not found
    tempresults <-
      as.data.frame(eval(parse(
        text = paste(modelNAMESMODEL[i], "$results", sep = "")
      )))
    temppath <-
      paste(
        "D:/activities/allsensors_cross_val_results/",
        modelNAMESMODEL[i],
        "_results.xlsx",
        sep = ""
      )
    write_xlsx(tempresults, temppath)
  }, error = function(e) {
  }) #this is to ignore errors like MODEL not found
}
rm(list = ls()[!(ls() %in% c("modelNAMESMODEL"))])
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_6_I.RData")
for (i in 1:length(modelNAMESMODEL)) {
  tryCatch({
    #this is to ignore errors like MODEL not found
    tempresults <-
      as.data.frame(eval(parse(
        text = paste(modelNAMESMODEL[i], "$results", sep = "")
      )))
    temppath <-
      paste(
        "D:/activities/allsensors_cross_val_results/",
        modelNAMESMODEL[i],
        "_results.xlsx",
        sep = ""
      )
    write_xlsx(tempresults, temppath)
  }, error = function(e) {
  }) #this is to ignore errors like MODEL not found
}
rm(list = ls()[!(ls() %in% c("modelNAMESMODEL"))])
gc(reset = TRUE)

#writing prediction error of random forest
load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_1_AtoC.RData")
rfMODEL_errorrate <-
  as.data.frame(cbind(1:500, rfMODEL[["finalModel"]][["err.rate"]][, 1]))
colnames(rfMODEL_errorrate) <- c("iter", "error_rate")
write_xlsx(
  rfMODEL_errorrate,
  "D:/activities/allsensors_cross_val_results/rfMODEL_errorrate.xlsx"
)
rm(list = ls())
gc(reset = TRUE)


####Writing confusion matrices####

rm(list = ls())

library(writexl)

modelNAMESCM <- c(
  "cartCM",
  "rfCM",
  "rrfCM",
  "grrfCM",
  "grfCM",
  "adaCM",
  ##"adaWEAKCM", skipped
  "logitWEAKCM",
  "gbmmultiCM",
  "xgbTreeCM",
  "xgbDARTCM",
  "xgbDARTCM_min"
)

load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_RESULTS_ONLY.RData")
for (i in 1:length(modelNAMESCM)) {
  tryCatch({
    #this is to ignore errors like MODEL not found
    tempresults <-
      eval(parse(text = paste(modelNAMESCM[i], "$table", sep = "")))
    tempresults <- t(tempresults[c(1, 12:19, 2:11), c(1, 12:19, 2:11)])
    correct <- mat.or.vec(19, 1)
    for (j in 1:19) {
      correct[j] <- round((tempresults[j, j] / sum(tempresults[j, ])) * 100, 2)
    }
    tempresults <- cbind(tempresults, correct)
    tempresults[c(1:19), c(1:19)] <- round(tempresults[c(1:19), c(1:19)], 0)
    
    temppath <-
      paste("D:/activities/allsensors_confusion_matrices/",
            modelNAMESCM[i],
            ".csv",
            sep = "")
    write.csv(tempresults, temppath)
  }, error = function(e) {
  }) #this is to ignore errors like MODEL not found
}
gc(reset = TRUE)

####Writing one-vs-all####

rm(list = ls())

library(writexl)

modelNAMESCM <- c(
  "cartCM",
  "rfCM",
  "rrfCM",
  "grrfCM",
  "grfCM",
  "adaCM",
  ##"adaWEAKCM", skipped
  "logitWEAKCM",
  "gbmmultiCM",
  "xgbTreeCM",
  "xgbDARTCM",
  "xgbDARTCM_min"
)

load("D:/activities/models_code_SAVED_allACTIVITIES_PCA_RESULTS_ONLY.RData")
for (i in 1:length(modelNAMESCM)) {
  tempresults <-
    eval(parse(text = paste(modelNAMESCM[i], "$byClass", sep = "")))
  tempresults <- tempresults[c(1, 12:19, 2:11), c(1, 2, 5, 7)]
  tempresults <- round(tempresults, 3)
  rownames(tempresults) <-
    c(
      "A1",
      "A2",
      "A3",
      "A4",
      "A5",
      "A6",
      "A7",
      "A8",
      "A9",
      "A10",
      "A11",
      "A12",
      "A13",
      "A14",
      "A15",
      "A16",
      "A17",
      "A18",
      "A19"
    )
  temppath <-
    paste("D:/activities/allsensors_one-vs-all/",
          modelNAMESCM[i],
          "_one-vs-all.csv",
          sep = "")
  write.csv(t(tempresults), temppath)
}
gc(reset = TRUE)
