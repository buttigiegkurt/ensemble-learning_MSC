rm(list = ls())
library(readxl)
library(writexl)
library(caret)
library(caretEnsemble)
library(rpart)
library(rpart.plot)
library(randomForest)
library(RRF)
library(ada)
library(caTools)
library(h2o)
library(xgboost)

##the below were used for parallel processing:
#library(doParallel)
#cl <- makePSOCKcluster(4)
#registerDoParallel(cl)
##and when ready:
#stopCluster(cl)

df_raw <- read_excel("D:/offside/dataset_final_balanced.xlsx")
names(df_raw) = gsub(x = names(df_raw),
                     pattern = "\\/",
                     replacement = ".")
df_raw <- as.data.frame(df_raw)

df_raw$offside <- as.factor(df_raw$offside)
levels(df_raw$offside) <- c("No", "Yes")
df_raw$camera_angle <- as.factor(df_raw$camera_angle)
df <- df_raw
df[, -c(1, 2, 383)] <-
  predict(preProcess(df_raw[, -c(1, 2, 383)], method = c("center", "scale")), df_raw[, -c(1, 2, 383)])

training_df <- df[(df$split == 1), ] #considering training set only
testing_df <- df[(df$split == 0), ] #considering testing set only
training_df$split <- NULL
testing_df$split <- NULL
rm(df_raw)
rm(df)

#will keep the same resampling method for all models
resamplingINDEX <- createFolds(training_df$offside, k = 3)
resamplingCTRL <- trainControl(
  method = "cv",
  number = 3,
  verboseIter = TRUE,
  savePredictions = "final",
  classProbs = TRUE,
  index = resamplingINDEX
)

save.image("D:/offside/models_code_SAVED_FINAL_vXX_BASICS.RData")

#####A - CART (INCLUDES VARIMP)####

rm(list = ls())
load("D:/offside/models_code_SAVED_FINAL_vXX_BASICS.RData")

cartGRID <- expand.grid(
  cp = seq(0, 0.1, 0.005),
  #complexity parameter
  maxdepth = seq(1, 7, 1),
  minbucket = c(6, 9, 12, 16)
)

results <- NULL

for (i in 1:nrow(cartGRID)) {
  mdValue <- expand.grid(maxdepth = cartGRID[i, 2])
  tempcartMODEL <- train(
    offside ~ .,
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
#optimal cp is 0, 0.005, 0.01 or 0.015
#optimal maxdepth is 4, 5, 6 or 7
#optimal minbucket is 12

cartMODEL <- train(
  offside ~ .,
  data = training_df,
  method = "rpart2",
  metric = "Kappa",
  trControl = trainControl(method = "none"),
  tuneGrid = expand.grid(maxdepth = 7),
  control = rpart.control(
    cp = 0.015,
    maxdepth = 7,
    minbucket = 12,
    xval = 0
  )
)
cartMODEL[["results"]] <- results
rm(tempcartMODEL, mdValue, results)

cartPREDICT <- predict(cartMODEL, testing_df)
cartprobPREDICT <- predict(cartMODEL, testing_df, type = "prob")
cartCM <-
  confusionMatrix(cartPREDICT, testing_df$offside, positive = 'Yes')

cartPREDICT_training <- predict(cartMODEL, training_df)
cartprobPREDICT_training <-
  predict(cartMODEL, training_df, type = "prob")
cartCM_training <-
  confusionMatrix(cartPREDICT_training, training_df$offside, positive = 'Yes')

rpart.plot(cartMODEL$finalModel)
cartVARIMP <- varImp(cartMODEL$finalModel)

save.image("D:/offside/models_code_SAVED_FINAL_vXX_1_AtoC.RData")

#####B - Random Forest (INCLUDES VARIMP)####

#CREATING CUSTOM CARET MODEL TO ADD nodesize
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
    
    out <-
      as.data.frame(varImp, stringsAsFactors = TRUE)
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
  expand.grid(mtry = c(seq(5, 19, 1), seq(20, 379, 15), 379),
              #number of variables randomly sampled as candidates at each split
              nodesize = c(1, 3, 6, 9, 12)) #default is 1
#default is mtry=floor(ncol(training_df)/3)=127
#maximum mtry is ncol(training_df)-1=379
#mtry is p_tk in the random forest algorithm in the dissertation, and is kept constant across all nodes across all trees

rfMODEL <- train(
  offside ~ .,
  data = training_df,
  method = rfcustom,
  metric = "Kappa",
  trControl = resamplingCTRL,
  tuneGrid = rfGRID,
  importance = TRUE
)

rfPREDICT <- predict(rfMODEL, testing_df)
rfprobPREDICT <- predict(rfMODEL, testing_df, type = "prob")
rfCM <- confusionMatrix(rfPREDICT, testing_df$offside, positive = 'Yes')

rfPREDICT_training <- predict(rfMODEL, training_df)
rfprobPREDICT_training <- predict(rfMODEL, training_df, type = "prob")
rfCM_training <-
  confusionMatrix(rfPREDICT_training, training_df$offside, positive = 'Yes')

#variable importance
rfVARIMP <- as.matrix(varImp(rfMODEL$finalModel))

#plotting the errors
plot(rfMODEL$finalModel, main = "Prediction Error of the Random Forest")
#Red line represents classification error rate of the NON-OFFSIDE class,
#green line represents classification error rate of the OFFSIDE class
#and black line represents overall error rate.

save.image("D:/offside/models_code_SAVED_FINAL_vXX_1_AtoC.RData")

#####C - RRF and GRRF and GRF (NO VARIMP)####

#TO AMEND CARET MODEL FUNCTION TO GRAB IMPORTANCE FROM PREVIOUSLY TRAINED RANDOM FOREST
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
    args <-
      list(x = x,
           y = y,
           mtry = min(param$mtry, ncol(x)))
    args <- c(args, theDots)
    #firstFit <- do.call(randomForest::randomForest, args) #REMOVED
    #firstImp <- randomForest:::importance(firstFit) #REMOVED
    impRF <-
      randomForest:::importance(rfMODEL$finalModel) #ADDED
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
      varImp <-
        data.frame(Overall = varImp[, "%IncMSE"])
    else {
      retainNames <- levels(object$y)
      if (all(retainNames %in% colnames(varImp))) {
        varImp <- varImp[, retainNames]
      } else {
        varImp <- data.frame(Overall = varImp[, 1])
      }
    }
    out <-
      as.data.frame(varImp, stringsAsFactors = TRUE)
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
  mtry = c(seq(5, 19, 1), seq(20, 379, 15), 379),
  coefReg = seq(0.1, 1, 0.1),
  #lambda in grrf2.pdf
  coefImp = 0
) #gamma in grrf2.pdf
rrfMODEL <- train(
  offside ~ .,
  data = training_df,
  method = RRFcustom,
  metric = "Kappa",
  trControl = resamplingCTRL,
  tuneGrid = rrfGRID,
  flagReg = 1
)

rrfPREDICT <- predict(rrfMODEL, testing_df)
rrfprobPREDICT <- predict(rrfMODEL, testing_df, type = "prob")
rrfCM <- confusionMatrix(rrfPREDICT, testing_df$offside, positive = 'Yes')

rrfPREDICT_training <- predict(rrfMODEL, training_df)
rrfprobPREDICT_training <- predict(rrfMODEL, training_df, type = "prob")
rrfCM_training <-
  confusionMatrix(rrfPREDICT_training, training_df$offside, positive = 'Yes')

#guided regularized random forest

grrfGRID <- expand.grid(
  mtry = c(seq(5, 19, 1), seq(20, 379, 15), 379),
  coefReg = seq(0, 1, 0.1),
  coefImp = seq(0, 1, 0.1)
)
grrfGRID <-
  grrfGRID[-which(grrfGRID[, 2] == 0 &
                    grrfGRID[, 3] == 0), ] #excluding both 0

grrfMODEL <- train(
  offside ~ .,
  data = training_df,
  method = RRFcustom,
  metric = "Kappa",
  trControl = resamplingCTRL,
  tuneGrid = grrfGRID,
  flagReg = 1
)

grrfPREDICT <- predict(grrfMODEL, testing_df)
grrfprobPREDICT <- predict(grrfMODEL, testing_df, type = "prob")
grrfCM <-
  confusionMatrix(grrfPREDICT, testing_df$offside, positive = 'Yes')

grrfPREDICT_training <- predict(grrfMODEL, training_df)
grrfprobPREDICT_training <-
  predict(grrfMODEL, training_df, type = "prob")
grrfCM_training <-
  confusionMatrix(grrfPREDICT_training, training_df$offside, positive = 'Yes')

#guided random forest
grfGRID <- expand.grid(
  mtry = c(seq(5, 19, 1), seq(20, 379, 15), 379),
  coefReg = 1,
  coefImp = seq(0, 1, 0.1)
)
grfMODEL <- train(
  offside ~ .,
  data = training_df,
  method = RRFcustom,
  metric = "Kappa",
  trControl = resamplingCTRL,
  tuneGrid = grfGRID,
  flagReg = 0
)

grfPREDICT <- predict(grfMODEL, testing_df)
grfprobPREDICT <- predict(grfMODEL, testing_df, type = "prob")
grfCM <- confusionMatrix(grfPREDICT, testing_df$offside, positive = 'Yes')

grfPREDICT_training <- predict(grfMODEL, training_df)
grfprobPREDICT_training <- predict(grfMODEL, training_df, type = "prob")
grfCM_training <-
  confusionMatrix(grfPREDICT_training, training_df$offside, positive = 'Yes')

rrfVARIMP <- rrfMODEL[["finalModel"]][["importance"]]
grrfVARIMP <- grrfMODEL[["finalModel"]][["importance"]]
grfVARIMP <- grfMODEL[["finalModel"]][["importance"]]

save.image("D:/offside/models_code_SAVED_FINAL_vXX_1_AtoC.RData")

#####D - AdaBoost (INCLUDES VARIMP)####

rm(list = ls())
gc(reset = TRUE)
load("D:/offside/models_code_SAVED_FINAL_vXX_BASICS.RData")

#CREATING CUSTOM CARET MODEL TO ADD bag.frac
adacustom <- list(
  label = "CUSTOM Boosted Classification Trees",
  library = c("ada", "plyr"),
  loop = function(grid) {
    loop <- plyr::ddply(grid, c("nu", "maxdepth", "bag.frac"), #AMENDED
                        function(x)
                          c(iter = max(x$iter)))
    submodels <-
      vector(mode = "list", length = nrow(loop))
    for (i in seq(along = loop$iter)) {
      index <- which(
        grid$maxdepth == loop$maxdepth[i] &
          grid$nu == loop$nu[i] &
          grid$bag.frac == loop$bag.frac[i]
      ) #ADDED
      trees <- grid[index, "iter"]
      submodels[[i]] <-
        data.frame(iter = trees[trees != loop$iter[i]])
    }
    list(loop = loop, submodels = submodels)
  },
  type = c("Classification"),
  parameters = data.frame(
    parameter = c('iter', 'maxdepth', 'nu', 'bag.frac'),
    #AMENDED
    class = rep("numeric", 4),
    #AMENDED
    label = c(
      '#Trees',
      'Max Tree Depth',
      'Learning Rate',
      'Subsampling Parameter for Stochastic'
    )
  ),
  grid = function(x, y, len = NULL, search = "grid") {
    if (search == "grid") {
      out = expand.grid(
        iter = floor((1:len) * 50),
        maxdepth = seq(1, len),
        nu = .1,
        bag.frac = .5
      ) #ADDED
    } else {
      out <- data.frame(
        iter = sample(1:1000, replace = TRUE, size = len),
        maxdepth = sample(1:10, replace = TRUE, size = len),
        nu = runif(len, min = .001, max = .5),
        bag.frac = runif(len, min = .1, max = 1)
      ) #ADDED
    }
    out
  },
  fit = function(x, y, wts, param, lev, last, classProbs, ...) {
    theDots <- list(...)
    
    if (any(names(theDots) == "control")) {
      theDots$control$maxdepth <- param$maxdepth
      ctl <- theDots$control
      theDots$control <- NULL
      
    } else
      ctl <- rpart::rpart.control(
        maxdepth = param$maxdepth,
        cp = -1,
        minsplit = 0,
        xval = 0
      )
    
    modelArgs <- c(
      list(
        x = x,
        y = y,
        iter = param$iter,
        nu = param$nu,
        bag.frac = param$bag.frac,
        #ADDED
        control = ctl
      ),
      theDots
    )
    out <- do.call(ada::ada, modelArgs)
    out
  },
  predict = function(modelFit, newdata, submodels = NULL) {
    if (!is.data.frame(newdata))
      newdata <- as.data.frame(newdata, stringsAsFactors = TRUE)
    out <-
      predict(modelFit, newdata, n.iter = modelFit$tuneValue$iter)
    
    if (!is.null(submodels)) {
      tmp <- vector(mode = "list", length = length(submodels$iter) + 1)
      tmp[[1]] <- out
      for (i in seq(along = submodels$iter)) {
        tmp[[i + 1]] <-
          predict(modelFit, newdata, n.iter = submodels$iter[[i]])
      }
      out <- lapply(tmp, as.character)
    }
    out
  },
  prob = function(modelFit, newdata, submodels = NULL) {
    if (!is.data.frame(newdata))
      newdata <- as.data.frame(newdata, stringsAsFactors = TRUE)
    out <- predict(modelFit,
                   newdata,
                   type = "prob",
                   n.iter = modelFit$tuneValue$iter)
    colnames(out) <- modelFit$obsLevels
    
    if (!is.null(submodels)) {
      tmp <- vector(mode = "list", length = length(submodels$iter) + 1)
      tmp[[1]] <- out
      for (i in seq(along = submodels$iter)) {
        tmp[[i + 1]] <- predict(modelFit,
                                newdata,
                                type = "prob",
                                n.iter = submodels$iter[[i]])
        colnames(tmp[[i + 1]]) <- modelFit$obsLevels
      }
      out <- lapply(tmp, as.data.frame)
    }
    out
  },
  levels = function(x)
    x$obsLevels,
  tags = c(
    "Tree-Based Model",
    "Ensemble Model",
    "Boosting",
    "Implicit Feature Selection",
    "Two Class Only",
    "Handle Missing Predictor Data"
  ),
  sort = function(x)
    x[order(x$iter, x$maxdepth, x$nu, x$bag.frac), ]
) #AMENDED

adaGRID <- expand.grid(
  maxdepth = c(3, 5, 7, 9),
  iter = seq(100, 1000, 100),
  nu = c(0.25, 0.5, 0.75, 1),
  #shrinkage
  bag.frac = c(0.25, 0.5, 0.75, 1)
) #percentage of training data to be sampled, <1 implies stochastic Gradient Boosting

adaMODEL <- train(
  offside ~ .,
  data = training_df,
  loss = "exponential",
  type = "discrete",
  method = adacustom,
  metric = "Kappa",
  trControl = resamplingCTRL,
  tuneGrid = adaGRID,
  control = rpart.control(minsplit = 5, cp = 0)
)

adaPREDICT <- predict(adaMODEL, testing_df)
adaprobPREDICT <- predict(adaMODEL, testing_df, type = "prob")
adaCM <- confusionMatrix(adaPREDICT, testing_df$offside, positive = 'Yes')

adaPREDICT_training <- predict(adaMODEL, training_df)
adaprobPREDICT_training <- predict(adaMODEL, training_df, type = "prob")
adaCM_training <-
  confusionMatrix(adaPREDICT_training, training_df$offside, positive = 'Yes')

adaVARIMP <-
  varplot(
    adaMODEL$finalModel,
    type = c("scores"),
    max.var.show = 379,
    plot.it = FALSE
  )
varplot(adaMODEL$finalModel, max.var.show = 20) #plotting top 20 variables

save.image("D:/offside/models_code_SAVED_FINAL_vXX_2_DtoE.RData")

#####E - Gentle AdaBoost (INCLUDES VARIMP)####

gentleadaGRID <- adaGRID

gentleadaMODEL <- train(
  offside ~ .,
  data = training_df,
  loss = "exponential",
  type = "gentle",
  method = adacustom,
  metric = "Kappa",
  trControl = resamplingCTRL,
  tuneGrid = gentleadaGRID,
  control = rpart.control(minsplit = 5, cp = 0)
)

gentleadaPREDICT <- predict(gentleadaMODEL, testing_df)
gentleadaprobPREDICT <-
  predict(gentleadaMODEL, testing_df, type = "prob")
gentleadaCM <-
  confusionMatrix(gentleadaPREDICT, testing_df$offside, positive = 'Yes')

gentleadaPREDICT_training <- predict(gentleadaMODEL, training_df)
gentleadaprobPREDICT_training <-
  predict(gentleadaMODEL, training_df, type = "prob")
gentleadaCM_training <-
  confusionMatrix(gentleadaPREDICT_training, training_df$offside, positive =
                    'Yes')

gentleadaVARIMP <-
  varplot(
    gentleadaMODEL$finalModel,
    type = c("scores"),
    max.var.show = 379,
    plot.it = FALSE
  )
varplot(gentleadaMODEL$finalModel, max.var.show = 20) #plotting top 20 variables

save.image("D:/offside/models_code_SAVED_FINAL_vXX_2_DtoE.RData")

#####F - Real AdaBoost (INCLUDES VARIMP)####

rm(list = ls())
gc(reset = TRUE)
load("D:/offside/models_code_SAVED_FINAL_vXX_BASICS.RData")

adacustom <- list(
  label = "CUSTOM Boosted Classification Trees",
  library = c("ada", "plyr"),
  loop = function(grid) {
    loop <- plyr::ddply(grid, c("nu", "maxdepth", "bag.frac"), #AMENDED
                        function(x)
                          c(iter = max(x$iter)))
    submodels <-
      vector(mode = "list", length = nrow(loop))
    for (i in seq(along = loop$iter)) {
      index <- which(
        grid$maxdepth == loop$maxdepth[i] &
          grid$nu == loop$nu[i] &
          grid$bag.frac == loop$bag.frac[i]
      ) #ADDED
      trees <- grid[index, "iter"]
      submodels[[i]] <-
        data.frame(iter = trees[trees != loop$iter[i]])
    }
    list(loop = loop, submodels = submodels)
  },
  type = c("Classification"),
  parameters = data.frame(
    parameter = c('iter', 'maxdepth', 'nu', 'bag.frac'),
    #AMENDED
    class = rep("numeric", 4),
    #AMENDED
    label = c(
      '#Trees',
      'Max Tree Depth',
      'Learning Rate',
      'Subsampling Parameter for Stochastic'
    )
  ),
  grid = function(x, y, len = NULL, search = "grid") {
    if (search == "grid") {
      out = expand.grid(
        iter = floor((1:len) * 50),
        maxdepth = seq(1, len),
        nu = .1,
        bag.frac = .5
      ) #ADDED
    } else {
      out <- data.frame(
        iter = sample(1:1000, replace = TRUE, size = len),
        maxdepth = sample(1:10, replace = TRUE, size = len),
        nu = runif(len, min = .001, max = .5),
        bag.frac = runif(len, min = .1, max = 1)
      ) #ADDED
    }
    out
  },
  fit = function(x, y, wts, param, lev, last, classProbs, ...) {
    theDots <- list(...)
    
    if (any(names(theDots) == "control")) {
      theDots$control$maxdepth <- param$maxdepth
      ctl <- theDots$control
      theDots$control <- NULL
      
    } else
      ctl <- rpart::rpart.control(
        maxdepth = param$maxdepth,
        cp = -1,
        minsplit = 0,
        xval = 0
      )
    
    modelArgs <- c(
      list(
        x = x,
        y = y,
        iter = param$iter,
        nu = param$nu,
        bag.frac = param$bag.frac,
        #ADDED
        control = ctl
      ),
      theDots
    )
    out <- do.call(ada::ada, modelArgs)
    out
  },
  predict = function(modelFit, newdata, submodels = NULL) {
    if (!is.data.frame(newdata))
      newdata <- as.data.frame(newdata, stringsAsFactors = TRUE)
    out <-
      predict(modelFit, newdata, n.iter = modelFit$tuneValue$iter)
    
    if (!is.null(submodels)) {
      tmp <- vector(mode = "list", length = length(submodels$iter) + 1)
      tmp[[1]] <- out
      for (i in seq(along = submodels$iter)) {
        tmp[[i + 1]] <-
          predict(modelFit, newdata, n.iter = submodels$iter[[i]])
      }
      out <- lapply(tmp, as.character)
    }
    out
  },
  prob = function(modelFit, newdata, submodels = NULL) {
    if (!is.data.frame(newdata))
      newdata <- as.data.frame(newdata, stringsAsFactors = TRUE)
    out <- predict(modelFit,
                   newdata,
                   type = "prob",
                   n.iter = modelFit$tuneValue$iter)
    colnames(out) <- modelFit$obsLevels
    
    if (!is.null(submodels)) {
      tmp <- vector(mode = "list", length = length(submodels$iter) + 1)
      tmp[[1]] <- out
      for (i in seq(along = submodels$iter)) {
        tmp[[i + 1]] <- predict(modelFit,
                                newdata,
                                type = "prob",
                                n.iter = submodels$iter[[i]])
        colnames(tmp[[i + 1]]) <- modelFit$obsLevels
      }
      out <- lapply(tmp, as.data.frame)
    }
    out
  },
  levels = function(x)
    x$obsLevels,
  tags = c(
    "Tree-Based Model",
    "Ensemble Model",
    "Boosting",
    "Implicit Feature Selection",
    "Two Class Only",
    "Handle Missing Predictor Data"
  ),
  sort = function(x)
    x[order(x$iter, x$maxdepth, x$nu, x$bag.frac), ]
) #AMENDED

realadaGRID <- expand.grid(
  maxdepth = c(3, 5, 7, 9),
  iter = seq(100, 1000, 100),
  nu = c(0.25, 0.5, 0.75, 1),
  #shrinkage
  bag.frac = c(0.25, 0.5, 0.75, 1)
) #percentage of training data to be sampled, <1 implies stochastic Gradient Boosting

realadaMODEL <- train(
  offside ~ .,
  data = training_df,
  loss = "exponential",
  type = "real",
  method = adacustom,
  metric = "Kappa",
  trControl = resamplingCTRL,
  tuneGrid = realadaGRID,
  control = rpart.control(minsplit = 5, cp = 0)
)

realadaPREDICT <- predict(realadaMODEL, testing_df)
realadaprobPREDICT <- predict(realadaMODEL, testing_df, type = "prob")
realadaCM <-
  confusionMatrix(realadaPREDICT, testing_df$offside, positive = 'Yes')

realadaPREDICT_training <- predict(realadaMODEL, training_df)
realadaprobPREDICT_training <-
  predict(realadaMODEL, training_df, type = "prob")
realadaCM_training <-
  confusionMatrix(realadaPREDICT_training, training_df$offside, positive =
                    'Yes')

realadaVARIMP <-
  varplot(
    realadaMODEL$finalModel,
    type = c("scores"),
    max.var.show = 379,
    plot.it = FALSE
  )
varplot(realadaMODEL$finalModel, max.var.show = 20) #plotting top 20 variables

save.image("D:/offside/models_code_SAVED_FINAL_vXX_3_F.RData")

#####G - AdaBoost - WEAK CLASSIFIERS (INCLUDES VARIMP)####

rm(list = ls())
gc(reset = TRUE)
load("D:/offside/models_code_SAVED_FINAL_vXX_BASICS.RData")

#CREATING CUSTOM CARET MODEL TO ADD bag.frac
adacustom <- list(
  label = "CUSTOM Boosted Classification Trees",
  library = c("ada", "plyr"),
  loop = function(grid) {
    loop <- plyr::ddply(grid, c("nu", "maxdepth", "bag.frac"), #AMENDED
                        function(x)
                          c(iter = max(x$iter)))
    submodels <-
      vector(mode = "list", length = nrow(loop))
    for (i in seq(along = loop$iter)) {
      index <- which(
        grid$maxdepth == loop$maxdepth[i] &
          grid$nu == loop$nu[i] &
          grid$bag.frac == loop$bag.frac[i]
      ) #ADDED
      trees <- grid[index, "iter"]
      submodels[[i]] <-
        data.frame(iter = trees[trees != loop$iter[i]])
    }
    list(loop = loop, submodels = submodels)
  },
  type = c("Classification"),
  parameters = data.frame(
    parameter = c('iter', 'maxdepth', 'nu', 'bag.frac'),
    #AMENDED
    class = rep("numeric", 4),
    #AMENDED
    label = c(
      '#Trees',
      'Max Tree Depth',
      'Learning Rate',
      'Subsampling Parameter for Stochastic'
    )
  ),
  grid = function(x, y, len = NULL, search = "grid") {
    if (search == "grid") {
      out = expand.grid(
        iter = floor((1:len) * 50),
        maxdepth = seq(1, len),
        nu = .1,
        bag.frac = .5
      ) #ADDED
    } else {
      out <- data.frame(
        iter = sample(1:1000, replace = TRUE, size = len),
        maxdepth = sample(1:10, replace = TRUE, size = len),
        nu = runif(len, min = .001, max = .5),
        bag.frac = runif(len, min = .1, max = 1)
      ) #ADDED
    }
    out
  },
  fit = function(x, y, wts, param, lev, last, classProbs, ...) {
    theDots <- list(...)
    
    if (any(names(theDots) == "control")) {
      theDots$control$maxdepth <- param$maxdepth
      ctl <- theDots$control
      theDots$control <- NULL
      
    } else
      ctl <- rpart::rpart.control(
        maxdepth = param$maxdepth,
        cp = -1,
        minsplit = 0,
        xval = 0
      )
    
    modelArgs <- c(
      list(
        x = x,
        y = y,
        iter = param$iter,
        nu = param$nu,
        bag.frac = param$bag.frac,
        #ADDED
        control = ctl
      ),
      theDots
    )
    out <- do.call(ada::ada, modelArgs)
    out
  },
  predict = function(modelFit, newdata, submodels = NULL) {
    if (!is.data.frame(newdata))
      newdata <- as.data.frame(newdata, stringsAsFactors = TRUE)
    out <-
      predict(modelFit, newdata, n.iter = modelFit$tuneValue$iter)
    
    if (!is.null(submodels)) {
      tmp <- vector(mode = "list", length = length(submodels$iter) + 1)
      tmp[[1]] <- out
      for (i in seq(along = submodels$iter)) {
        tmp[[i + 1]] <-
          predict(modelFit, newdata, n.iter = submodels$iter[[i]])
      }
      out <- lapply(tmp, as.character)
    }
    out
  },
  prob = function(modelFit, newdata, submodels = NULL) {
    if (!is.data.frame(newdata))
      newdata <- as.data.frame(newdata, stringsAsFactors = TRUE)
    out <- predict(modelFit,
                   newdata,
                   type = "prob",
                   n.iter = modelFit$tuneValue$iter)
    colnames(out) <- modelFit$obsLevels
    
    if (!is.null(submodels)) {
      tmp <- vector(mode = "list", length = length(submodels$iter) + 1)
      tmp[[1]] <- out
      for (i in seq(along = submodels$iter)) {
        tmp[[i + 1]] <- predict(modelFit,
                                newdata,
                                type = "prob",
                                n.iter = submodels$iter[[i]])
        colnames(tmp[[i + 1]]) <- modelFit$obsLevels
      }
      out <- lapply(tmp, as.data.frame)
    }
    out
  },
  levels = function(x)
    x$obsLevels,
  tags = c(
    "Tree-Based Model",
    "Ensemble Model",
    "Boosting",
    "Implicit Feature Selection",
    "Two Class Only",
    "Handle Missing Predictor Data"
  ),
  sort = function(x)
    x[order(x$iter, x$maxdepth, x$nu, x$bag.frac), ]
) #AMENDED

adaWEAKGRID <- expand.grid(
  maxdepth = 1,
  iter = seq(100, 1000, 100),
  nu = c(0.25, 0.5, 0.75, 1),
  #shrinkage
  bag.frac = c(0.25, 0.5, 0.75, 1)
)

adaWEAKMODEL <- train(
  offside ~ .,
  data = training_df,
  loss = "exponential",
  type = "discrete",
  method = adacustom,
  metric = "Kappa",
  trControl = resamplingCTRL,
  tuneGrid = adaWEAKGRID,
  control = rpart.control(
    maxdepth = 1,
    cp = -1,
    minsplit = 0,
    xval = 0
  )
) #as suggested by author for stumps

adaWEAKPREDICT <- predict(adaWEAKMODEL, testing_df)
adaWEAKprobPREDICT <- predict(adaWEAKMODEL, testing_df, type = "prob")
adaWEAKCM <-
  confusionMatrix(adaWEAKPREDICT, testing_df$offside, positive = 'Yes')

adaWEAKPREDICT_training <- predict(adaWEAKMODEL, training_df)
adaWEAKprobPREDICT_training <-
  predict(adaWEAKMODEL, training_df, type = "prob")
adaWEAKCM_training <-
  confusionMatrix(adaWEAKPREDICT_training, training_df$offside, positive =
                    'Yes')

adaWEAKVARIMP <-
  varplot(
    adaWEAKMODEL$finalModel,
    type = c("scores"),
    max.var.show = 379,
    plot.it = FALSE
  )
varplot(adaWEAKMODEL$finalModel, max.var.show = 20) #plotting top 20 variables

save.image("D:/offside/models_code_SAVED_FINAL_vXX_4_GtoH.RData")

#####H - LogitBoost - WEAK CLASSIFIERS (NO VARIMP)####

logitWEAKGRID <- expand.grid(nIter = seq(100, 1000, 100))

logitWEAKMODEL <- train(
  offside ~ .,
  data = training_df,
  method = "LogitBoost",
  #stumps are used automatically by the package
  metric = "Kappa",
  trControl = resamplingCTRL,
  tuneGrid = logitWEAKGRID
)

logitWEAKPREDICT <- predict(logitWEAKMODEL, testing_df)
logitWEAKprobPREDICT <-
  predict(logitWEAKMODEL, testing_df, type = "prob")
logitWEAKPREDICT[which(is.na(logitWEAKPREDICT) == TRUE)] <-
  "Yes" #Those which obtained a prob of 0.5 are given a YES manually since function gives NA
logitWEAKCM <-
  confusionMatrix(logitWEAKPREDICT, testing_df$offside, positive = 'Yes')

logitWEAKPREDICT_training <- predict(logitWEAKMODEL, training_df)
logitWEAKprobPREDICT_training <-
  predict(logitWEAKMODEL, training_df, type = "prob")
logitWEAKPREDICT_training[which(is.na(logitWEAKPREDICT_training) == TRUE)] <-
  "Yes" #Those which obtained a prob of 0.5 are given a YES manually since function gives NA
logitWEAKCM_training <-
  confusionMatrix(logitWEAKPREDICT_training, training_df$offside, positive =
                    'Yes')

save.image("D:/offside/models_code_SAVED_FINAL_vXX_4_GtoH.RData")

#####I - Real AdaBoost - WEAK CLASSIFIERS (INCLUDES VARIMP)####

rm(list = ls())
gc(reset = TRUE)
load("D:/offside/models_code_SAVED_FINAL_vXX_BASICS.RData")

#CREATING CUSTOM CARET MODEL TO ADD bag.frac
adacustom <- list(
  label = "CUSTOM Boosted Classification Trees",
  library = c("ada", "plyr"),
  loop = function(grid) {
    loop <- plyr::ddply(grid, c("nu", "maxdepth", "bag.frac"), #AMENDED
                        function(x)
                          c(iter = max(x$iter)))
    submodels <-
      vector(mode = "list", length = nrow(loop))
    for (i in seq(along = loop$iter)) {
      index <- which(
        grid$maxdepth == loop$maxdepth[i] &
          grid$nu == loop$nu[i] &
          grid$bag.frac == loop$bag.frac[i]
      ) #ADDED
      trees <- grid[index, "iter"]
      submodels[[i]] <-
        data.frame(iter = trees[trees != loop$iter[i]])
    }
    list(loop = loop, submodels = submodels)
  },
  type = c("Classification"),
  parameters = data.frame(
    parameter = c('iter', 'maxdepth', 'nu', 'bag.frac'),
    #AMENDED
    class = rep("numeric", 4),
    #AMENDED
    label = c(
      '#Trees',
      'Max Tree Depth',
      'Learning Rate',
      'Subsampling Parameter for Stochastic'
    )
  ),
  grid = function(x, y, len = NULL, search = "grid") {
    if (search == "grid") {
      out = expand.grid(
        iter = floor((1:len) * 50),
        maxdepth = seq(1, len),
        nu = .1,
        bag.frac = .5
      ) #ADDED
    } else {
      out <- data.frame(
        iter = sample(1:1000, replace = TRUE, size = len),
        maxdepth = sample(1:10, replace = TRUE, size = len),
        nu = runif(len, min = .001, max = .5),
        bag.frac = runif(len, min = .1, max = 1)
      ) #ADDED
    }
    out
  },
  fit = function(x, y, wts, param, lev, last, classProbs, ...) {
    theDots <- list(...)
    
    if (any(names(theDots) == "control")) {
      theDots$control$maxdepth <- param$maxdepth
      ctl <- theDots$control
      theDots$control <- NULL
      
    } else
      ctl <- rpart::rpart.control(
        maxdepth = param$maxdepth,
        cp = -1,
        minsplit = 0,
        xval = 0
      )
    
    modelArgs <- c(
      list(
        x = x,
        y = y,
        iter = param$iter,
        nu = param$nu,
        bag.frac = param$bag.frac,
        #ADDED
        control = ctl
      ),
      theDots
    )
    out <- do.call(ada::ada, modelArgs)
    out
  },
  predict = function(modelFit, newdata, submodels = NULL) {
    if (!is.data.frame(newdata))
      newdata <- as.data.frame(newdata, stringsAsFactors = TRUE)
    out <-
      predict(modelFit, newdata, n.iter = modelFit$tuneValue$iter)
    
    if (!is.null(submodels)) {
      tmp <- vector(mode = "list", length = length(submodels$iter) + 1)
      tmp[[1]] <- out
      for (i in seq(along = submodels$iter)) {
        tmp[[i + 1]] <-
          predict(modelFit, newdata, n.iter = submodels$iter[[i]])
      }
      out <- lapply(tmp, as.character)
    }
    out
  },
  prob = function(modelFit, newdata, submodels = NULL) {
    if (!is.data.frame(newdata))
      newdata <- as.data.frame(newdata, stringsAsFactors = TRUE)
    out <- predict(modelFit,
                   newdata,
                   type = "prob",
                   n.iter = modelFit$tuneValue$iter)
    colnames(out) <- modelFit$obsLevels
    
    if (!is.null(submodels)) {
      tmp <- vector(mode = "list", length = length(submodels$iter) + 1)
      tmp[[1]] <- out
      for (i in seq(along = submodels$iter)) {
        tmp[[i + 1]] <- predict(modelFit,
                                newdata,
                                type = "prob",
                                n.iter = submodels$iter[[i]])
        colnames(tmp[[i + 1]]) <- modelFit$obsLevels
      }
      out <- lapply(tmp, as.data.frame)
    }
    out
  },
  levels = function(x)
    x$obsLevels,
  tags = c(
    "Tree-Based Model",
    "Ensemble Model",
    "Boosting",
    "Implicit Feature Selection",
    "Two Class Only",
    "Handle Missing Predictor Data"
  ),
  sort = function(x)
    x[order(x$iter, x$maxdepth, x$nu, x$bag.frac), ]
) #AMENDED

realadaWEAKGRID <- expand.grid(
  maxdepth = 1,
  iter = seq(100, 1000, 100),
  nu = c(0.25, 0.5, 0.75, 1),
  #shrinkage
  bag.frac = c(0.25, 0.5, 0.75, 1)
)

realadaWEAKMODEL <- train(
  offside ~ .,
  data = training_df,
  loss = "exponential",
  type = "real",
  method = adacustom,
  metric = "Kappa",
  trControl = resamplingCTRL,
  tuneGrid = realadaWEAKGRID,
  control = rpart.control(
    maxdepth = 1,
    cp = -1,
    minsplit = 0,
    xval = 0
  )
) #as suggested by author for stumps

realadaWEAKPREDICT <- predict(realadaWEAKMODEL, testing_df)
realadaWEAKprobPREDICT <-
  predict(realadaWEAKMODEL, testing_df, type = "prob")
realadaWEAKCM <-
  confusionMatrix(realadaWEAKPREDICT, testing_df$offside, positive = 'Yes')

realadaWEAKPREDICT_training <- predict(realadaWEAKMODEL, training_df)
realadaWEAKprobPREDICT_training <-
  predict(realadaWEAKMODEL, training_df, type = "prob")
realadaWEAKCM_training <-
  confusionMatrix(realadaWEAKPREDICT_training, training_df$offside, positive =
                    'Yes')

realadaWEAKVARIMP <-
  varplot(
    realadaWEAKMODEL$finalModel,
    type = c("scores"),
    max.var.show = 379,
    plot.it = FALSE
  )
varplot(realadaWEAKMODEL$finalModel, max.var.show = 20) #plotting top 20 variables

save.image("D:/offside/models_code_SAVED_FINAL_vXX_5_ItoJ.RData")

#####J - Gentle AdaBoost - WEAK CLASSIFIERS (INCLUDES VARIMP)####

gentleadaWEAKGRID <- realadaWEAKGRID

gentleadaWEAKMODEL <- train(
  offside ~ .,
  data = training_df,
  loss = "exponential",
  type = "gentle",
  method = adacustom,
  metric = "Kappa",
  trControl = resamplingCTRL,
  tuneGrid = gentleadaWEAKGRID,
  control = rpart.control(
    maxdepth = 1,
    cp = -1,
    minsplit = 0,
    xval = 0
  )
) #as suggested by author for stumps

gentleadaWEAKPREDICT <- predict(gentleadaWEAKMODEL, testing_df)
gentleadaWEAKprobPREDICT <-
  predict(gentleadaWEAKMODEL, testing_df, type = "prob")
gentleadaWEAKCM <-
  confusionMatrix(gentleadaWEAKPREDICT, testing_df$offside, positive = 'Yes')

gentleadaWEAKPREDICT_training <-
  predict(gentleadaWEAKMODEL, training_df)
gentleadaWEAKprobPREDICT_training <-
  predict(gentleadaWEAKMODEL, training_df, type = "prob")
gentleadaWEAKCM_training <-
  confusionMatrix(gentleadaWEAKPREDICT_training,
                  training_df$offside,
                  positive = 'Yes')

gentleadaWEAKVARIMP <-
  varplot(
    gentleadaWEAKMODEL$finalModel,
    type = c("scores"),
    max.var.show = 379,
    plot.it = FALSE
  )
varplot(gentleadaWEAKMODEL$finalModel, max.var.show = 20) #plotting top 20 variables

save.image("D:/offside/models_code_SAVED_FINAL_vXX_5_ItoJ.RData")

#####K - Gradient Boosting logistic loss function (INCLUDES VARIMP)####

rm(list = ls())
gc(reset = TRUE)
load("D:/offside/models_code_SAVED_FINAL_vXX_BASICS.RData")

#CREATING CUSTOM CARET MODEL TO ADD sample_rate
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
    frame_name <-
      paste0("tmp_gbm_dat_", sample.int(100000, 1))
    tmp_train_dat = h2o::as.h2o(dat, destination_frame = frame_name)
    
    out <-
      h2o::h2o.gbm(
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
    newdata <-
      h2o::as.h2o(newdata, destination_frame = frame_name)
    as.data.frame(predict(modelFit, newdata), stringsAsFactors = TRUE)[, 1]
  },
  prob = function(modelFit, newdata, submodels = NULL) {
    frame_name <- paste0("new_gbm_dat_", sample.int(100000, 1))
    newdata <-
      h2o::as.h2o(newdata, destination_frame = frame_name)
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
    colnames(out)[colnames(out) == "relative_importance"] <-
      "Overall"
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

h2o.init(min_mem_size = '4g')

gbmGRID <- expand.grid(
  max_depth = c(3, 5, 7),
  ntrees = seq(100, 500, 100),
  min_rows = c(3, 6, 9, 12),
  #minimum number of observations in a node
  learn_rate = c(0.05, 0.1, 0.25, 0.5),
  #shrinkage
  sample_rate = c(0.5, 0.75, 1),
  #percentage of training data to be sampled per tree, <1 implies stochastic Gradient Boosting
  col_sample_rate_per_tree = c(0.5, 0.75, 1)
) #percentage of predictors to be sampled per tree

gbmlogMODEL <- train(
  offside ~ .,
  data = training_df,
  method = gbm_h2ocustom,
  metric = "Kappa",
  trControl = resamplingCTRL,
  tuneGrid = gbmGRID
)

gbmlogPREDICT <- predict(gbmlogMODEL2, testing_df)
gbmlogprobPREDICT <- predict(gbmlogMODEL, testing_df, type = "prob")
gbmlogCM <-
  confusionMatrix(gbmlogPREDICT, testing_df$offside, positive = 'Yes')

gbmlogPREDICT_training <- predict(gbmlogMODEL, training_df)
gbmlogprobPREDICT_training <-
  predict(gbmlogMODEL, training_df, type = "prob")
gbmlogCM_training <-
  confusionMatrix(gbmlogPREDICT_training, training_df$offside, positive =
                    'Yes')

gbmlogVARIMP <- h2o.varimp(gbmlogMODEL$finalModel)

save.image("D:/offside/models_code_SAVED_FINAL_vXX_6_KtoM.RData")

#####L - EXTREME GRADIENT BOOSTING - TREES (INCLUDES VARIMP)####

#CREATING CUSTOM CARET MODEL TO ADD lambda
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
    submodels <-
      vector(mode = "list", length = nrow(loop))
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
          x <-
            xgboost::xgb.DMatrix(x, label = y, missing = NA)
        else
          xgboost::setinfo(x, "label", y)
        
        if (!is.null(wts))
          xgboost::setinfo(x, 'weight', wts)
        
        out <-
          xgboost::xgb.train(
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
          x <-
            xgboost::xgb.DMatrix(x, label = y, missing = NA)
        else
          xgboost::setinfo(x, "label", y)
        
        if (!is.null(wts))
          xgboost::setinfo(x, 'weight', wts)
        
        out <-
          xgboost::xgb.train(
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
        x <-
          xgboost::xgb.DMatrix(x, label = y, missing = NA)
      else
        xgboost::setinfo(x, "label", y)
      
      if (!is.null(wts))
        xgboost::setinfo(x, 'weight', wts)
      
      out <-
        xgboost::xgb.train(
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
      newdata <-
        xgboost::xgb.DMatrix(data = newdata, missing = NA)
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
        out <-
          modelFit$obsLevels[apply(out, 1, which.max)]
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
            tmp_pred <-
              modelFit$obsLevels[apply(tmp_pred, 1, which.max)]
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
      newdata <-
        xgboost::xgb.DMatrix(data = newdata, missing = NA)
    }
    
    if (!is.null(modelFit$param$objective) &&
        modelFit$param$objective == 'binary:logitraw') {
      p <- predict(modelFit, newdata)
      out <-
        binomial()$linkinv(p) # exp(p)/(1+exp(p))
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
    out <-
      as.data.frame(out, stringsAsFactors = TRUE)
    
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
        tmp_pred <-
          as.data.frame(tmp_pred, stringsAsFactors = TRUE)
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
    imp <-
      as.data.frame(imp, stringsAsFactors = TRUE)[, 1:2]
    rownames(imp) <- as.character(imp[, 1])
    imp <- imp[, 2, drop = FALSE]
    colnames(imp) <- "Overall"
    
    missing <-
      object$xNames[!(object$xNames %in% rownames(imp))]
    missing_imp <-
      data.frame(Overall = rep(0, times = length(missing)))
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
    max_depth = c(3, 5, 7, 9),
    #max depth of a tree - default 6
    eta = c(0.05, 0.1, 0.25, 0.5),
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
  offside ~ .,
  data = training_df,
  method = xgbcustom,
  metric = "Kappa",
  trControl = resamplingCTRL,
  tuneGrid = xgbTreeGRID,
  verbosity = 0
)

xgbTreePREDICT <- predict(xgbTreeMODEL, testing_df)
xgbTreeprobPREDICT <- predict(xgbTreeMODEL, testing_df, type = "prob")
xgbTreeCM <-
  confusionMatrix(xgbTreePREDICT, testing_df$offside, positive = 'Yes')

xgbTreePREDICT_training <- predict(xgbTreeMODEL, training_df)
xgbTreeprobPREDICT_training <-
  predict(xgbTreeMODEL, training_df, type = "prob")
xgbTreeCM_training <-
  confusionMatrix(xgbTreePREDICT_training, training_df$offside, positive =
                    'Yes')

xgbTreeVARIMP <-
  xgb.importance(model = xgbTreeMODEL$finalModel,
                 feature_names = colnames(training_df[2:ncol(training_df)]))

save.image("D:/offside/models_code_SAVED_FINAL_vXX_6_KtoM.RData")

#####M - EXTREME GRADIENT BOOSTING - TREES WITH DROPOUT (INCLUDES VARIMP)####

#CREATING CUSTOM CARET MODEL TO ADD lambda, rate_drop and skip_drop
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
      submodels <-
        vector(mode = "list", length = nrow(loop))
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
            x <-
              xgboost::xgb.DMatrix(x, label = y, missing = NA)
          else
            xgboost::setinfo(x, "label", y)
          
          if (!is.null(wts))
            xgboost::setinfo(x, 'weight', wts)
          
          out <-
            xgboost::xgb.train(
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
            x <-
              xgboost::xgb.DMatrix(x, label = y, missing = NA)
          else
            xgboost::setinfo(x, "label", y)
          
          if (!is.null(wts))
            xgboost::setinfo(x, 'weight', wts)
          
          out <-
            xgboost::xgb.train(
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
          x <-
            xgboost::xgb.DMatrix(x, label = y, missing = NA)
        else
          xgboost::setinfo(x, "label", y)
        
        if (!is.null(wts))
          xgboost::setinfo(x, 'weight', wts)
        
        out <-
          xgboost::xgb.train(
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
        newdata <-
          xgboost::xgb.DMatrix(data = newdata, missing = NA)
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
          out <-
            modelFit$obsLevels[apply(out, 1, which.max)]
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
              tmp_pred <-
                modelFit$obsLevels[apply(tmp_pred, 1, which.max)]
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
        newdata <-
          xgboost::xgb.DMatrix(data = newdata, missing = NA)
      }
      
      if (!is.null(modelFit$param$objective) &&
          modelFit$param$objective == 'binary:logitraw') {
        p <- predict(modelFit, newdata)
        out <-
          binomial()$linkinv(p) # exp(p)/(1+exp(p))
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
      out <-
        as.data.frame(out, stringsAsFactors = TRUE)
      
      if (!is.null(submodels)) {
        tmp <- vector(mode = "list", length = nrow(submodels) + 1)
        tmp[[1]] <- out
        for (j in seq(along = submodels$nrounds)) {
          tmp_pred <-
            predict(modelFit, newdata, ntreelimit = submodels$nrounds[j])
          if (length(modelFit$obsLevels) == 2) {
            tmp_pred <- cbind(tmp_pred, 1 - tmp_pred)
            colnames(tmp_pred) <-
              modelFit$obsLevels
          } else {
            tmp_pred <-
              matrix(tmp_pred,
                     ncol = length(modelFit$obsLevels),
                     byrow = TRUE)
            colnames(tmp_pred) <-
              modelFit$obsLevels
          }
          tmp_pred <-
            as.data.frame(tmp_pred, stringsAsFactors = TRUE)
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
      imp <-
        as.data.frame(imp, stringsAsFactors = TRUE)[, 1:2]
      rownames(imp) <- as.character(imp[, 1])
      imp <- imp[, 2, drop = FALSE]
      colnames(imp) <- "Overall"
      
      missing <-
        object$xNames[!(object$xNames %in% rownames(imp))]
      missing_imp <-
        data.frame(Overall = rep(0, times = length(missing)))
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
  offside ~ .,
  data = training_df,
  method = xgbdartcustom,
  metric = "Kappa",
  trControl = resamplingCTRL,
  tuneGrid = xgbDARTGRID_min,
  verbosity = 0
)

xgbDARTPREDICT_min <- predict(xgbDARTMODEL_min, testing_df)
xgbDARTprobPREDICT_min <-
  predict(xgbDARTMODEL_min, testing_df, type = "prob")
xgbDARTCM_min <-
  confusionMatrix(xgbDARTPREDICT_min, testing_df$offside, positive = 'Yes')

xgbDARTPREDICT_min_training <- predict(xgbDARTMODEL_min, training_df)
xgbDARTprobPREDICT_min_training <-
  predict(xgbDARTMODEL_min, training_df, type = "prob")
xgbDARTCM_min_training <-
  confusionMatrix(xgbDARTPREDICT_min_training, training_df$offside, positive =
                    'Yes')

xgbDARTVARIMP_min <- xgb.importance(model = xgbDARTMODEL_min$finalModel)

xgbDARTGRID <-
  expand.grid(
    nrounds = c(50, 100, 150, 200, 300),
    #max boosting iterations - no default
    max_depth = c(3, 5, 7, 9),
    #max depth of a tree - default 6
    eta = c(0.05, 0.1, 0.25, 0.5),
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
  offside ~ .,
  data = training_df,
  method = xgbdartcustom,
  metric = "Kappa",
  trControl = resamplingCTRL,
  tuneGrid = xgbDARTGRID,
  verbosity = 0
)

xgbDARTPREDICT <- predict(xgbDARTMODEL, testing_df)
xgbDARTprobPREDICT <- predict(xgbDARTMODEL, testing_df, type = "prob")
xgbDARTCM <-
  confusionMatrix(xgbDARTPREDICT, testing_df$offside, positive = 'Yes')

xgbDARTPREDICT_training <- predict(xgbDARTMODEL, training_df)
xgbDARTprobPREDICT_training <-
  predict(xgbDARTMODEL, training_df, type = "prob")
xgbDARTCM_training <-
  confusionMatrix(xgbDARTPREDICT_training, training_df$offside, positive =
                    'Yes')

xgbDARTVARIMP <- xgb.importance(model = xgbDARTMODEL$finalModel)

save.image("D:/offside/models_code_SAVED_FINAL_vXX_6_KtoM.RData")

####Stacking 1 - (TOP 2 - DEPENDING ON VALIDATION) GRF and RF####

rm(list = ls())
to_load_1 <- c("resamplingCTRL",
               "resamplingINDEX",
               "testing_df",
               "grfMODEL",
               "rfMODEL")
#loading required objects
load("D:/offside/models_code_SAVED_FINAL_vXX_BASICS.RData")
load("D:/offside/models_code_SAVED_FINAL_vXX_1_AtoC.RData")
rm(list = ls()[!(ls() %in% c(to_load_1, "to_load_1"))])
load("D:/offside/models_code_SAVED_FINAL_vXX_6_KtoM.RData")
rm(list = ls()[!(ls() %in% c(to_load_1, "to_load_1"))])
gc(reset = TRUE)

gbmcustom <- list(
  label = "Stochastic Gradient Boosting",
  library = c("gbm", "plyr"),
  type = c("Regression", "Classification"),
  parameters = data.frame(
    parameter = c(
      'n.trees',
      'interaction.depth',
      'shrinkage',
      'n.minobsinnode',
      'bag.fraction'
    ),
    class = rep("numeric", 5),
    label = c(
      '# Boosting Iterations',
      'Max Tree Depth',
      'Shrinkage',
      'Min. Terminal Node Size',
      'Subsampling'
    )
  ),
  grid = function(x, y, len = NULL, search = "grid") {
    if (search == "grid") {
      out <- expand.grid(
        interaction.depth = seq(1, len),
        n.trees = floor((1:len) * 50),
        shrinkage = .1,
        n.minobsinnode = 10,
        bag.fraction = 1
      )
    } else {
      out <- data.frame(
        n.trees = floor(runif(len, min = 1, max = 5000)),
        interaction.depth = sample(1:10, replace = TRUE, size = len),
        shrinkage = runif(len, min = .001, max = .6),
        n.minobsinnode = sample(5:25, replace = TRUE, size = len),
        bag.fraction = runif(len, min = .25, max = 1)
      )
    }
    out
  },
  loop = function(grid) {
    loop <-
      plyr::ddply(grid, c(
        "shrinkage",
        "interaction.depth",
        "n.minobsinnode",
        "bag.fraction"
      ),
      function(x)
        c(n.trees = max(x$n.trees)))
    submodels <-
      vector(mode = "list", length = nrow(loop))
    for (i in seq(along = loop$n.trees)) {
      index <- which(
        grid$interaction.depth == loop$interaction.depth[i] &
          grid$shrinkage == loop$shrinkage[i] &
          grid$n.minobsinnode == loop$n.minobsinnode[i] &
          grid$bag.fraction == loop$bag.fraction[i]
      )
      trees <- grid[index, "n.trees"]
      submodels[[i]] <-
        data.frame(n.trees = trees[trees != loop$n.trees[i]])
    }
    list(loop = loop, submodels = submodels)
  },
  fit = function(x, y, wts, param, lev, last, classProbs, ...) {
    theDots <- list(...)
    if (any(names(theDots) == "distribution")) {
      modDist <- theDots$distribution
      theDots$distribution <- NULL
    } else {
      if (is.numeric(y)) {
        modDist <- "gaussian"
      } else
        modDist <- if (length(lev) == 2)
          "bernoulli"
      else
        "multinomial"
    }
    
    if (!is.null(wts))
      theDots$w <- wts
    if (is.factor(y) &&
        length(lev) == 2)
      y <- ifelse(y == lev[1], 1, 0)
    if (!is.data.frame(x) | inherits(x, "tbl_df"))
      x <- as.data.frame(x, stringsAsFactors = TRUE)
    
    modArgs <- list(
      x = x,
      y = y,
      interaction.depth = param$interaction.depth,
      n.trees = param$n.trees,
      shrinkage = param$shrinkage,
      n.minobsinnode = param$n.minobsinnode,
      bag.fraction = param$bag.fraction,
      distribution = modDist
    )
    if (any(names(theDots) == "family"))
      modArgs$distribution <- NULL
    
    if (length(theDots) > 0)
      modArgs <- c(modArgs, theDots)
    
    do.call(gbm::gbm.fit, modArgs)
  },
  predict = function(modelFit, newdata, submodels = NULL) {
    out <- predict(modelFit,
                   newdata,
                   type = "response",
                   n.trees = modelFit$tuneValue$n.trees)
    out[is.nan(out)] <- NA
    out <- switch(
      modelFit$distribution$name,
      multinomial = {
        colnames(out[, , 1, drop = FALSE])[apply(out[, , 1, drop = FALSE], 1, which.max)]
      },
      bernoulli = ,
      adaboost = ,
      huberized = {
        ifelse(out >= .5,
               modelFit$obsLevels[1],
               modelFit$obsLevels[2])
      },
      gaussian = ,
      laplace = ,
      tdist = ,
      poisson = ,
      quantile = {
        out
      }
    )
    
    if (!is.null(submodels)) {
      tmp <-
        predict(modelFit,
                newdata,
                type = "response",
                n.trees = submodels$n.trees)
      out <- switch(
        modelFit$distribution$name,
        multinomial = {
          lvl <- colnames(tmp[, , 1, drop = FALSE])
          tmp <-
            apply(tmp, 3, function(x)
              apply(x, 1, which.max))
          if (is.vector(tmp))
            tmp <- matrix(tmp, nrow = 1)
          tmp <-
            t(apply(tmp, 1, function(x, lvl)
              lvl[x], lvl = lvl))
          if (nrow(tmp) == 1 &
              nrow(newdata) > 1)
            tmp <- t(tmp)
          tmp <-
            as.list(as.data.frame(tmp, stringsAsFactors = FALSE))
          c(list(out), tmp)
        },
        bernoulli = ,
        adaboost = ,
        huberized = {
          tmp <- ifelse(tmp >= .5,
                        modelFit$obsLevels[1],
                        modelFit$obsLevels[2])
          tmp <-
            as.list(as.data.frame(tmp, stringsAsFactors = FALSE))
          c(list(out), tmp)
        },
        gaussian = ,
        laplace = ,
        tdist = ,
        poisson = ,
        quantile = {
          tmp <- as.list(as.data.frame(tmp, stringsAsFactors = TRUE))
          c(list(out), tmp)
        }
      )
    }
    out
  },
  prob = function(modelFit, newdata, submodels = NULL) {
    out <- predict(modelFit,
                   newdata,
                   type = "response",
                   n.trees = modelFit$tuneValue$n.trees)
    
    out[is.nan(out)] <- NA
    
    out <- switch(
      modelFit$distribution$name,
      multinomial = {
        out <-
          if (dim(out)[3] == 1)
            as.data.frame(out, stringsAsFactors = TRUE)
        else
          out[, , 1]
        colnames(out) <-
          modelFit$obsLevels
        out
      },
      bernoulli = ,
      adaboost = ,
      huberized = {
        out <- cbind(out, 1 - out)
        colnames(out) <-
          modelFit$obsLevels
        out
      },
      gaussian = ,
      laplace = ,
      tdist = ,
      poisson = {
        out
      }
    )
    
    if (!is.null(submodels)) {
      tmp <-
        predict(modelFit,
                newdata,
                type = "response",
                n.trees = submodels$n.trees)
      tmp <- switch(
        modelFit$distribution$name,
        multinomial = {
          apply(tmp, 3, function(x)
            data.frame(x))
        },
        bernoulli = ,
        adaboost = ,
        huberized = {
          tmp <- as.list(as.data.frame(tmp, stringsAsFactors = TRUE))
          lapply(tmp, function(x, lvl) {
            x <- cbind(x, 1 - x)
            colnames(x) <- lvl
            x
          }, lvl = modelFit$obsLevels)
        }
      )
      out <- c(list(out), tmp)
    }
    out
  },
  predictors = function(x, ...) {
    vi <- relative.influence(x, n.trees = x$tuneValue$n.trees)
    names(vi)[vi > 0]
  },
  varImp = function(object, numTrees = NULL, ...) {
    if (is.null(numTrees))
      numTrees <- object$tuneValue$n.trees
    varImp <-
      relative.influence(object, n.trees = numTrees)
    out <- data.frame(varImp)
    colnames(out) <- "Overall"
    rownames(out) <- object$var.names
    out
  },
  levels = function(x) {
    if (x$distribution$name %in% c("gaussian", "laplace", "tdist"))
      return(NULL)
    if (is.null(x$classes)) {
      out <- if (any(names(x) == "obsLevels"))
        x$obsLevels
      else
        NULL
    } else {
      out <- x$classes
    }
    out
  },
  tags = c(
    "Tree-Based Model",
    "Boosting",
    "Ensemble Model",
    "Implicit Feature Selection",
    "Accepts Case Weights"
  ),
  sort = function(x) {
    x[order(x$n.trees, x$interaction.depth, x$shrinkage), ]
  }
)

gbmGRID <-
  expand.grid(
    interaction.depth = seq(1, 7, 1),
    #max tree depth
    n.trees = seq(100, 300, 50),
    #number of trees
    n.minobsinnode = c(3, 6, 9, 12),
    #minimum number of observations in a node
    shrinkage = c(0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
    #shrinkage
    bag.fraction = c(0.25, 0.5, 0.75, 1)
  ) #subsampling

stacking1MODEL <-
  caretStack(
    as.caretList(list(grf = grfMODEL, rf = rfMODEL)),
    method = gbmcustom,
    distribution = "bernoulli",
    #logistic loss function
    tuneGrid = gbmGRID,
    trControl = trainControl(
      method = "cv",
      number = 3,
      verboseIter = TRUE
    ),
    verbose = FALSE
  )
stacking1PREDICT <-
  predict(stacking1MODEL, as.data.frame(testing_df[, -c(1)]))
stacking1CM <-
  confusionMatrix(stacking1PREDICT, testing_df$offside, positive = 'Yes')
stacking1CM

save(stacking1CM, stacking1PREDICT, file = "D:/offside/models_code_SAVED_FINAL_vXX_stacking1CM&PREDICT.RData")
save(stacking1MODEL, file = "D:/offside/models_code_SAVED_FINAL_vXX_stacking1MODEL.RData")

####Stacking 2 - (TOP 3 - DEPENDING ON VALIDATION) GRF and RF and GRRF####

rm(list = ls()[!(ls() %in% c("gbmcustom"))])
to_load_2 <- c(
  "resamplingCTRL",
  "resamplingINDEX",
  "testing_df",
  "grfMODEL",
  "rfMODEL",
  "grrfMODEL"
)
#loading required objects
load("D:/offside/models_code_SAVED_FINAL_vXX_BASICS.RData")
load("D:/offside/models_code_SAVED_FINAL_vXX_1_AtoC.RData")
rm(list = ls()[!(ls() %in% c(to_load_2, "to_load_2"))])
load("D:/offside/models_code_SAVED_FINAL_vXX_6_KtoM.RData")
rm(list = ls()[!(ls() %in% c(to_load_2, "to_load_2"))])
gc(reset = TRUE)

gbmGRID <-
  expand.grid(
    interaction.depth = seq(1, 7, 1),
    #max tree depth
    n.trees = seq(100, 300, 50),
    #number of trees
    n.minobsinnode = c(3, 6, 9, 12),
    #minimum number of observations in a node
    shrinkage = c(0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
    #shrinkage
    bag.fraction = c(0.25, 0.5, 0.75, 1)
  ) #subsampling

stacking2MODEL <-
  caretStack(
    as.caretList(list(
      grf = grfMODEL, rf = rfMODEL, grrf = grrfMODEL
    )),
    method = gbmcustom,
    distribution = "bernoulli",
    #logistic loss function
    tuneGrid = gbmGRID,
    trControl = trainControl(
      method = "cv",
      number = 3,
      verboseIter = TRUE
    ),
    verbose = FALSE
  )
stacking2PREDICT <-
  predict(stacking2MODEL, as.data.frame(testing_df[, -c(1)]))
stacking2CM <-
  confusionMatrix(stacking2PREDICT, testing_df$offside, positive = 'Yes')

save(stacking2CM, stacking2PREDICT, file = "D:/offside/models_code_SAVED_FINAL_vXX_stacking2CM&PREDICT.RData")
save(stacking2MODEL, file = "D:/offside/models_code_SAVED_FINAL_vXX_stacking2MODEL.RData")

####Stacking 3 - (TOP 4 - DEPENDING ON VALIDATION) GRF and RF and GRRF and AdaBoost####

rm(list = ls()[!(ls() %in% c("gbmcustom"))])
to_load_3 <- c(
  "resamplingCTRL",
  "resamplingINDEX",
  "testing_df",
  "grfMODEL",
  "rfMODEL",
  "grrfMODEL",
  "adaMODEL"
)
#loading required objects
load("D:/offside/models_code_SAVED_FINAL_vXX_BASICS.RData")
load("D:/offside/models_code_SAVED_FINAL_vXX_1_AtoC.RData")
rm(list = ls()[!(ls() %in% c(to_load_3, "to_load_3"))])
load("D:/offside/models_code_SAVED_FINAL_vXX_2_DtoE.RData")
rm(list = ls()[!(ls() %in% c(to_load_3, "to_load_3"))])
load("D:/offside/models_code_SAVED_FINAL_vXX_6_KtoM.RData")
rm(list = ls()[!(ls() %in% c(to_load_3, "to_load_3"))])
gc(reset = TRUE)

gbmGRID <-
  expand.grid(
    interaction.depth = seq(1, 7, 1),
    #max tree depth
    n.trees = seq(100, 300, 50),
    #number of trees
    n.minobsinnode = c(3, 6, 9, 12),
    #minimum number of observations in a node
    shrinkage = c(0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
    #shrinkage
    bag.fraction = c(0.25, 0.5, 0.75, 1)
  ) #subsampling

stacking3MODEL <-
  caretStack(
    as.caretList(list(
      grf = grfMODEL,
      rf = rfMODEL,
      grrf = grrfMODEL,
      ada = adaMODEL
    )),
    method = gbmcustom,
    distribution = "bernoulli",
    #logistic loss function
    tuneGrid = gbmGRID,
    trControl = trainControl(
      method = "cv",
      number = 3,
      verboseIter = TRUE
    ),
    verbose = FALSE
  )
stacking3PREDICT <-
  predict(stacking3MODEL, as.data.frame(testing_df[, -c(1)]))
stacking3CM <-
  confusionMatrix(stacking3PREDICT, testing_df$offside, positive = 'Yes')

save(stacking3CM, stacking3PREDICT, file = "D:/offside/models_code_SAVED_FINAL_vXX_stacking3CM&PREDICT.RData")
save(stacking3MODEL, file = "D:/offside/models_code_SAVED_FINAL_vXX_stacking3MODEL.RData")

####Stacking 4 - (TOP 5 - DEPENDING ON VALIDATION) GRF and RF and GRRF and AdaBoost and Gentle AdaBoost####

rm(list = ls()[!(ls() %in% c("gbmcustom"))])
to_load_4 <- c(
  "resamplingCTRL",
  "resamplingINDEX",
  "testing_df",
  "grfMODEL",
  "rfMODEL",
  "grrfMODEL",
  "adaMODEL",
  "gentleadaMODEL"
)
#loading required objects
load("D:/offside/models_code_SAVED_FINAL_vXX_BASICS.RData")
load("D:/offside/models_code_SAVED_FINAL_vXX_1_AtoC.RData")
rm(list = ls()[!(ls() %in% c(to_load_4, "to_load_4"))])
load("D:/offside/models_code_SAVED_FINAL_vXX_2_DtoE.RData")
rm(list = ls()[!(ls() %in% c(to_load_4, "to_load_4"))])
load("D:/offside/models_code_SAVED_FINAL_vXX_6_KtoM.RData")
rm(list = ls()[!(ls() %in% c(to_load_4, "to_load_4"))])
gc(reset = TRUE)

gbmGRID <-
  expand.grid(
    interaction.depth = seq(1, 7, 1),
    #max tree depth
    n.trees = seq(100, 300, 50),
    #number of trees
    n.minobsinnode = c(3, 6, 9, 12),
    #minimum number of observations in a node
    shrinkage = c(0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
    #shrinkage
    bag.fraction = c(0.25, 0.5, 0.75, 1)
  ) #subsampling

stacking4MODEL <-
  caretStack(
    as.caretList(
      list(
        grf = grfMODEL,
        rf = rfMODEL,
        grrf = grrfMODEL,
        ada = adaMODEL,
        gentleada = gentleadaMODEL
      )
    ),
    method = gbmcustom,
    distribution = "bernoulli",
    #logistic loss function
    tuneGrid = gbmGRID,
    trControl = trainControl(
      method = "cv",
      number = 3,
      verboseIter = TRUE
    ),
    verbose = FALSE
  )
stacking4PREDICT <-
  predict(stacking4MODEL, as.data.frame(testing_df[, -c(1)]))
stacking4CM <-
  confusionMatrix(stacking4PREDICT, testing_df$offside, positive = 'Yes')

save(stacking4CM, stacking4PREDICT, file = "D:/offside/models_code_SAVED_FINAL_vXX_stacking4CM&PREDICT.RData")
save(stacking4MODEL, file = "D:/offside/models_code_SAVED_FINAL_vXX_stacking4MODEL.RData")

####SUMMARY - VARIMP####
rm(list = ls())

modelVARIMPNAMES <- c(
  "cartVARIMP",
  "rfVARIMP",
  "rrfVARIMP",
  "grrfVARIMP",
  "grfVARIMP",
  "adaVARIMP",
  "realadaVARIMP",
  "gentleadaVARIMP",
  "adaWEAKVARIMP",
  "realadaWEAKVARIMP",
  "gentleadaWEAKVARIMP",
  "gbmlogVARIMP",
  "xgbTreeVARIMP",
  "xgbDARTVARIMP"
)

load("D:/offside/models_code_SAVED_FINAL_vXX_1_AtoC.RData")
rm(list = ls()[!(ls() %in% c(modelVARIMPNAMES, "modelVARIMPNAMES"))])
load("D:/offside/models_code_SAVED_FINAL_vXX_2_DtoE.RData")
rm(list = ls()[!(ls() %in% c(modelVARIMPNAMES, "modelVARIMPNAMES"))])
load("D:/offside/models_code_SAVED_FINAL_vXX_3_F.RData")
rm(list = ls()[!(ls() %in% c(modelVARIMPNAMES, "modelVARIMPNAMES"))])
load("D:/offside/models_code_SAVED_FINAL_vXX_4_GtoH.RData")
rm(list = ls()[!(ls() %in% c(modelVARIMPNAMES, "modelVARIMPNAMES"))])
load("D:/offside/models_code_SAVED_FINAL_vXX_5_ItoJ.RData")
rm(list = ls()[!(ls() %in% c(modelVARIMPNAMES, "modelVARIMPNAMES"))])
load("D:/offside/models_code_SAVED_FINAL_vXX_6_KtoM.RData")
rm(list = ls()[!(ls() %in% c(modelVARIMPNAMES, "modelVARIMPNAMES"))])
gc(reset = TRUE)

adaVARIMP <- as.data.frame(adaVARIMP)
adaWEAKVARIMP <- as.data.frame(adaWEAKVARIMP)
gentleadaVARIMP <- as.data.frame(gentleadaVARIMP)
gentleadaWEAKVARIMP <- as.data.frame(gentleadaWEAKVARIMP)
realadaVARIMP <- as.data.frame(realadaVARIMP)
realadaWEAKVARIMP <- as.data.frame(realadaWEAKVARIMP)
rfVARIMP <- as.data.frame(rfVARIMP[, 2])
rrfVARIMP <- as.data.frame(rrfVARIMP)
grrfVARIMP <- as.data.frame(grrfVARIMP)
grfVARIMP <- as.data.frame(grfVARIMP)
colnames(rfVARIMP) <- NULL
colnames(rrfVARIMP) <- NULL
colnames(grrfVARIMP) <- NULL
colnames(grfVARIMP) <- NULL

#load("D:/offside/models_code_SAVED_FINAL_vXX_BASICS.RData")

adaVARIMP <- cbind(rownames(adaVARIMP), adaVARIMP)
adaWEAKVARIMP <- cbind(rownames(adaWEAKVARIMP), adaWEAKVARIMP)
cartVARIMP <- cbind(rownames(cartVARIMP), cartVARIMP)
gentleadaVARIMP <- cbind(rownames(gentleadaVARIMP), gentleadaVARIMP)
gentleadaWEAKVARIMP <-
  cbind(rownames(gentleadaWEAKVARIMP), gentleadaWEAKVARIMP)
realadaVARIMP <- cbind(rownames(realadaVARIMP), realadaVARIMP)
realadaWEAKVARIMP <-
  cbind(rownames(realadaWEAKVARIMP), realadaWEAKVARIMP)
rfVARIMP <- cbind(rownames(rfVARIMP), rfVARIMP)
rrfVARIMP <- cbind(rownames(rrfVARIMP), rrfVARIMP)
grrfVARIMP <- cbind(rownames(grrfVARIMP), grrfVARIMP)
grfVARIMP <- cbind(rownames(grfVARIMP), grfVARIMP)

rownames(adaVARIMP) <- NULL
rownames(adaWEAKVARIMP) <- NULL
rownames(cartVARIMP) <- NULL
rownames(gentleadaVARIMP) <- NULL
rownames(gentleadaWEAKVARIMP) <- NULL
rownames(realadaVARIMP) <- NULL
rownames(realadaWEAKVARIMP) <- NULL
rownames(rfVARIMP) <- NULL
rownames(rrfVARIMP) <- NULL
rownames(grrfVARIMP) <- NULL
rownames(grfVARIMP) <- NULL

gbmlogVARIMP <- gbmlogVARIMP[, -c(3, 4)]
xgbDARTVARIMP <- xgbDARTVARIMP[, -c(3, 4)]
xgbTreeVARIMP <- xgbTreeVARIMP[, -c(3, 4)]

#converting camera_angle1 to camera_angle for merging correctly later
for (i in 1:length(modelVARIMPNAMES)) {
  if (length(which(eval(parse(
    text = paste(modelVARIMPNAMES[i], "[,1]", sep = "")
  )) == "camera_angle1")) != 0) {
    indicator <-
      which(eval(parse(
        text = paste(modelVARIMPNAMES[i], "[,1]", sep = "")
      )) == "camera_angle1")
    eval(parse(
      text = paste(modelVARIMPNAMES[i], "[indicator,1]<-'camera_angle'", sep =
                     "")
    ))
  }
}

###

not_included_xgbTree <-
  which(!(colnames(training_df[2:380]) %in% xgbTreeVARIMP[["Feature"]]))
names_xgbTree <-
  colnames(training_df[not_included_xgbTree + 1]) #+1 since we starting from 2 above
to_bind_xgbTree <- mat.or.vec(length(names_xgbTree), 2)
to_bind_xgbTree[, 1] <- names_xgbTree
colnames(to_bind_xgbTree) <- colnames(xgbTreeVARIMP)
xgbTreeVARIMP <- rbind(xgbTreeVARIMP, to_bind_xgbTree)

not_included_xgbDART <-
  which(!(colnames(training_df[2:380]) %in% xgbDARTVARIMP[["Feature"]]))
names_xgbDART <-
  colnames(training_df[not_included_xgbDART + 1]) #+1 since we starting from 2 above
to_bind_xgbDART <- mat.or.vec(length(names_xgbDART), 2)
to_bind_xgbDART[, 1] <- names_xgbDART
colnames(to_bind_xgbDART) <- colnames(xgbDARTVARIMP)
xgbDARTVARIMP <- rbind(xgbDARTVARIMP, to_bind_xgbDART)

#put all data frames into list
VARIMP_list <-
  list(
    adaVARIMP,
    adaWEAKVARIMP,
    cartVARIMP,
    gentleadaVARIMP,
    gentleadaWEAKVARIMP,
    realadaVARIMP,
    realadaWEAKVARIMP,
    rfVARIMP,
    rrfVARIMP,
    grrfVARIMP,
    grfVARIMP,
    gbmlogVARIMP,
    xgbDARTVARIMP,
    xgbTreeVARIMP
  )

#merge all data frames in list
VARIMP <- Reduce(function(x, y)
  merge(x, y, by = 1), VARIMP_list)

colnames(VARIMP) <-
  c(
    "variable",
    "adaVARIMP",
    "adaWEAKVARIMP",
    "cartVARIMP",
    "gentleadaVARIMP",
    "gentleadaWEAKVARIMP",
    "realadaVARIMP",
    "realadaWEAKVARIMP",
    "rfVARIMP",
    "rrfVARIMP",
    "grrfVARIMP",
    "grfVARIMP",
    "gbmlogVARIMP",
    "xgbDARTVARIMP",
    "xgbTreeVARIMP"
  )

VARIMP[, 14] <- as.numeric(VARIMP[, 14])
VARIMP[, 15] <- as.numeric(VARIMP[, 15])

VARIMP[, 2:15] <- scale(VARIMP[, 2:15], center = TRUE, scale = TRUE)
VARIMP$TOTAL <- rowSums(VARIMP[, 2:15])
VARIMP$MEAN <- rowMeans(VARIMP[, 2:15])

VARIMP_top20 <-
  VARIMP[order(VARIMP[["MEAN"]], decreasing = TRUE), ][c(1:20), ]

VARIMP_modelwise <- mat.or.vec(379, 0)
for (i in 2:15) {
  #2:15 = the models
  VARIMP_modelwise <-
    cbind(VARIMP_modelwise, VARIMP[order(VARIMP[, i], decreasing = TRUE), c(1, i)])
}

write.csv(VARIMP_modelwise[1:15, ], "varimp_modelwise.csv")
write.csv(VARIMP_top20, "varimp.csv")

rm(
  modelVARIMPNAMES,
  VARIMP_list,
  resamplingCTRL,
  resamplingINDEX,
  training_df,
  testing_df,
  to_bind_xgbDART,
  to_bind_xgbTree,
  not_included_xgbDART,
  not_included_xgbTree,
  names_xgbDART,
  names_xgbTree,
  i,
  indicator
)

save.image("D:/offside/models_code_SAVED_FINAL_vXX_VARIMP_ONLY.RData")


####SUMMARY - TESTING####

rm(list = ls())

modelNAMES <- c(
  "cartCM",
  "rfCM",
  "rrfCM",
  "grrfCM",
  "grfCM",
  "adaCM",
  "realadaCM",
  "gentleadaCM",
  "adaWEAKCM",
  "logitWEAKCM",
  "realadaWEAKCM",
  "gentleadaWEAKCM",
  "gbmlogCM",
  "xgbTreeCM",
  "xgbDARTCM"
)

modelNAMESPREDICT <- c(
  "cartPREDICT",
  "rfPREDICT",
  "rrfPREDICT",
  "grrfPREDICT",
  "grfPREDICT",
  "adaPREDICT",
  "realadaPREDICT",
  "gentleadaPREDICT",
  "adaWEAKPREDICT",
  "logitWEAKPREDICT",
  "realadaWEAKPREDICT",
  "gentleadaWEAKPREDICT",
  "gbmlogPREDICT",
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
  "realadaprobPREDICT",
  "gentleadaprobPREDICT",
  "adaWEAKprobPREDICT",
  "logitWEAKprobPREDICT",
  "realadaWEAKprobPREDICT",
  "gentleadaWEAKprobPREDICT",
  "gbmlogprobPREDICT",
  "xgbTreeprobPREDICT",
  "xgbDARTprobPREDICT"
)

load("D:/offside/models_code_SAVED_FINAL_vXX_1_AtoC.RData")
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
load("D:/offside/models_code_SAVED_FINAL_vXX_2_DtoE.RData")
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
load("D:/offside/models_code_SAVED_FINAL_vXX_3_F.RData")
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
load("D:/offside/models_code_SAVED_FINAL_vXX_4_GtoH.RData")
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
load("D:/offside/models_code_SAVED_FINAL_vXX_5_ItoJ.RData")
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
load("D:/offside/models_code_SAVED_FINAL_vXX_6_KtoM.RData")
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

summaryMETRICS <- mat.or.vec(length(modelNAMES), 6)
colnames(summaryMETRICS) <-
  c("Accuracy",
    "Kappa",
    "Precision/PPV",
    "Recall/Sensitivity",
    "F1",
    "Specificity")
row.names(summaryMETRICS) <- c(
  "Decision Tree",
  "Random Forest",
  "Regularized Random Forests",
  "Guided Regularized Random Forests",
  "Guided Random Forests",
  "(Stochastic) AdaBoost (NOT WEAK)",
  "(Stochastic) Real AdaBoost (NOT WEAK)",
  "(Stochastic) Gentle AdaBoost (NOT WEAK)",
  "(Stochastic) AdaBoost (WEAK CLASSIFIERS)",
  "(Stochastic) LogitBoost (WEAK CLASSIFIERS)",
  "(Stochastic) Real AdaBoost (WEAK CLASSIFIERS)",
  "(Stochastic) Gentle AdaBoost (WEAK CLASSIFIERS)",
  "(Stochastic) Gradient Boosting using LOGISTIC Loss",
  "EXTREME GRADIENT BOOSTING - TREES",
  "EXTREME GRADIENT BOOSTING - TREES WITH DROPOUT"
)

for (i in 1:nrow(summaryMETRICS)) {
  summaryMETRICS[i, 1] = as.numeric(eval(parse(
    text = paste(modelNAMES[i], "$overall[1]", sep = "")
  ))) #Accuracy
  summaryMETRICS[i, 2] = as.numeric(eval(parse(
    text = paste(modelNAMES[i], "$overall[2]", sep = "")
  ))) #Kappa
  summaryMETRICS[i, 3] = as.numeric(eval(parse(
    text = paste(modelNAMES[i], "$byClass[5]", sep = "")
  ))) #Precision/Positive Predictive Value
  summaryMETRICS[i, 4] = as.numeric(eval(parse(
    text = paste(modelNAMES[i], "$byClass[6]", sep = "")
  ))) #Recall/Sensitivity
  summaryMETRICS[i, 5] = as.numeric(eval(parse(
    text = paste(modelNAMES[i], "$byClass[7]", sep = "")
  ))) #F1
  summaryMETRICS[i, 6] = as.numeric(eval(parse(
    text = paste(modelNAMES[i], "$byClass[2]", sep = "")
  ))) #Specificity
}

summaryMETRICS[order(summaryMETRICS[, 2], decreasing = TRUE), ]

save.image("D:/offside/models_code_SAVED_FINAL_vXX_RESULTS_ONLY.RData")

####SUMMARY - TRAINING####

rm(list = ls())

modelNAMES_training <- c(
  "cartCM_training",
  "rfCM_training",
  "rrfCM_training",
  "grrfCM_training",
  "grfCM_training",
  "adaCM_training",
  "realadaCM_training",
  "gentleadaCM_training",
  "adaWEAKCM_training",
  "logitWEAKCM_training",
  "realadaWEAKCM_training",
  "gentleadaWEAKCM_training",
  "gbmlogCM_training",
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
  "realadaPREDICT_training",
  "gentleadaPREDICT_training",
  "adaWEAKPREDICT_training",
  "logitWEAKPREDICT_training",
  "realadaWEAKPREDICT_training",
  "gentleadaWEAKPREDICT_training",
  "gbmlogPREDICT_training",
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
  "realadaprobPREDICT_training",
  "gentleadaprobPREDICT_training",
  "adaWEAKprobPREDICT_training",
  "logitWEAKprobPREDICT_training",
  "realadaWEAKprobPREDICT_training",
  "gentleadaWEAKprobPREDICT_training",
  "gbmlogprobPREDICT_training",
  "xgbTreeprobPREDICT_training",
  "xgbDARTprobPREDICT_training"
)

load("D:/offside/models_code_SAVED_FINAL_vXX_1_AtoC.RData")
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
load("D:/offside/models_code_SAVED_FINAL_vXX_2_DtoE.RData")
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
load("D:/offside/models_code_SAVED_FINAL_vXX_3_F.RData")
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
load("D:/offside/models_code_SAVED_FINAL_vXX_4_GtoH.RData")
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
load("D:/offside/models_code_SAVED_FINAL_vXX_5_ItoJ.RData")
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
load("D:/offside/models_code_SAVED_FINAL_vXX_6_KtoM.RData")
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

load("D:/offside/models_code_SAVED_FINAL_vXX_RESULTS_ONLY.RData")

summaryMETRICS_training <- mat.or.vec(length(modelNAMES_training), 6)
colnames(summaryMETRICS_training) <-
  c("Accuracy",
    "Kappa",
    "Precision/PPV",
    "Recall/Sensitivity",
    "F1",
    "Specificity")
row.names(summaryMETRICS_training) <- c(
  "Decision Tree",
  "Random Forest",
  "Regularized Random Forests",
  "Guided Regularized Random Forests",
  "Guided Random Forests",
  "(Stochastic) AdaBoost (NOT WEAK)",
  "(Stochastic) Real AdaBoost (NOT WEAK)",
  "(Stochastic) Gentle AdaBoost (NOT WEAK)",
  "(Stochastic) AdaBoost (WEAK CLASSIFIERS)",
  "(Stochastic) LogitBoost (WEAK CLASSIFIERS)",
  "(Stochastic) Real AdaBoost (WEAK CLASSIFIERS)",
  "(Stochastic) Gentle AdaBoost (WEAK CLASSIFIERS)",
  "(Stochastic) Gradient Boosting using LOGISTIC Loss",
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
  summaryMETRICS_training[i, 3] = as.numeric(eval(parse(
    text = paste(modelNAMES_training[i], "$byClass[5]", sep = "")
  ))) #Precision/Positive Predictive Value
  summaryMETRICS_training[i, 4] = as.numeric(eval(parse(
    text = paste(modelNAMES_training[i], "$byClass[6]", sep = "")
  ))) #Recall/Sensitivity
  summaryMETRICS_training[i, 5] = as.numeric(eval(parse(
    text = paste(modelNAMES_training[i], "$byClass[7]", sep = "")
  ))) #F1
  summaryMETRICS_training[i, 6] = as.numeric(eval(parse(
    text = paste(modelNAMES_training[i], "$byClass[2]", sep = "")
  ))) #Specificity
}

summaryMETRICS_training[order(summaryMETRICS_training[, 2], decreasing = TRUE), ]

save.image("D:/offside/models_code_SAVED_FINAL_vXX_RESULTS_ONLY.RData")

####SUMMARY - VALIDATION####

rm(list = ls())

modelNAMESMODEL <- c(
  "cartMODEL",
  "rfMODEL",
  "rrfMODEL",
  "grrfMODEL",
  "grfMODEL",
  "adaMODEL",
  "realadaMODEL",
  "gentleadaMODEL",
  "adaWEAKMODEL",
  "logitWEAKMODEL",
  "realadaWEAKMODEL",
  "gentleadaWEAKMODEL",
  "gbmlogMODEL",
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
  "(Stochastic) AdaBoost (NOT WEAK)",
  "(Stochastic) Real AdaBoost (NOT WEAK)",
  "(Stochastic) Gentle AdaBoost (NOT WEAK)",
  "(Stochastic) AdaBoost (WEAK CLASSIFIERS)",
  "(Stochastic) LogitBoost (WEAK CLASSIFIERS)",
  "(Stochastic) Real AdaBoost (WEAK CLASSIFIERS)",
  "(Stochastic) Gentle AdaBoost (WEAK CLASSIFIERS)",
  "(Stochastic) Gradient Boosting using LOGISTIC Loss",
  "EXTREME GRADIENT BOOSTING - TREES",
  "EXTREME GRADIENT BOOSTING - TREES WITH DROPOUT"
)

load("D:/offside/models_code_SAVED_FINAL_vXX_1_AtoC.RData")
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
load("D:/offside/models_code_SAVED_FINAL_vXX_2_DtoE.RData")
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
load("D:/offside/models_code_SAVED_FINAL_vXX_3_F.RData")
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
load("D:/offside/models_code_SAVED_FINAL_vXX_4_GtoH.RData")
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
load("D:/offside/models_code_SAVED_FINAL_vXX_5_ItoJ.RData")
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
load("D:/offside/models_code_SAVED_FINAL_vXX_6_KtoM.RData")
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

load("D:/offside/models_code_SAVED_FINAL_vXX_RESULTS_ONLY.RData")

save.image("D:/offside/models_code_SAVED_FINAL_vXX_RESULTS_ONLY.RData")

####SUMMARY -- STACKING####

rm(list = ls())
load("D:/offside/models_code_SAVED_FINAL_vXX_RESULTS_ONLY.RData")

stackingmodelNAMES <- c("stacking1CM",
                        "stacking2CM",
                        "stacking3CM",
                        "stacking4CM")

#loading stacking CM and PREDICT
for (i in 1:length(stackingmodelNAMES)) {
  load(
    paste0(
      "D:/offside/models_code_SAVED_FINAL_vXX_stacking",
      i,
      "CM&PREDICT.RData"
    )
  )
}

stackingsummaryMETRICS <- mat.or.vec(length(stackingmodelNAMES), 6)
colnames(stackingsummaryMETRICS) <-
  c("Accuracy",
    "Kappa",
    "Precision/PPV",
    "Recall/Sensitivity",
    "F1",
    "Specificity")
row.names(stackingsummaryMETRICS) <-
  c(
    "Stacking 1 - (TOP 2 - DEPENDING ON VALIDATION) GRF and RF",
    "Stacking 2 - (TOP 3 - DEPENDING ON VALIDATION) GRF and RF and GRRF",
    "Stacking 3 - (TOP 4 - DEPENDING ON VALIDATION) GRF and RF and GRRF and Real AdaBoost",
    "Stacking 4 - (TOP 5 - DEPENDING ON VALIDATION) GRF and RF and GRRF and Real AdaBoost and Gentle AdaBoost"
  )

for (i in 1:nrow(stackingsummaryMETRICS)) {
  stackingsummaryMETRICS[i, 1] = as.numeric(eval(parse(
    text = paste(stackingmodelNAMES[i], "$overall[1]", sep = "")
  ))) #Accuracy
  stackingsummaryMETRICS[i, 2] = as.numeric(eval(parse(
    text = paste(stackingmodelNAMES[i], "$overall[2]", sep = "")
  ))) #Kappa
  stackingsummaryMETRICS[i, 3] = as.numeric(eval(parse(
    text = paste(stackingmodelNAMES[i], "$byClass[5]", sep = "")
  ))) #Precision/Positive Predictive Value
  stackingsummaryMETRICS[i, 4] = as.numeric(eval(parse(
    text = paste(stackingmodelNAMES[i], "$byClass[6]", sep = "")
  ))) #Recall/Sensitivity
  stackingsummaryMETRICS[i, 5] = as.numeric(eval(parse(
    text = paste(stackingmodelNAMES[i], "$byClass[7]", sep = "")
  ))) #F1
  stackingsummaryMETRICS[i, 6] = as.numeric(eval(parse(
    text = paste(stackingmodelNAMES[i], "$byClass[2]", sep = "")
  ))) #Specificity
}

stackingsummaryMETRICS[order(stackingsummaryMETRICS[, 2], decreasing = TRUE), ]

save.image("D:/offside/models_code_SAVED_FINAL_vXX_RESULTS_ONLY.RData")

####MAJORITY VOTING (HARD)#####

rm(list = ls())
load("D:/offside/models_code_SAVED_FINAL_vXX_BASICS.RData")
load("D:/offside/models_code_SAVED_FINAL_vXX_RESULTS_ONLY.RData")

#IN THE BELOW, WE ARE TAKING THE TOP MODELS DEPENDING ON THE THE PERFORMANCE (KAPPA) OF THE VALIDATION SETS (AVERAGE PERFORMANCE)

top <- seq(3, 15, 2) #taking all top odd models, MAXIMUM IS 15
majorityHARDsummaryMETRICS <- mat.or.vec(length(top), 7)
colnames(majorityHARDsummaryMETRICS) <-
  c(
    "TOP",
    "Accuracy",
    "Kappa",
    "Precision/PPV",
    "Recall/Sensitivity",
    "F1",
    "Specificity"
  )
k <- 1

for (j in top) {
  topmodelhardPREDICT <-
    modelNAMESPREDICT[order(summaryMETRICS_validation[, 2], decreasing = TRUE)][1:j] #taking PREDICT of top models
  
  #creating the MAJORITY column
  tophardPREDICT <- do.call("cbind", mget(topmodelhardPREDICT))
  MAJORITY <- integer(nrow(tophardPREDICT))
  tophardPREDICT <- cbind(tophardPREDICT, MAJORITY)
  
  for (i in 1:nrow(tophardPREDICT)) {
    tophardPREDICT[i, "MAJORITY"] <-
      as.numeric(names(table(tophardPREDICT[i, ]))[table(tophardPREDICT[i, ]) == max(table(tophardPREDICT[i, ]))]) #calculating the majority
  }
  
  tophardPREDICT <- as.data.frame(tophardPREDICT)
  tophardPREDICT[, "MAJORITY"] <-
    as.factor(tophardPREDICT[, "MAJORITY"])
  levels(tophardPREDICT[, "MAJORITY"]) <- c("No", "Yes")
  
  temp <- paste0("top", j, "majorityHARDCM")
  assign(temp,
         confusionMatrix(tophardPREDICT[, "MAJORITY"], testing_df$offside, positive =
                           'Yes'))
  
  majorityHARDsummaryMETRICS[k, 1] = j
  majorityHARDsummaryMETRICS[k, 2] = as.numeric(eval(parse(text = paste(
    temp, "$overall[1]", sep = ""
  )))) #Accuracy
  majorityHARDsummaryMETRICS[k, 3] = as.numeric(eval(parse(text = paste(
    temp, "$overall[2]", sep = ""
  )))) #Kappa
  majorityHARDsummaryMETRICS[k, 4] = as.numeric(eval(parse(text = paste(
    temp, "$byClass[5]", sep = ""
  )))) #Precision/Positive Predictive Value
  majorityHARDsummaryMETRICS[k, 5] = as.numeric(eval(parse(text = paste(
    temp, "$byClass[6]", sep = ""
  )))) #Recall/Sensitivity
  majorityHARDsummaryMETRICS[k, 6] = as.numeric(eval(parse(text = paste(
    temp, "$byClass[7]", sep = ""
  )))) #F1
  majorityHARDsummaryMETRICS[k, 7] = as.numeric(eval(parse(text = paste(
    temp, "$byClass[2]", sep = ""
  )))) #Specificity
  k <- k + 1
}

majorityHARDsummaryMETRICS

library(tidyr)
majorityHARDsummaryMETRICS <-
  as.data.frame(majorityHARDsummaryMETRICS)
majorityHARDsummaryMETRICS_pivot <-
  majorityHARDsummaryMETRICS %>% pivot_longer(!TOP, names_to = "metric", values_to = "value")

save.image("D:/offside/models_code_SAVED_FINAL_vXX_RESULTS_ONLY.RData")

####MAJORITY VOTING (SOFT)#####

rm(list = ls())
load("D:/offside/models_code_SAVED_FINAL_vXX_BASICS.RData")
load("D:/offside/models_code_SAVED_FINAL_vXX_RESULTS_ONLY.RData")

#IN THE BELOW, WE ARE TAKING THE TOP MODELS DEPENDING ON THE THE PERFORMANCE (KAPPA) OF THE VALIDATION SETS (AVERAGE PERFORMANCE)

top <- seq(2, 15, 1) #taking all top odd models, MAXIMUM IS 15
majoritySOFTsummaryMETRICS <- mat.or.vec(length(top), 7)
colnames(majoritySOFTsummaryMETRICS) <-
  c(
    "TOP",
    "Accuracy",
    "Kappa",
    "Precision/PPV",
    "Recall/Sensitivity",
    "F1",
    "Specificity"
  )
k <- 1

for (j in top) {
  topmodelsoftPREDICT <-
    modelNAMESprobPREDICT[order(summaryMETRICS_validation[, 2], decreasing = TRUE)][1:j] #taking PREDICT of top models
  
  #creating the AVERAGEPROB column
  topsoftPREDICT <-
    do.call("cbind", lapply(mget(topmodelsoftPREDICT), "[", , "Yes")) #grabbing the probabilities of offside for all models and then combining them to a single data frame
  AVERAGEPROB <- integer(nrow(topsoftPREDICT))
  MAJORITYAVERAGE <- integer(nrow(topsoftPREDICT))
  topsoftPREDICT <- cbind(topsoftPREDICT, AVERAGEPROB, MAJORITYAVERAGE)
  
  topsoftPREDICT[, "AVERAGEPROB"] <-
    rowMeans(topsoftPREDICT[, seq(1, j, 1)])#calculating the average probability
  
  for (i in 1:nrow(topsoftPREDICT)) {
    if (topsoftPREDICT[i, "AVERAGEPROB"] < 0.5)
      topsoftPREDICT[i, "MAJORITYAVERAGE"] <- 0
    if (topsoftPREDICT[i, "AVERAGEPROB"] >= 0.5)
      topsoftPREDICT[i, "MAJORITYAVERAGE"] <- 1
  }
  
  topsoftPREDICT <- as.data.frame(topsoftPREDICT)
  topsoftPREDICT[, "MAJORITYAVERAGE"] <-
    as.factor(topsoftPREDICT[, "MAJORITYAVERAGE"])
  levels(topsoftPREDICT[, "MAJORITYAVERAGE"]) <- c("No", "Yes")
  
  temp <- paste0("top", j, "majoritySOFTCM")
  assign(temp,
         confusionMatrix(topsoftPREDICT[, "MAJORITYAVERAGE"], testing_df$offside, positive =
                           'Yes'))
  
  majoritySOFTsummaryMETRICS[k, 1] = j
  majoritySOFTsummaryMETRICS[k, 2] = as.numeric(eval(parse(text = paste(
    temp, "$overall[1]", sep = ""
  )))) #Accuracy
  majoritySOFTsummaryMETRICS[k, 3] = as.numeric(eval(parse(text = paste(
    temp, "$overall[2]", sep = ""
  )))) #Kappa
  majoritySOFTsummaryMETRICS[k, 4] = as.numeric(eval(parse(text = paste(
    temp, "$byClass[5]", sep = ""
  )))) #Precision/Positive Predictive Value
  majoritySOFTsummaryMETRICS[k, 5] = as.numeric(eval(parse(text = paste(
    temp, "$byClass[6]", sep = ""
  )))) #Recall/Sensitivity
  majoritySOFTsummaryMETRICS[k, 6] = as.numeric(eval(parse(text = paste(
    temp, "$byClass[7]", sep = ""
  )))) #F1
  majoritySOFTsummaryMETRICS[k, 7] = as.numeric(eval(parse(text = paste(
    temp, "$byClass[2]", sep = ""
  )))) #Specificity
  k <- k + 1
}

majoritySOFTsummaryMETRICS

library(tidyr)
majoritySOFTsummaryMETRICS <-
  as.data.frame(majoritySOFTsummaryMETRICS)
majoritySOFTsummaryMETRICS_pivot <-
  majoritySOFTsummaryMETRICS %>% pivot_longer(!TOP, names_to = "metric", values_to = "value")

save.image("D:/offside/models_code_SAVED_FINAL_vXX_RESULTS_ONLY.RData")

#####Printing k-fold cross-validation results####

rm(list = ls())

library(writexl)

modelNAMESMODEL <- c(
  "cartMODEL",
  "rfMODEL",
  "rrfMODEL",
  "grrfMODEL",
  "grfMODEL",
  "adaMODEL",
  "realadaMODEL",
  "gentleadaMODEL",
  "adaWEAKMODEL",
  "logitWEAKMODEL",
  "realadaWEAKMODEL",
  "gentleadaWEAKMODEL",
  "gbmlogMODEL",
  "xgbTreeMODEL",
  "xgbDARTMODEL",
  "xgbDARTMODEL_min"
)

load("D:/offside/models_code_SAVED_FINAL_vXX_1_AtoC.RData")
for (i in 1:length(modelNAMESMODEL)) {
  tryCatch({
    #this is to ignore errors like MODEL not found
    tempresults <-
      as.data.frame(eval(parse(
        text = paste(modelNAMESMODEL[i], "$results", sep = "")
      )))
    temppath <-
      paste("D:/offside/cross_val_results/",
            modelNAMESMODEL[i],
            "_results.xlsx",
            sep = "")
    write_xlsx(tempresults, temppath)
  }, error = function(e) {
  }) #this is to ignore errors like MODEL not found
}
rm(list = ls()[!(ls() %in% c("modelNAMESMODEL"))])
load("D:/offside/models_code_SAVED_FINAL_vXX_2_DtoE.RData")
for (i in 1:length(modelNAMESMODEL)) {
  tryCatch({
    #this is to ignore errors like MODEL not found
    tempresults <-
      as.data.frame(eval(parse(
        text = paste(modelNAMESMODEL[i], "$results", sep = "")
      )))
    temppath <-
      paste("D:/offside/cross_val_results/",
            modelNAMESMODEL[i],
            "_results.xlsx",
            sep = "")
    write_xlsx(tempresults, temppath)
  }, error = function(e) {
  }) #this is to ignore errors like MODEL not found
}
rm(list = ls()[!(ls() %in% c("modelNAMESMODEL"))])
load("D:/offside/models_code_SAVED_FINAL_vXX_3_F.RData")
for (i in 1:length(modelNAMESMODEL)) {
  tryCatch({
    #this is to ignore errors like MODEL not found
    tempresults <-
      as.data.frame(eval(parse(
        text = paste(modelNAMESMODEL[i], "$results", sep = "")
      )))
    temppath <-
      paste("D:/offside/cross_val_results/",
            modelNAMESMODEL[i],
            "_results.xlsx",
            sep = "")
    write_xlsx(tempresults, temppath)
  }, error = function(e) {
  }) #this is to ignore errors like MODEL not found
}
rm(list = ls()[!(ls() %in% c("modelNAMESMODEL"))])
load("D:/offside/models_code_SAVED_FINAL_vXX_4_GtoH.RData")
for (i in 1:length(modelNAMESMODEL)) {
  tryCatch({
    #this is to ignore errors like MODEL not found
    tempresults <-
      as.data.frame(eval(parse(
        text = paste(modelNAMESMODEL[i], "$results", sep = "")
      )))
    temppath <-
      paste("D:/offside/cross_val_results/",
            modelNAMESMODEL[i],
            "_results.xlsx",
            sep = "")
    write_xlsx(tempresults, temppath)
  }, error = function(e) {
  }) #this is to ignore errors like MODEL not found
}
rm(list = ls()[!(ls() %in% c("modelNAMESMODEL"))])
load("D:/offside/models_code_SAVED_FINAL_vXX_5_ItoJ.RData")
for (i in 1:length(modelNAMESMODEL)) {
  tryCatch({
    #this is to ignore errors like MODEL not found
    tempresults <-
      as.data.frame(eval(parse(
        text = paste(modelNAMESMODEL[i], "$results", sep = "")
      )))
    temppath <-
      paste("D:/offside/cross_val_results/",
            modelNAMESMODEL[i],
            "_results.xlsx",
            sep = "")
    write_xlsx(tempresults, temppath)
  }, error = function(e) {
  }) #this is to ignore errors like MODEL not found
}
rm(list = ls()[!(ls() %in% c("modelNAMESMODEL"))])
load("D:/offside/models_code_SAVED_FINAL_vXX_6_KtoM.RData")
for (i in 1:length(modelNAMESMODEL)) {
  tryCatch({
    #this is to ignore errors like MODEL not found
    tempresults <-
      as.data.frame(eval(parse(
        text = paste(modelNAMESMODEL[i], "$results", sep = "")
      )))
    temppath <-
      paste("D:/offside/cross_val_results/",
            modelNAMESMODEL[i],
            "_results.xlsx",
            sep = "")
    write_xlsx(tempresults, temppath)
  }, error = function(e) {
  }) #this is to ignore errors like MODEL not found
}
rm(list = ls()[!(ls() %in% c("modelNAMESMODEL"))])
gc(reset = TRUE)

#writing prediction error of random forest
load("D:/offside/models_code_SAVED_FINAL_vXX_1_AtoC.RData")
rfMODEL_errorrate <-
  as.data.frame(cbind(1:500, rfMODEL[["finalModel"]][["err.rate"]][, 1]))
colnames(rfMODEL_errorrate) <- c("iter", "error_rate")
write_xlsx(rfMODEL_errorrate,
           "D:/offside/cross_val_results/rfMODEL_errorrate.xlsx")
rm(list = ls())
gc(reset = TRUE)
