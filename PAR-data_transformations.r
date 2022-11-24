rm(list = ls())
library(moments)
library(spectral)
####txt to data frame####
data_file_paths <-
  dir(path = "D:/activities/raw_data_archive/data",
      full.names = TRUE,
      recursive = TRUE)

col_names <-
  c(
    "T_xacc",
    "T_yacc",
    "T_zacc",
    "T_xgyro",
    "T_ygyro",
    "T_zgyro",
    "T_xmag",
    "T_ymag",
    "T_zmag",
    "RA_xacc",
    "RA_yacc",
    "RA_zacc",
    "RA_xgyro",
    "RA_ygyro",
    "RA_zgyro",
    "RA_xmag",
    "RA_ymag",
    "RA_zmag",
    "LA_xacc",
    "LA_yacc",
    "LA_zacc",
    "LA_xgyro",
    "LA_ygyro",
    "LA_zgyro",
    "LA_xmag",
    "LA_ymag",
    "LA_zmag",
    "RL_xacc",
    "RL_yacc",
    "RL_zacc",
    "RL_xgyro",
    "RL_ygyro",
    "RL_zgyro",
    "RL_xmag",
    "RL_ymag",
    "RL_zmag",
    "LL_xacc",
    "LL_yacc",
    "LL_zacc",
    "LL_xgyro",
    "LL_ygyro",
    "LL_zgyro",
    "LL_xmag",
    "LL_ymag",
    "LL_zmag"
  )

df <- NULL
temp_alls <- NULL
i <- 1

for (a in 1:19) {
  for (p in 1:8) {
    for (s in 1:60) {
      temp <-
        read.delim2(
          data_file_paths[i],
          header = FALSE,
          col.names = col_names,
          sep = ","
        )
      temp <- cbind(a, p, s, temp)
      temp_alls <- rbind(temp_alls, temp)
      i = i + 1
    }
    print(paste0("a=", a, ", p=", p))
    df <- rbind(df, temp_alls)
    temp_alls <- NULL
  }
}

write.csv(df, "D:/activities/dataset.csv")
rm(a, i, p, s, col_names)

####time series to cross-sectional####

df <- read.csv("D:/activities/dataset.csv")
df <- df[, -1]

col_names <- colnames(df)
df[col_names] <-
  sapply(df[col_names], as.numeric) #converting all variables to numeric
a_set <- as.integer(names(summary(as.factor(df[, "a"]))))
p_set <- as.integer(names(summary(as.factor(df[, "p"]))))
s_set <- as.integer(names(summary(as.factor(df[, "s"]))))

df_new_grid <- expand.grid(s = s_set, p = p_set, a = a_set)[3:1]
df_new <- NULL
for (aval in a_set) {
  for (pval in p_set) {
    for (sval in s_set) {
      cond <- which(df[, "a"] == aval & df[, "p"] == pval &
                      df[, "s"] == sval)
      cond2 <-
        which(df_new_grid[, "a"] == aval &
                df_new_grid[, "p"] == pval & df_new_grid[, "s"] == sval)
      
      #mean
      fun_mean <- apply(df[cond, -c(1:3)], 2, mean)
      names(fun_mean) <- paste0(names(fun_mean), "_mean")
      fun_mean <- t(as.data.frame(fun_mean))
      row.names(fun_mean) <- NULL
      
      #min
      fun_min <- apply(df[cond, -c(1:3)], 2, min)
      names(fun_min) <- paste0(names(fun_min), "_min")
      fun_min <- t(as.data.frame(fun_min))
      row.names(fun_min) <- NULL
      
      #max
      fun_max <- apply(df[cond, -c(1:3)], 2, max)
      names(fun_max) <- paste0(names(fun_max), "_max")
      fun_max <- t(as.data.frame(fun_max))
      row.names(fun_max) <- NULL
      
      #skewness
      fun_skew <- apply(df[cond, -c(1:3)], 2, skewness)
      names(fun_skew) <- paste0(names(fun_skew), "_skew")
      fun_skew <- t(as.data.frame(fun_skew))
      row.names(fun_skew) <- NULL
      
      #kurtosis
      fun_kurt <- apply(df[cond, -c(1:3)], 2, kurtosis)
      names(fun_kurt) <- paste0(names(fun_kurt), "_kurt")
      fun_kurt <- t(as.data.frame(fun_kurt))
      row.names(fun_kurt) <- NULL
      
      #dft (top 5 peaks and corresponding frequencies)
      dft <- apply(df[cond, -c(1:3)], 2, spec.fft, center = F)
      n <-
        ceiling(125 / 2) #half the length of the discrete fourier transform (since symmetrical)
      psd <- mat.or.vec(n, length(dft))
      top_psd <- mat.or.vec(5, length(dft))
      top_freq <- mat.or.vec(5, length(dft))
      samfreq = 25
      # sampling frequency (25 Hz) (25 samples per second)
      l = 125
      # length of signal (5 seconds, thus we have 125 samples)
      f = samfreq * (0:(ceiling(l / 2) - 1)) / l
      # frequency
      #or f = dft[[i]][["fx"]]*25[0:ceiling(l/2)] #ANY i since all equal or insert in loop below but slower
      for (i in 1:length(dft)) {
        psd[, i] <- dft[[i]][["PSD"]][1:n]
        top_psd[, i] <- psd[order(-psd[, i])[1:5]]
        top_freq[, i] <- f[order(-psd[, i])[1:5]]
      }
      fun_dft <- NULL
      fun_dft_freq <- NULL
      for (i in 1:5) {
        fun_dft <- c(fun_dft, top_psd[i, ])
        fun_dft_freq <- c(fun_dft_freq, top_freq[i, ])
      }
      names(fun_dft) <- c(
        paste0(names(dft), "_dft1"),
        paste0(names(dft), "_dft2"),
        paste0(names(dft), "_dft3"),
        paste0(names(dft), "_dft4"),
        paste0(names(dft), "_dft5")
      )
      fun_dft <- t(as.data.frame(fun_dft))
      row.names(fun_dft) <- NULL
      names(fun_dft_freq) <- c(
        paste0(names(dft), "_dftfreq1"),
        paste0(names(dft), "_dftfreq2"),
        paste0(names(dft), "_dftfreq3"),
        paste0(names(dft), "_dftfreq4"),
        paste0(names(dft), "_dftfreq5")
      )
      fun_dft_freq <- t(as.data.frame(fun_dft_freq))
      row.names(fun_dft_freq) <- NULL
      
      #autocovariance
      acov <-
        apply(df[cond, -c(1:3)],
              2,
              acf,
              lag.max = 50,
              type = "covariance",
              plot = FALSE)
      req_acov <- mat.or.vec(11, length(acov))
      for (i in 1:length(acov)) {
        req_acov[, i] <- acov[[i]][["acf"]][seq(1, 51, 5)]
      }
      fun_acov <- NULL
      for (i in 1:11) {
        fun_acov <- c(fun_acov, req_acov[i, ])
      }
      names(fun_acov) <- c(
        paste0(names(acov), "_acov0"),
        paste0(names(acov), "_acov5"),
        paste0(names(acov), "_acov10"),
        paste0(names(acov), "_acov15"),
        paste0(names(acov), "_acov20"),
        paste0(names(acov), "_acov25"),
        paste0(names(acov), "_acov30"),
        paste0(names(acov), "_acov35"),
        paste0(names(acov), "_acov40"),
        paste0(names(acov), "_acov45"),
        paste0(names(acov), "_acov50")
      )
      fun_acov <- t(as.data.frame(fun_acov))
      row.names(fun_acov) <- NULL
      
      #combining
      to_add <-
        cbind(
          df_new_grid[cond2, ],
          fun_mean,
          fun_min,
          fun_max,
          fun_skew,
          fun_kurt,
          fun_dft,
          fun_dft_freq,
          fun_acov
        )
      df_new <- rbind(df_new, to_add)
    }
    print(paste0("a=", aval, "/19 p=", pval, "/8"))
  }
}

colnames(df_new)[1:3] <- c("activity", "person", "segment")

save.image("D:/activities/data_transformations_backup.RData")

write.csv(df_new, "D:/activities/dataset_full.csv")
