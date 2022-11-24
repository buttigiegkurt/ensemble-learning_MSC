rm(list = ls())
library(readxl)
library(writexl)
library(stringr)
raw_df <- read_excel("D:/offside/dataset_raw.xlsx")

##FIXING NA OF PLAYERS (WITHOUT GK)

#x-coordinate
no_GK <- !str_detect(names(raw_df), "[GK]")
only_x <- str_detect(names(raw_df), "[x]")
condition_2 <- no_GK & only_x
x_coord <- names(raw_df[, condition_2])
for (i in 1:nrow(raw_df)) {
  if (raw_df[i, "camera_angle"] == 0) {
    raw_df[i, x_coord][is.na(raw_df[i, x_coord])] <-
      2560 #camera_angle=0 implies GOAL POST IS ON THE LEFT
  }
  if (raw_df[i, "camera_angle"] == 1) {
    raw_df[i, x_coord][is.na(raw_df[i, x_coord])] <-
      0 #camera_angle=1 implies GOAL POST IS ON THE RIGHT
  }
}

#y-coordinate
no_GK <- !str_detect(names(raw_df), "[GK]")
only_y <- str_detect(names(raw_df), "[y]")
condition_1 <- no_GK & only_y
y_coord <- names(raw_df[, condition_1])
raw_df[y_coord][is.na(raw_df[y_coord])] <-
  1440 #pixel 1440 kept constant

##FIXING NA OF GK

#x-coordinate
only_GK <- str_detect(names(raw_df), "[GK]")
only_x <- str_detect(names(raw_df), "[x]")
condition_3 <- only_GK & only_x
x_coord_GK <- names(raw_df[, condition_3])
for (i in 1:nrow(raw_df)) {
  if (raw_df[i, "camera_angle"] == 0) {
    raw_df[i, x_coord_GK][is.na(raw_df[i, x_coord_GK])] <-
      0 #camera_angle=0 -> GOAL POST IS ON THE LEFT
  }
  if (raw_df[i, "camera_angle"] == 1) {
    raw_df[i, x_coord_GK][is.na(raw_df[i, x_coord_GK])] <-
      2560 #camera_angle=1 -> GOAL POST IS ON THE RIGHT
  }
}

#y-coordinate
only_GK <- str_detect(names(raw_df), "[GK]")
only_y <- str_detect(names(raw_df), "[y]")
condition_4 <- only_GK & only_y
y_coord_GK <- names(raw_df[, condition_4])
raw_df[y_coord_GK][is.na(raw_df[y_coord_GK])] <-
  720 #pixel 720 kept constant; GK is middle of screen

#removing Image_ID and row_ID
df <- raw_df[, !(names(raw_df) %in% c("Image_ID", "row_ID"))]

write_xlsx(df, "D:/offside/dataset_final_unbalanced.xlsx")
