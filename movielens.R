#This code directly computes the final RMSE using validation set.
#It could take 20 minutes or more to finish the computation of this code.
#Please refer to the movielens.rmd or movielens.pdf for the codes used to reach the model conclusion.

#Load libraries
library(tidyverse)
library(data.table)
library(caret)
library(lubridate)
library(stringr)
library(recosystem)

#Download data 
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

#Extract key variables from data files
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")

#Generate edx and validation set. edx is 10% of movielens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#Split edx into train set and test set
#Test set will be 10% of edx set
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

#Make sure userId and movieId in validation set are also in train set 
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

#Average rating
mu <- mean(train_set$rating) 

#Loss Function
RMSE <- function(true, predicted){
  sqrt(mean((true - predicted)^2, na.rm = TRUE))
}

#Movie effect b_i
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

#User effect b_u
user_avgs <- train_set %>% 
  left_join(movie_avgs, by = "movieId") %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating - mu - b_i))

#Regularized using lambda = 5 (refer to movielens.rmd or movielens.pdf for tuning process)
l = 5
b_ir <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n() + l))
b_ur <- train_set %>% 
  left_join(b_ir, by = "movieId") %>% 
  group_by(userId) %>% 
  summarize(b_u = sum(rating - mu - b_i)/(n() + l))

#Matrix factorisation
#Compute model residual
model_residual <- train_set %>% 
  left_join(b_ir, by = "movieId") %>% 
  left_join(b_ur, by = "userId") %>%
  mutate(residual = rating - mu - b_i - b_u) %>%
  select(userId, movieId, residual)

#Base prediction on validation set
y_hat_v <- validation %>%
  left_join(b_ir, by = "movieId") %>%
  left_join(b_ur, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>% pull(pred)

#Convert data to matrices then save to disk
mf_train <- as.matrix(model_residual)
mf_validation <- validation %>% select(userId, movieId, rating)
mf_validation <- as.matrix(mf_validation)

write.table(mf_train, file = "mf_train.txt", 
            sep = " ", row.names = FALSE, col.names = FALSE)
write.table(mf_validation, file = "mf_validation.txt", 
            sep = " ", row.names = FALSE, col.names = FALSE)

set.seed(2019, sample.kind = "Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead

#Loading files for recosystem
mftrain_set <- data_file("mf_train.txt")
mfvalidation_set <- data_file("mf_validation.txt")

#Loading reco object
r <- Reco()

#Training train_set
opts <- r$tune(mftrain_set, opts = list(dim = c(10, 20, 30), lrate = c(0.1, 0.2),
                                        costp_l1 = 0, costq_l1 = 0,
                                        nthread = 1, niter = 10))
r$train(mftrain_set, opts = c(opts$min, nthread = 1, niter = 20))

#Save prediction into temp file
predict_file_validation <- tempfile()
r$predict(mfvalidation_set, out_file(predict_file_validation))
residuals_hat_mf_validation <- scan(predict_file_validation)

#Add predicted residuals back to base predictions
y_hat_mf_validation <- y_hat_v + residuals_hat_mf_validation

#Calculate RMSE
rmse_mf_validation <- RMSE(validation$rating, y_hat_mf_validation)

#Final RMSE value
rmse_mf_validation
