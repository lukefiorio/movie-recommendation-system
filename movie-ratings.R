
#### install/load packages ####
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(gam)) install.packages("gam", repos = "http://cran.us.r-project.org")
if(!require(scales)) install.packages("scales", repos = "http://cran.us.r-project.org")

#### set options ####

options(digits = 5)
options(pillar.sigfig = 6) # show more sig fig in group_by

#### download zipped data and combine movie/rating datasets ####

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

#### data prep ####

# set seed and set aside 10% of MovieLens for Validation set
set.seed(1, sample.kind="Rounding")
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

# remove temp objects/variables
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# add date to train data, rounded to nearest month
edx <- edx %>% 
  mutate(date = round_date(as_datetime(timestamp), unit = "month"))

#### data exploration ####

# figure 1: Distribution of movie ratings
edx %>%
  ggplot(aes(rating)) +
  geom_bar() +
  ggtitle("Figure 1: Distribution of Ratings") +
  xlab("Movie Rating") +
  scale_y_continuous(name="# of Ratings", label=comma)

# figure 2: User Effect
edx %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black") +
  ggtitle("Figure 2: User Effect") +
  xlab("Avg Movie Rating Given") +
  scale_y_continuous(name="# of Users", label=comma)

# figure 3: Movie Effect
edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating)) %>% 
  ggplot(aes(b_i)) + 
  geom_histogram(bins = 30, color = "black") +
  ggtitle("Figure 3: Movie Effect") +
  xlab("Avg Movie Rating Received") +
  scale_y_continuous(name="# of Movies", label=comma)

# figure 4: Genre Effect
edx %>% 
  group_by(genres) %>% 
  summarize(b_g = mean(rating)) %>% 
  ggplot(aes(b_g)) + 
  geom_histogram(bins = 30, color = "black") +
  ggtitle("Figure 4: Genres Effect") +
  xlab("Avg Movie Rating") +
  scale_y_continuous(name="# of Genres", label=comma)

# figure 5: Time Effect
edx %>% 
  group_by(date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(date, rating)) +
  geom_point() +
  geom_smooth() +
  ggtitle("Figure 5: Time Effect") +
  xlab("Date") +
  ylab("Movie Rating")

# figure 6: Best/Worst Genres
edx %>%
  group_by(genres) %>%
  summarize(rating = mean(rating), n=n()) %>%
  arrange(rating) %>%
  filter(row_number() <=5 | row_number() >= n()-4) %>%
  mutate(genres_label = paste0(genres, ' [n = ',format(n, big.mark=",", trim=TRUE),']')) %>%
  ggplot(aes(reorder(genres_label, rating), rating, group=genres_label)) +
  geom_bar(stat='identity', width = 0.6, color = 'black') +
  ggtitle("Figure 6: Best/Worst Genres") +
  scale_x_discrete(name = "Genre [sample size]", position = "top") +
  ylab("Avg Rating") +
  theme(axis.text.x = element_text(size = 10),
        axis.title.x = element_text(size = 12),
        axis.text.y = element_text(size = 8),
        axis.title.y = element_text(size = 12),
        plot.margin = unit(c(0,0.5,0,0.5),"cm")
  ) +
  coord_flip()

# figure 7: Best/Worst COMMON Genres
edx %>%
  group_by(genres) %>%
  summarize(rating = mean(rating), n=n()) %>%
  filter(n >= 25000) %>%
  arrange(rating) %>%
  filter(row_number() <=5 | row_number() >= n()-4) %>%
  mutate(genres_label = paste0(genres, ' [n = ',format(n, big.mark=",", trim=TRUE),']')) %>%
  ggplot(aes(reorder(genres_label, rating), rating, group=genres_label)) +
  geom_bar(stat='identity', width = 0.6, color = 'black') +
  ggtitle("Figure 7: Best/Worst Common Genres") +
  scale_x_discrete(name = "Genre [sample size]", position = "top") +
  ylab("Avg Rating") +
  theme(axis.text.x = element_text(size = 10),
        axis.title.x = element_text(size = 12),
        axis.text.y = element_text(size = 8),
        axis.title.y = element_text(size = 12),
        plot.margin = unit(c(0,0.5,0,0.5),"cm")
  ) +
  coord_flip()


#### functions ####

# function to calculate RMSE
RMSE <- function(actuals, predictions){
  sqrt(mean((actuals - predictions)^2))
}

# function to calculate RSS
RSS <- function(actuals, predictions) {
  sum((actuals - predictions)^2)
}

# function to make & evaluate movie rating predictions using regularization
# input training data, testing data, and parameter values (lambda, span)
predict_ratings <- function(train_data, new_data, lambda, span) {
  
  # avg rating in training data
  mu_hat <- mean(train_data$rating)
  
  # movie effect (regularized)
  b_i <- train_data %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu_hat)/(n()+lambda))
  
  # user effect (regularized)
  b_u <- train_data %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu_hat)/(n()+lambda))
  
  # genre effect (regularized)
  b_g <- train_data %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - mu_hat)/(n()+lambda))
  
  ## time effect (smoothed model, using gam)
  
  # append movie, user, genres effect to training data
  train_data <- train_data %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_g, by="genres")
  
  # train model to estimate smoothed time effect -- controlling for movie, user, genre effects
  train_loess_b_t <- train(
    rating - b_i - b_u - b_g ~ date, # control for the other effects
    method = "gamLoess",
    tuneGrid=data.frame(span = span, degree = 1),
    trControl = trainControl(method = "none"),
    data = train_data)
  
  # drop movie, user, genre effect columns from training data
  train_data <- within(train_data, rm(b_i, b_u, b_g))
  
  # use the TRAINED gam model to make a PREDICTION for each date in the test data
  # note: we have FINISHED the gam model training.  this is NOT using the test data to train the model.
  # need to `predict` rather than `left_join` because not all dates in the test_set are also in train_set
  new_data['b_t_hat'] <- predict(train_loess_b_t, newdata=new_data)
  
  # calculate b_t using smoothed values from trained model
  # note: do NOT regularize the time effect >> it's already smoothed (no need to apply a penalty to smoothed estimates)
  b_t <- new_data %>%
    group_by(date) %>%
    summarize(b_t = sum(b_t_hat - mu_hat)/n())
  
  # make predictions on test data
  predicted_ratings <- 
    new_data %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_t, by = "date") %>%
    mutate(pred = mu_hat + b_i + b_u + b_g + b_t) %>%
    pull(pred)
  
  # return parameter values and accuracy stats
  # note: the RSS is used later to calculate the pooled RMSE in k-fold crossvalidation
  return(data.frame(
    lambda=lambda, 
    span=span,
    rmse=RMSE(new_data$rating, predicted_ratings),
    rss=RSS(new_data$rating, predicted_ratings), 
    n=nrow(new_data)
  ))
  
}

# function to split train_set into folds for crossvalidation...
# ...and then pass to `predict_ratings()` to make/evaluate predictions
crossvalidate <- function(data, fold_index, lambda, span) {
  
  # designate the training folds and test fold (based on `fold_index` parameter)
  train_folds <- data.frame(data[folds!=fold_index,])
  cv_fold_temp <- data.frame(data[folds==fold_index,])
  
  # only keep rows that have a corresponding movie, user in the training folds
  cv_fold <- cv_fold_temp %>% 
    semi_join(train_folds, by = "movieId") %>%
    semi_join(train_folds, by = "userId")
  
  # add removed rows back to training folds
  filtered <- anti_join(cv_fold_temp, cv_fold)
  train_folds <- rbind(train_folds, filtered)
  
  # remove temporary variables
  rm(cv_fold_temp, filtered)
  
  # pass the train & test fold to make/evaluate predictions
  accuracy <- predict_ratings(train_folds, cv_fold, lambda, span)
  accuracy <- cbind(accuracy, fold_index = fold_index) # append the fold_index value to the accuracy stats
  
  return(accuracy)
}

# function to calcuate RMSE for each crossvalidated tune
# expected input: matrix, where:  rows == each parameter/accuracy stat; columns == each tune loop
summarize_cv_stats <- function (cv_stats_matrix) {
  
  cv_stats <- data.frame(t(cv_stats_matrix)) # transpose matrix and convert to df
  cv_stats <- data.frame(sapply(cv_stats, unlist)) # convert each column to a vector and again save as df

  # group the crossvalidated data by matching lambda, span
  # then calculate the pooled rmse for each (lambda, span) pair
  cv_summary <-
    cv_stats %>%
    group_by(lambda, span) %>%
    summarize(
      pooled_rmse = sqrt(sum(rss)/sum(n))
    )
  
  # return df with columns: lambda, span, pooled_rmse
  return(cv_summary)
}


#### train the model ####

# create folds on train data for k-fold cross validation
k <- 5 # number of folds
folds <- createFolds(y = edx$rating, k = k, list = FALSE) # give each row an index (representing its fold #)

# specify tune grid for crossvalidation
grid <- expand.grid(
  fold_index = seq(from=1, to=k, by=1),
  lambda = seq(from=4, to=6, by=1),
  span = seq(from=0.05, to=0.15, by=0.05))

# call `crossvalidate()` and store results for each tune/fold
cv_fold_stats <- mapply(crossvalidate, 
                        fold_index=grid$fold_index, 
                        lambda=grid$lambda,
                        span=grid$span, 
                        MoreArgs = list(data=edx)
                        )

# summarize croosvalidated results for each tune
cv_results <- summarize_cv_stats(cv_fold_stats)


## visualize RMSE by each parameter

# figure 8: RMSE by lambda
cv_results %>%
  ggplot(aes(lambda, pooled_rmse, color = factor(span))) +
  geom_point() +
  ggtitle("Figure 8: Crossvalidated RMSE\nby lambda (color=span)") +
  xlab("lambda") +
  ylab("Pooled RMSE") +
  labs(color = "span") +
  theme(plot.title = element_text(size=14),
        plot.margin = unit(c(1,0.5,0,0.5),"cm"))

# figure 9: RMSE by span
cv_results %>%
  ggplot(aes(span, pooled_rmse, color = factor(lambda))) +
  geom_point() +
  ggtitle("Figure 9: Crossvalidated RMSE\nby span (color=lambda)") +
  xlab("span") +
  ylab("Pooled RMSE") +
  labs(color = "lambda") +
  theme(plot.title = element_text(size=14),
        plot.margin = unit(c(1,0.5,0,0.5),"cm"))

#### apply model to validation data ####

# store best combination of parameter values
best_index <- which.min(cv_results$pooled_rmse) # index of best model
best_lambda <- cv_results$lambda[best_index]
best_span <- cv_results$span[best_index]

# add date to validation data, rounded to nearest month
validation <- 
  validation %>% 
  mutate(date = round_date(as_datetime(timestamp), unit = "month"))

# make final predictions and calculate rmse
test_results <- predict_ratings(edx, validation, best_lambda, best_span)

# print rmse: 0.86441
test_results$rmse
