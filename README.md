# Movie Recommendation System
### edX Data Science Final Project - Movie Rating Prediction

This project uses data from the GroupLens research lab on movie ratings to build a movie recommendation system.  The data can be found and downloaded here: 
- http://files.grouplens.org/datasets/movielens/ml-10m.zip

Our goal is to make a movie recommendation system by predicting, as accurately as possible, how users will rate movies. Specifically, we aim to build a model that predicts users' movie ratings with minimal root-mean-square error (RMSE).

**This repo contains 3 files:**
| File              | Description |
| ----------------- | ----------- |
| movie-ratings.R   | The R code used to process the data and build our predictive models |
| movie-ratings.Rmd | The R Markdown file, which produces the full PDF report with narrative |
| movie-ratings.pdf | The final report, knitted from the RMD syntax |

**Problem Description:**

The dataset that we use to build and test our model contains approximately 10 million movie ratings made by nearly 70,000 different users for more than 10,000 different movies. Our data has 1 target variable (`rating`) and 5 predictors (`userId`, `movieId`, `timestamp`, `title`, `genres`).

Movie ratings range from 0.5 up to 5, in half-point (0.5) increments. Each row in our data represents a movie rating (`rating`), given by a specific user (`userId`) for a specific movie (`movieId`).  The movie's title (`title`) and genre(s) (`genres`) are also provided. The `timestamp` field indicates when the movie rating was given.
