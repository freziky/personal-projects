#Before we proceed to the edX code, we may meet an issue on the downloads, so, this line of code helped me overcome this issue
options(timeout = 3600)
##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
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
####################End of the edX Code ########################################
####################Beginning of my code ###########################
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-projects.org")
library(lubridate)
library(matrixStats)
##Statistics and data visualization
#About the data sets
str(edx)
str(validation)
#Further variable extraction
edx <- edx %>% mutate(rate_year=year(as_datetime(timestamp)), 
               release=parse_number(str_extract(title, pattern="\\(\\d{4}\\)$")))
validation <- validation %>% mutate(rate_year=year(as_datetime(timestamp)),
                                    release=parse_number(str_extract(title, pattern = "\\(\\d{4}\\)$")))
#Basic statistical parameters
edx %>% summarize(avg=mean(rating), se= sd(rating))
#Variable statistics and distribution
edx %>% group_by(userId) %>% summarize(avg=mean(rating))%>% ggplot(aes(x=avg))+
  geom_histogram(binwidth = .1)+
  labs(title="Rating by user")
edx %>% group_by(movieId) %>% summarize(avg=mean(rating)) %>% ggplot(aes(x=avg))+
  geom_histogram(binwidth = .1)+
  labs(title="Rating by movie")
p <- edx %>% separate_rows(genres, sep= "\\|") 
p %>% ggplot(aes(x=genres, y=rating))+
  geom_boxplot()+theme(axis.text.x = element_text(angle = 90))+
  labs(title = "Rating per genre distribution")
edx %>% group_by(rate_year) %>% summarize(avg=mean(rating), count=n()) %>% filter(count>=100)  %>%
  ggplot(aes(rate_year, avg))+
  geom_point()+
  geom_smooth(method = "lm")+
  geom_smooth(col="green")
edx %>% group_by(release) %>% summarize(avg=mean(rating), count=n()) %>% filter(count>=100)  %>%
  ggplot(aes(release, avg))+
  geom_point()+
  geom_smooth(method = "lm")+
  geom_smooth(col="red")

###Machine learning part
##Predicting using the average value: baseline scenario
RMSE(mean(edx$rating), validation$rating)
avg <- mean(edx$rating)
avg
##Finding the remaining variability using the discrete variables
user_bias <- edx %>% group_by(userId) %>% summarize(ubias=mean(rating-avg))
validation <- user_bias %>% right_join(validation, by="userId")
edx <- left_join(edx, user_bias, by="userId")
RMSE(avg+validation$ubias, validation$rating)
movie_bias <- edx %>% group_by(movieId) %>% summarize(mbias=mean(rating-avg-ubias))
validation <- validation %>% left_join(., movie_bias)
edx <- left_join(edx, movie_bias, by="movieId")
RMSE(avg+validation$ubias+validation$mbias, validation$rating)
genre_average <- p %>% group_by(genres) %>% summarize(avg=mean(rating-avg-ubias-mbias))
genre_average
##Regression using discrete variables
edx <-edx %>% mutate(yeffect = rating-avg-mbias-ubias)
validation <- validation %>% mutate(yeffect = rating-avg-mbias-ubias)
edx %>% group_by(rate_year) %>% summarize(yeffects=mean(yeffects)) %>%ggplot(aes(rate_year, yeffects))+
  geom_point()+
  geom_smooth(method="lm", col="green")+
  geom_smooth(col="red")
edx %>% group_by(release) %>% summarize(yeffects=mean(yeffects)) %>% ggplot(aes(release, yeffects))+
  geom_point()+
  geom_smooth(method="lm", col="blue")+
  geom_smooth(col="pink")
#Linear regression
fit_lm <- lm(yeffect ~ rate_year , data=edx)
pred_lm <- predict(fit_lm, validation)+validation$mbias+validation$ubias+avg
rm(fit_lm)
RMSE(pred_lm, validation$rating)
#Polynomial regression
fit_polylm <- lm(yeffect~poly(rate_year, 4), data=edx)
pred_polylm <- predict(fit_polylm, validation)+avg+validation$ubias+validation$mbias
rm(fit_polylm)
RMSE(pred_polylm, validation$rating)
#Loess regression
if(!require(gam)) install.packages("gam")
library(gam)
x <- edx %>% select(rate_year)
y <- edx$yeffect
fit_loess <- train(x, y, method="gamLoess")
pred_loess <- predict(fit_loess, validation)+avg+validation$ubias+validation$mbias
rm(fit_loess)
RMSE(pred_loess, validation$rating)
#Neural network regression
if(!require(brnn)) install.packages("brnn")
library(brnn)
fit_nnet <- train(x, y, method="brnn")
pred_nnet <- predict(fit_nnet, validation)+avg+validation$ubias+validation$mbias
rm(fit_nnet)
RMSE(pred_nnet, validation$rating)
#Final model
pred <- (pred_lm+pred_polylm+pred_loess+pred_nnet)/4
result <- RMSE(pred, validation$rating)
result