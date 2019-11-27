# 1. import data
library(foreign)
data0 <- read.dta("./sen108kh.dta")
data1 <- as.matrix(data0[1:nrow(data0),10:ncol(data0)]) # response data
prop9 <- sort(apply(data1, 2, FUN = function(x) {sum(x == 9)/nrow(data1)}), decreasing = T)

# 2. proportion of not voting
att9 <- sort(apply(data1, 1, FUN = function(x) {sum(x == 9)/ncol(data1)}), decreasing = T)
data2 <- data0[names(att9[att9 < 0.05]), 10:ncol(data0)]

# 3. split training set and test set
index_all <- expand.grid(1:nrow(data2),1:ncol(data2))
set.seed(1)
test_index <- sample(1:nrow(index_all), 0.2*nrow(index_all), replace = F)
test <- index_all[test_index, ]
train <- index_all[setdiff(1:nrow(index_all), test_index),]

# 4. mapping training/test matrix and y vector
train_input <- data2
set.seed(12)
train_input[train_input == 9] <- NA
train_input[train_input == 6] <- 0
test_true <- numeric(nrow(test))

# 5. generate input matrix and true test value
for (i in 1:nrow(test)){
  test_true[i] <- train_input[test[i,1], test[i,2]]
  train_input[test[i,1], test[i,2]] <- NA
}

# 6. DJML with K=2
library(mirtjml)
train_input <- as.matrix(train_input)
res <- mirtjml_expr(train_input, K = 2)  # CJML
pred_res <- matrix(NA, nrow = nrow(train_input), ncol = ncol(train_input))
f <- function(x) {exp(x)/(1 + exp(x))}  # sigmoid

# 7. predict the test set
for (i in 1:nrow(train_input)) {
  for (j in 1:ncol(train_input)) {
    pred_res[i,j] <- f(res$d_hat[j,] + t(res$A_hat[j,]) %*% res$theta_hat[i,])
  }
}
pred <- numeric(nrow(test))
for (i in 1:nrow(test)){
  pred[i] <- pred_res[test[i,1], test[i,2]]
}
pred_class <- ifelse(pred >= 0.5, 1, 0)  # set the threshold as 0.5

# 8. compute accuracy
sum(na.omit(pred_class == test_true))/length(pred_class)

# 9. the result of section 8 is to compare with result in neural network model (details in .py file)

