# =====================================================================
# THE ELEMENTS OF STATISTICAL LEARNING — SECTION 2.3
# REPRODUCTION OF FIGURES 2.1–2.4 Assignement 
#
# Codes for:
#   Scenario 1: one Gaussian distribution per class
#   Scenario 2: a mixture of 10 Gaussian distributions per class
# =====================================================================


# =====================================================================
# 0. PACKAGES
# =====================================================================

# Install these packages once if they are not already installed.
# install.packages("MASS")
# install.packages("class")

# MASS provides mvrnorm(), which generates multivariate normal data.
library(MASS)

# class provides knn(), which fits k-nearest-neighbor classifiers.
library(class)


# =====================================================================
# 1. COMMON SETTINGS
# =====================================================================

# Colors used throughout the figures.
blue      <- "#56B4E9"   # Color for class 0
orange    <- "#E69F00"   # Color for class 1
train_col <- "#5DB7C2"   # Color for training-error curve
test_col  <- "#E99B4A"   # Color for test-error curve
bayes_col <- "#8B5AA5"   # Color for Bayes-error line

# Number of training observations generated for EACH class.
n_train_class <- 100

# Number of independent test observations generated for EACH class.
n_test_class <- 5000

# Total training sample size:
# 100 observations from class 0 + 100 observations from class 1 = 200.
N <- 200

# Population mean vector for class 0 (blue).
mu0 <- c(1, 0)

# Population mean vector for class 1 (orange).
mu1 <- c(0, 1)


# =====================================================================
# 2. HELPER FUNCTION: BIVARIATE NORMAL DENSITY
# =====================================================================

# This function evaluates a multivariate normal density.
# It is used later to construct the Bayes classifier.
dmvnorm2 <- function(x, mean, Sigma) {

  # Convert the input to a matrix so matrix calculations work consistently.
  x <- as.matrix(x)

  # Subtract the mean vector from every observation.
  # Each row becomes x_i - mean.
  xc <- sweep(x, 2, mean, "-")

  # Calculate the inverse covariance matrix Sigma^{-1}.
  invS <- solve(Sigma)

  # Calculate the quadratic form
  # (x - mean)' Sigma^{-1} (x - mean)
  # for every observation.
  q <- rowSums((xc %*% invS) * xc)

  # Number of predictor variables.
  # Here p = 2 because we use X1 and X2.
  p <- ncol(x)

  # Normalizing constant of the p-dimensional Gaussian density.
  c0 <- 1 / sqrt((2 * pi)^p * det(Sigma))

  # Return the multivariate normal density value for every row of x.
  c0 * exp(-0.5 * q)
}


# =====================================================================
# 3. K VALUES USED IN FIGURE 2.4
# =====================================================================

# These values move from a very smooth KNN classifier (large k)
# to a very flexible classifier (small k).
ks <- c(151, 101, 69, 45, 31, 21, 11, 7, 5, 3, 1)


# #####################################################################
# #####################################################################
# SCENARIO 1
# ONE GAUSSIAN DISTRIBUTION PER CLASS
# #####################################################################
# #####################################################################


# =====================================================================
# 4. SCENARIO 1 — GENERATE TRAINING AND TEST DATA
# =====================================================================

# Scenario 1 uses the identity covariance matrix:
#
#          [1  0]
# Sigma =  [    ]
#          [0  1]
#
# The zero off-diagonal terms mean X1 and X2 are uncorrelated.
Sigma1 <- diag(2)


# ---------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------

# Fix the random seed so the training sample can be reproduced exactly.
set.seed(20261008)

# Generate 100 observations from N(mu0, Sigma1)
# and 100 observations from N(mu1, Sigma1),
# then stack both classes into one 200 x 2 predictor matrix.
x_s1 <- rbind(
  mvrnorm(n_train_class, mu = mu0, Sigma = Sigma1),
  mvrnorm(n_train_class, mu = mu1, Sigma = Sigma1)
)

# Create the corresponding response labels:
# first 100 observations = class 0,
# next 100 observations = class 1.
y_s1 <- c(
  rep(0, n_train_class),
  rep(1, n_train_class)
)


# ---------------------------------------------------------------------
# Independent test data
# ---------------------------------------------------------------------

# Use a different seed so the test data are independent of the training data.
set.seed(20261009)

# Generate 5,000 test observations from each class.
# Total test sample size = 10,000.
x_test_s1 <- rbind(
  mvrnorm(n_test_class, mu = mu0, Sigma = Sigma1),
  mvrnorm(n_test_class, mu = mu1, Sigma = Sigma1)
)

# Create the true class labels for the 10,000 test observations.
y_test_s1 <- c(
  rep(0, n_test_class),
  rep(1, n_test_class)
)

# Display the class counts in the training sample.
table(y_s1)

# Display the class counts in the test sample.
table(y_test_s1)


# =====================================================================
# 5. SCENARIO 1 — CREATE A PREDICTION GRID
# =====================================================================

# Create 220 evenly spaced X1 values covering the observed training range,
# with an additional margin of 0.5 on each side.
x1_s1 <- seq(
  min(x_s1[, 1]) - 0.5,
  max(x_s1[, 1]) + 0.5,
  length.out = 220
)

# Do the same for X2.
x2_s1 <- seq(
  min(x_s1[, 2]) - 0.5,
  max(x_s1[, 2]) + 0.5,
  length.out = 220
)

# Create every X1-X2 combination.
# This grid is used only to draw classification regions.
# It is NOT the independent test dataset.
grid_s1 <- expand.grid(
  X1 = x1_s1,
  X2 = x2_s1
)


# =====================================================================
# 6. SCENARIO 1 — FIGURE 2.1
# LINEAR REGRESSION OF A 0/1 RESPONSE
# =====================================================================

# Add a column of 1s to the training predictors.
# This creates the design matrix [1, X1, X2],
# where the first column represents the intercept.
X_s1 <- cbind(1, x_s1)

# Calculate the ordinary least-squares estimator:
#
# beta_hat = (X'X)^(-1) X'Y
beta_s1 <- solve(
  t(X_s1) %*% X_s1,
  t(X_s1) %*% y_s1
)

# Add an intercept column to every prediction-grid location
# and calculate the fitted least-squares score.
score_s1 <- as.vector(
  cbind(1, as.matrix(grid_s1)) %*% beta_s1
)

# Convert the continuous least-squares score into a class:
# score > 0.5  -> class 1
# score <= 0.5 -> class 0
class_s1 <- ifelse(
  score_s1 > 0.5,
  1,
  0
)

# Create an empty plotting area.
plot(
  x_s1[, 1],
  x_s1[, 2],
  type = "n",
  xlim = range(x1_s1),
  ylim = range(x2_s1),
  xlab = expression(X[1]),
  ylab = expression(X[2]),
  main = "Scenario 1: Linear Regression of 0/1 Response",
  asp = 1
)

# Color each grid point according to the predicted class.
points(
  grid_s1$X1,
  grid_s1$X2,
  pch = ".",
  col = ifelse(
    class_s1 == 1,
    adjustcolor(orange, alpha.f = 0.30),
    adjustcolor(blue, alpha.f = 0.30)
  )
)

# Add the observed class-0 training observations.
points(
  x_s1[y_s1 == 0, ],
  pch = 1,
  col = blue
)

# Add the observed class-1 training observations.
points(
  x_s1[y_s1 == 1, ],
  pch = 1,
  col = orange
)

# Draw the decision boundary.
#
# The cutoff occurs at:
# beta0 + beta1*X1 + beta2*X2 = 0.5
#
# Solving for X2 gives the intercept and slope below.
abline(
  a = (0.5 - beta_s1[1]) / beta_s1[3],
  b = -beta_s1[2] / beta_s1[3],
  lwd = 1.5
)

# Draw a border around the plot.
box()


# =====================================================================
# 7. SCENARIO 1 — FIGURE 2.2
# 15-NEAREST-NEIGHBOR CLASSIFIER
# =====================================================================

# Classify every prediction-grid location using the 15 nearest
# observations in the training sample.
pred15_s1 <- knn(
  train = x_s1,
  test = as.matrix(grid_s1),
  cl = y_s1,
  k = 15
)

# Convert the predicted factor classes to numeric values,
# then reshape the predictions into a matrix matching the grid.
# The matrix is required by contour().
z15_s1 <- matrix(
  as.numeric(as.character(pred15_s1)),
  nrow = length(x1_s1),
  ncol = length(x2_s1)
)

# Create an empty plotting region.
plot(
  x_s1[, 1],
  x_s1[, 2],
  type = "n",
  xlim = range(x1_s1),
  ylim = range(x2_s1),
  xlab = expression(X[1]),
  ylab = expression(X[2]),
  main = "Scenario 1: 15-Nearest Neighbor Classifier",
  asp = 1
)

# Shade the prediction grid according to the 15-NN prediction.
points(
  grid_s1$X1,
  grid_s1$X2,
  pch = ".",
  col = ifelse(
    as.numeric(as.character(pred15_s1)) == 1,
    adjustcolor(orange, alpha.f = 0.30),
    adjustcolor(blue, alpha.f = 0.30)
  )
)

# Draw the boundary separating class 0 from class 1.
# A contour level of 0.5 marks the transition between 0 and 1.
contour(
  x1_s1,
  x2_s1,
  z15_s1,
  levels = 0.5,
  add = TRUE,
  drawlabels = FALSE,
  lwd = 1.25
)

# Add class-0 training observations.
points(
  x_s1[y_s1 == 0, ],
  pch = 1,
  col = blue
)

# Add class-1 training observations.
points(
  x_s1[y_s1 == 1, ],
  pch = 1,
  col = orange
)

box()


# =====================================================================
# 8. SCENARIO 1 — FIGURE 2.3
# 1-NEAREST-NEIGHBOR CLASSIFIER
# =====================================================================

# 1-NN creates a much more irregular boundary, so use a finer grid.
x1f_s1 <- seq(
  min(x1_s1),
  max(x1_s1),
  length.out = 400
)

x2f_s1 <- seq(
  min(x2_s1),
  max(x2_s1),
  length.out = 400
)

# Create the 400 x 400 = 160,000 prediction locations.
fine_s1 <- expand.grid(
  X1 = x1f_s1,
  X2 = x2f_s1
)

# Predict the class using only the single nearest training observation.
pred1_s1 <- knn(
  train = x_s1,
  test = as.matrix(fine_s1),
  cl = y_s1,
  k = 1
)

# Convert the predictions from factors to numeric 0/1 values.
cl1_s1 <- as.numeric(
  as.character(pred1_s1)
)

# Reshape the predictions into the grid matrix needed by contour().
z1_s1 <- matrix(
  cl1_s1,
  nrow = length(x1f_s1),
  ncol = length(x2f_s1)
)

# Create an empty plotting region.
plot(
  x_s1[, 1],
  x_s1[, 2],
  type = "n",
  xlim = range(x1f_s1),
  ylim = range(x2f_s1),
  xlab = expression(X[1]),
  ylab = expression(X[2]),
  main = "Scenario 1: 1-Nearest Neighbor Classifier",
  asp = 1
)

# Plot every fourth fine-grid point to reduce graphical density.
id_s1 <- seq(
  1,
  nrow(fine_s1),
  by = 4
)

# Shade the background according to the 1-NN prediction.
points(
  fine_s1$X1[id_s1],
  fine_s1$X2[id_s1],
  pch = ".",
  col = ifelse(
    cl1_s1[id_s1] == 1,
    adjustcolor(orange, alpha.f = 0.30),
    adjustcolor(blue, alpha.f = 0.30)
  )
)

# Draw the highly flexible 1-NN decision boundary.
contour(
  x1f_s1,
  x2f_s1,
  z1_s1,
  levels = 0.5,
  add = TRUE,
  drawlabels = FALSE,
  lwd = 1.25
)

# Add class-0 training observations.
points(
  x_s1[y_s1 == 0, ],
  pch = 1,
  col = blue
)

# Add class-1 training observations.
points(
  x_s1[y_s1 == 1, ],
  pch = 1,
  col = orange
)

box()


# =====================================================================
# 9. SCENARIO 1 — FIGURE 2.4
# TRAINING ERROR, TEST ERROR, AND BAYES ERROR
# =====================================================================

# Create empty vectors in which KNN training and test errors will be stored.
tr_s1 <- numeric(length(ks))
te_s1 <- numeric(length(ks))

# Fix the seed because knn() can randomly break ties.
set.seed(20261020)

# Repeat the KNN analysis for every selected value of k.
for (i in seq_along(ks)) {

  # Current number of nearest neighbors.
  k <- ks[i]

  # Predict the classes of the training observations themselves.
  pred_train_s1 <- knn(
    train = x_s1,
    test = x_s1,
    cl = y_s1,
    k = k
  )

  # Training misclassification rate.
  tr_s1[i] <- mean(
    as.numeric(as.character(pred_train_s1)) != y_s1
  )

  # Predict the classes of the independent test observations.
  pred_test_s1 <- knn(
    train = x_s1,
    test = x_test_s1,
    cl = y_s1,
    k = k
  )

  # Test misclassification rate.
  te_s1[i] <- mean(
    as.numeric(as.character(pred_test_s1)) != y_test_s1
  )
}

# Approximate effective KNN degrees of freedom.
#
# df ≈ N / k
df_s1 <- N / ks


# ---------------------------------------------------------------------
# Linear-regression training and test errors
# ---------------------------------------------------------------------

# Predict the training classes from the fitted least-squares model
# and calculate its training error.
lin_tr_s1 <- mean(
  ifelse(
    as.vector(X_s1 %*% beta_s1) > 0.5,
    1,
    0
  ) != y_s1
)

# Apply the same fitted linear model to the independent test sample
# and calculate its test error.
lin_te_s1 <- mean(
  ifelse(
    as.vector(cbind(1, x_test_s1) %*% beta_s1) > 0.5,
    1,
    0
  ) != y_test_s1
)


# ---------------------------------------------------------------------
# Bayes error
# ---------------------------------------------------------------------

# Evaluate the true Scenario-1 class-0 density at each test observation.
d0_s1 <- dmvnorm2(
  x_test_s1,
  mu0,
  Sigma1
)

# Evaluate the true Scenario-1 class-1 density.
d1_s1 <- dmvnorm2(
  x_test_s1,
  mu1,
  Sigma1
)

# With equal class priors, choose class 1 whenever its density is larger.
bayes_pred_s1 <- ifelse(
  d1_s1 > d0_s1,
  1,
  0
)

# Estimate the population Bayes error using the 10,000 test observations.
bayes_s1 <- mean(
  bayes_pred_s1 != y_test_s1
)


# ---------------------------------------------------------------------
# Prepare and draw Figure 2.4
# ---------------------------------------------------------------------

# Order the degrees of freedom from low to high flexibility.
order_s1 <- order(df_s1)

# Choose a y-axis range that contains all relevant error values.
ylim_s1 <- c(
  0,
  max(c(
    tr_s1,
    te_s1,
    lin_tr_s1,
    lin_te_s1
  )) + 0.04
)

# Create an empty log-scaled plotting region.

# Give extra room at the top
par(
  mar = c(5, 5, 3, 2) + 0.1,   # inner margins
  oma = c(0, 0, 1.6, 0)           # outer top margin for main title
)

plot(
  df_s1[order_s1],
  te_s1[order_s1],
  type = "n",
  log = "x",
  xlim = c(1.2, 220),
  ylim = ylim_s1,
  xlab = "Degrees of Freedom - N/k",
  ylab = "Misclassification Error",
  main = "",
  xaxt = "n"
)

# Add the independent test-error curve.
lines(
  df_s1[order_s1],
  te_s1[order_s1],
  type = "b",
  pch = 16,
  lty = 2,
  col = test_col
)

# Add the training-error curve.
lines(
  df_s1[order_s1],
  tr_s1[order_s1],
  type = "b",
  pch = 16,
  lty = 2,
  col = train_col
)

# Add the Bayes-error reference line.
abline(
  h = bayes_s1,
  col = bayes_col,
  lwd = 1.5
)

# Represent the linear model at approximately 3 degrees of freedom:
# intercept + coefficient for X1 + coefficient for X2.
points(
  3,
  lin_tr_s1,
  pch = 15,
  col = train_col,
  cex = 1.1
)

points(
  3,
  lin_te_s1,
  pch = 15,
  col = test_col,
  cex = 1.1
)

# Bottom axis: effective degrees of freedom.
axis(
  1,
  at = c(2, 3, 5, 8, 12, 18, 29, 67, 200)
)

# Top x-axis with K values
axis(
  3,
  at = N / ks,
  labels = ks,
  tick = FALSE,
  line = -0.2,
  cex.axis = 0.9
)

# Label for the top K-axis
mtext(
  "K - Number of Nearest Neighbors",
  side = 3,
  line = 2.0
)

# Main title placed in OUTER margin
mtext(
  "Scenario 1: Training/Test Error vs Flexibility",
  side = 3,
  outer = TRUE,
  line = -0.2,
  font = 2,
  cex = 1.05
)


# Add a legend.
legend(
  "bottomleft",
  legend = c("Train", "Test", "Bayes"),
  col = c(train_col, test_col, bayes_col),
  lty = c(2, 2, 1),
  pch = c(16, 16, NA),
  bty = "n"
)

# Print important Scenario-1 numerical results.
cat("Scenario 1 Bayes test error =", bayes_s1, "\n")
cat("Scenario 1 linear training error =", lin_tr_s1, "\n")
cat("Scenario 1 linear test error =", lin_te_s1, "\n")


# #####################################################################
# #####################################################################
# SCENARIO 2
# MIXTURE OF 10 GAUSSIAN COMPONENTS PER CLASS
# #####################################################################
# #####################################################################


# =====================================================================
# 10. SCENARIO 2 — GENERATE COMPONENT CENTERS
# =====================================================================

# Covariance matrix used to generate the 10 component centers
# around each class-level mean.
Sigma_center <- diag(2)

# Small covariance used for observations around each selected center.
# This is 0.2 * I, or equivalently I / 5.
Sigma_within <- 0.2 * diag(2)

# Fix the seed so the 20 Gaussian centers are reproducible.
set.seed(20261010)

# Generate 10 component centers around the class-0 population mean.
centers0 <- mvrnorm(
  10,
  mu = mu0,
  Sigma = Sigma_center
)

# Generate 10 component centers around the class-1 population mean.
centers1 <- mvrnorm(
  10,
  mu = mu1,
  Sigma = Sigma_center
)


# =====================================================================
# 11. HELPER FUNCTION FOR SCENARIO 2 MIXTURE DATA
# =====================================================================

# This function generates observations from one class-specific
# 10-component Gaussian mixture.
gen_mix <- function(n, centers, Sigma) {

  # Randomly choose one of the available Gaussian centers
  # for each requested observation.
  z <- sample(
    1:nrow(centers),
    size = n,
    replace = TRUE
  )

  # Generate low-variance Gaussian noise around zero.
  noise <- mvrnorm(
    n,
    mu = c(0, 0),
    Sigma = Sigma
  )

  # Add the Gaussian noise to the selected component centers.
  centers[z, , drop = FALSE] + noise
}


# =====================================================================
# 12. SCENARIO 2 — GENERATE TRAINING AND TEST DATA
# =====================================================================

# Fix the seed for the Scenario-2 training sample.
set.seed(20261011)

# Generate 100 training observations from each class mixture.
x_s2 <- rbind(
  gen_mix(
    n_train_class,
    centers0,
    Sigma_within
  ),
  gen_mix(
    n_train_class,
    centers1,
    Sigma_within
  )
)

# Create the corresponding 0/1 training labels.
y_s2 <- c(
  rep(0, n_train_class),
  rep(1, n_train_class)
)

# Fix a different seed for the independent test sample.
set.seed(20261012)

# Generate 5,000 test observations from each class mixture.
x_test_s2 <- rbind(
  gen_mix(
    n_test_class,
    centers0,
    Sigma_within
  ),
  gen_mix(
    n_test_class,
    centers1,
    Sigma_within
  )
)

# Create the true test labels.
y_test_s2 <- c(
  rep(0, n_test_class),
  rep(1, n_test_class)
)

# Confirm the training class sizes.
table(y_s2)

# Confirm the test class sizes.
table(y_test_s2)


# =====================================================================
# 13. SCENARIO 2 — CREATE A PREDICTION GRID
# =====================================================================

# Combine the training observations and the 20 mixture centers
# only to determine suitable plotting limits.
all_s2 <- rbind(
  x_s2,
  centers0,
  centers1
)

# Generate 220 X1 grid values covering the plotted region.
x1_s2 <- seq(
  min(all_s2[, 1]) - 0.5,
  max(all_s2[, 1]) + 0.5,
  length.out = 220
)

# Generate 220 X2 grid values.
x2_s2 <- seq(
  min(all_s2[, 2]) - 0.5,
  max(all_s2[, 2]) + 0.5,
  length.out = 220
)

# Construct the full prediction grid.
grid_s2 <- expand.grid(
  X1 = x1_s2,
  X2 = x2_s2
)


# =====================================================================
# 14. SCENARIO 2 — FIGURE 2.1
# LINEAR REGRESSION OF A 0/1 RESPONSE
# =====================================================================

# Build the Scenario-2 design matrix with an intercept.
X_s2 <- cbind(
  1,
  x_s2
)

# Estimate the least-squares coefficients.
beta_s2 <- solve(
  t(X_s2) %*% X_s2,
  t(X_s2) %*% y_s2
)

# Calculate the fitted least-squares score over the prediction grid.
score_s2 <- as.vector(
  cbind(1, as.matrix(grid_s2)) %*% beta_s2
)

# Convert scores to classes using the 0.5 cutoff.
class_s2 <- ifelse(
  score_s2 > 0.5,
  1,
  0
)

# Create an empty plotting region.
plot(
  x_s2[, 1],
  x_s2[, 2],
  type = "n",
  xlim = range(x1_s2),
  ylim = range(x2_s2),
  xlab = expression(X[1]),
  ylab = expression(X[2]),
  main = "Scenario 2: Linear Regression of 0/1 Response",
  asp = 1
)

# Shade the prediction regions.
points(
  grid_s2$X1,
  grid_s2$X2,
  pch = ".",
  col = ifelse(
    class_s2 == 1,
    adjustcolor(orange, alpha.f = 0.30),
    adjustcolor(blue, alpha.f = 0.30)
  )
)

# Add class-0 training observations.
points(
  x_s2[y_s2 == 0, ],
  pch = 1,
  col = blue
)

# Add class-1 training observations.
points(
  x_s2[y_s2 == 1, ],
  pch = 1,
  col = orange
)

# Show the 10 class-0 mixture centers as filled triangles.
points(
  centers0,
  pch = 17,
  col = blue,
  cex = 1.0
)

# Show the 10 class-1 mixture centers.
points(
  centers1,
  pch = 17,
  col = orange,
  cex = 1.0
)

# Draw the least-squares classification boundary.
abline(
  a = (0.5 - beta_s2[1]) / beta_s2[3],
  b = -beta_s2[2] / beta_s2[3],
  lwd = 1.5
)

box()


# =====================================================================
# 15. SCENARIO 2 — FIGURE 2.2
# 15-NEAREST-NEIGHBOR CLASSIFIER
# =====================================================================

# Classify the Scenario-2 prediction grid using 15-NN.
pred15_s2 <- knn(
  train = x_s2,
  test = as.matrix(grid_s2),
  cl = y_s2,
  k = 15
)

# Convert predictions to numeric values.
cl15_s2 <- as.numeric(
  as.character(pred15_s2)
)

# Reshape predictions into a grid matrix for contour().
z15_s2 <- matrix(
  cl15_s2,
  nrow = length(x1_s2),
  ncol = length(x2_s2)
)

# Create an empty plotting region.
plot(
  x_s2[, 1],
  x_s2[, 2],
  type = "n",
  xlim = range(x1_s2),
  ylim = range(x2_s2),
  xlab = expression(X[1]),
  ylab = expression(X[2]),
  main = "Scenario 2: 15-Nearest Neighbor Classifier",
  asp = 1
)

# Shade the prediction grid.
points(
  grid_s2$X1,
  grid_s2$X2,
  pch = ".",
  col = ifelse(
    cl15_s2 == 1,
    adjustcolor(orange, alpha.f = 0.30),
    adjustcolor(blue, alpha.f = 0.30)
  )
)

# Draw the nonlinear 15-NN decision boundary.
contour(
  x1_s2,
  x2_s2,
  z15_s2,
  levels = 0.5,
  add = TRUE,
  drawlabels = FALSE,
  lwd = 1.25
)

# Add training observations from both classes.
points(
  x_s2[y_s2 == 0, ],
  pch = 1,
  col = blue
)

points(
  x_s2[y_s2 == 1, ],
  pch = 1,
  col = orange
)

box()


# =====================================================================
# 16. SCENARIO 2 — FIGURE 2.3
# 1-NEAREST-NEIGHBOR CLASSIFIER
# =====================================================================

# Build a finer plotting grid because 1-NN produces a highly irregular boundary.
x1f_s2 <- seq(
  min(x1_s2),
  max(x1_s2),
  length.out = 400
)

x2f_s2 <- seq(
  min(x2_s2),
  max(x2_s2),
  length.out = 400
)

# Create 160,000 prediction locations.
fine_s2 <- expand.grid(
  X1 = x1f_s2,
  X2 = x2f_s2
)

# Predict each fine-grid location using its single nearest neighbor.
pred1_s2 <- knn(
  train = x_s2,
  test = as.matrix(fine_s2),
  cl = y_s2,
  k = 1
)

# Convert predicted classes to numeric values.
cl1_s2 <- as.numeric(
  as.character(pred1_s2)
)

# Reshape into a matrix for contour().
z1_s2 <- matrix(
  cl1_s2,
  nrow = length(x1f_s2),
  ncol = length(x2f_s2)
)

# Create an empty plotting region.
plot(
  x_s2[, 1],
  x_s2[, 2],
  type = "n",
  xlim = range(x1f_s2),
  ylim = range(x2f_s2),
  xlab = expression(X[1]),
  ylab = expression(X[2]),
  main = "Scenario 2: 1-Nearest Neighbor Classifier",
  asp = 1
)

# Plot every fourth grid point to keep the background readable.
id_s2 <- seq(
  1,
  nrow(fine_s2),
  by = 4
)

# Shade the background according to 1-NN predictions.
points(
  fine_s2$X1[id_s2],
  fine_s2$X2[id_s2],
  pch = ".",
  col = ifelse(
    cl1_s2[id_s2] == 1,
    adjustcolor(orange, alpha.f = 0.30),
    adjustcolor(blue, alpha.f = 0.30)
  )
)

# Draw the 1-NN decision boundary.
contour(
  x1f_s2,
  x2f_s2,
  z1_s2,
  levels = 0.5,
  add = TRUE,
  drawlabels = FALSE,
  lwd = 1.25
)

# Add the observed training data.
points(
  x_s2[y_s2 == 0, ],
  pch = 1,
  col = blue
)

points(
  x_s2[y_s2 == 1, ],
  pch = 1,
  col = orange
)

box()


# =====================================================================
# 17. SCENARIO 2 — FIGURE 2.4
# TRAINING ERROR, TEST ERROR, AND BAYES ERROR
# =====================================================================

# Allocate vectors for the KNN training and test errors.
tr_s2 <- numeric(length(ks))
te_s2 <- numeric(length(ks))

# Fix the seed because knn() can randomly break ties.
set.seed(20261021)

# Calculate KNN errors for each selected value of k.
for (i in seq_along(ks)) {

  # Current K value.
  k <- ks[i]

  # Classify the training observations.
  pred_train_s2 <- knn(
    train = x_s2,
    test = x_s2,
    cl = y_s2,
    k = k
  )

  # Calculate training misclassification error.
  tr_s2[i] <- mean(
    as.numeric(as.character(pred_train_s2)) != y_s2
  )

  # Classify the independent test observations.
  pred_test_s2 <- knn(
    train = x_s2,
    test = x_test_s2,
    cl = y_s2,
    k = k
  )

  # Calculate independent test error.
  te_s2[i] <- mean(
    as.numeric(as.character(pred_test_s2)) != y_test_s2
  )
}

# Approximate effective degrees of freedom.
df_s2 <- N / ks


# ---------------------------------------------------------------------
# Linear-regression training and test errors
# ---------------------------------------------------------------------

# Scenario-2 linear-model training error.
lin_tr_s2 <- mean(
  ifelse(
    as.vector(X_s2 %*% beta_s2) > 0.5,
    1,
    0
  ) != y_s2
)

# Scenario-2 linear-model test error.
lin_te_s2 <- mean(
  ifelse(
    as.vector(cbind(1, x_test_s2) %*% beta_s2) > 0.5,
    1,
    0
  ) != y_test_s2
)


# ---------------------------------------------------------------------
# Helper function for Scenario-2 mixture densities
# ---------------------------------------------------------------------

# For a given class, the density is the average of the densities
# from its 10 Gaussian components.
mix_density <- function(x, centers, Sigma) {

  # Evaluate the density under each of the 10 component distributions.
  D <- sapply(
    1:nrow(centers),
    function(j) {
      dmvnorm2(
        x,
        centers[j, ],
        Sigma
      )
    }
  )

  # Equal mixing probabilities are used, so the mixture density
  # is simply the row-wise average of the 10 component densities.
  rowMeans(D)
}


# ---------------------------------------------------------------------
# Bayes error for Scenario 2
# ---------------------------------------------------------------------

# True class-0 mixture density at each test observation.
d0_s2 <- mix_density(
  x_test_s2,
  centers0,
  Sigma_within
)

# True class-1 mixture density at each test observation.
d1_s2 <- mix_density(
  x_test_s2,
  centers1,
  Sigma_within
)

# Under equal priors, predict the class with the larger mixture density.
bayes_pred_s2 <- ifelse(
  d1_s2 > d0_s2,
  1,
  0
)

# Estimate the Bayes population error on the large independent test set.
bayes_s2 <- mean(
  bayes_pred_s2 != y_test_s2
)

# ---------------------------------------------------------------------
# Prepare and draw Figure 2.4
# ---------------------------------------------------------------------

# Sort degrees of freedom from low to high flexibility.
order_s2 <- order(df_s2)

# Set the y-axis range.
ylim_s2 <- c(
  0,
  max(c(
    tr_s2,
    te_s2,
    lin_tr_s2,
    lin_te_s2
  )) + 0.04
)

# Create the error-curve plotting area.
# Give extra room at the top
par(
  mar = c(5, 5, 3, 2) + 0.1,   # inner margins
  oma = c(0, 0, 1.6, 0)           # outer top margin for main title
)

plot(
  df_s2[order_s2],
  te_s2[order_s2],
  type = "n",
  log = "x",
  xlim = c(1.2, 220),
  ylim = ylim_s2,
  xlab = "Degrees of Freedom - N/k",
  ylab = "Misclassification Error",
  main = "",
  xaxt = "n"
)

# Add the test-error curve.
lines(
  df_s2[order_s2],
  te_s2[order_s2],
  type = "b",
  pch = 16,
  lty = 2,
  col = test_col
)

# Add the training-error curve.
lines(
  df_s2[order_s2],
  tr_s2[order_s2],
  type = "b",
  pch = 16,
  lty = 2,
  col = train_col
)

# Add the Bayes-error line.
abline(
  h = bayes_s2,
  col = bayes_col,
  lwd = 1.5
)

# Add the linear-model training and test errors at df ≈ 3.
points(
  3,
  lin_tr_s2,
  pch = 15,
  col = train_col,
  cex = 1.1
)

points(
  3,
  lin_te_s2,
  pch = 15,
  col = test_col,
  cex = 1.1
)


# Bottom axis: effective degrees of freedom.
axis(
  1,
  at = c(2, 3, 5, 8, 12, 18, 29, 67, 200)
)

# Top x-axis with K values
axis(
  3,
  at = N / ks,
  labels = ks,
  tick = FALSE,
  line = -0.2,
  cex.axis = 0.9
)

# Label for the top K-axis
mtext(
  "K - Number of Nearest Neighbors",
  side = 3,
  line = 2.0
)

# Main title placed in OUTER margin
mtext(
  "Scenario 2: Training/Test Error vs Flexibility",
  side = 3,
  outer = TRUE,
  line = -0.2,
  font = 2,
  cex = 1.05
)

# legend
legend(
  "bottomleft",
  legend = c("Train", "Test", "Bayes"),
  col = c(train_col, test_col, bayes_col),
  lty = c(2, 2, 1),
  pch = c(16, 16, NA),
  bty = "n"
)

# Print important Scenario-2 numerical results.
cat("Scenario 2 Bayes test error =", bayes_s2, "\n")
cat("Scenario 2 linear training error =", lin_tr_s2, "\n")
cat("Scenario 2 linear test error =", lin_te_s2, "\n")


# =====================================================================
# 18. FINAL COMPARISON TABLE
# =====================================================================

# Combine the most important numerical results from both scenarios.
comparison <- data.frame(

  # Identify the two simulation settings.
  Scenario = c(
    "Scenario 1",
    "Scenario 2"
  ),

  # Linear-classifier training errors.
  Linear_Train_Error = c(
    lin_tr_s1,
    lin_tr_s2
  ),

  # Linear-classifier test errors.
  Linear_Test_Error = c(
    lin_te_s1,
    lin_te_s2
  ),

  # Estimated Bayes errors.
  Bayes_Test_Error = c(
    bayes_s1,
    bayes_s2
  ),

  # K value giving the lowest test error among the tested KNN models.
  Best_KNN_k = c(
    ks[which.min(te_s1)],
    ks[which.min(te_s2)]
  ),

  # Lowest KNN test error obtained in each scenario.
  Best_KNN_Test_Error = c(
    min(te_s1),
    min(te_s2)
  )
)

# Display the final comparison table.
print(comparison)

