# Generate skewed LSNMs:
#
# Datasets are simulated according to https://github.com/tagas/bQCD/blob/master/R/utils/data_generators.R.
# The linked repository is part of the supplementary material for the paper:
# Tagasovska, N., Chavez-Demoulin, V., & Vatter, T. (2020, November). Distinguishing cause from effect
# using quantiles: Bivariate quantile causal discovery. In International Conference on Machine
# Learning (pp. 9311-9323). PMLR.
#
# The code was adapted to include skew-normal and generalized normal errors (AGN / GNO).
# (R version 4.4.2 was used to generate the datasets)

library(MASS)
library(VGAM) # sample from rskewnorm

gauss_kernel <- function(x, sigmay, sigmax) {
  if (is.matrix(x) == FALSE)
    x <- as.matrix(x)

  n <- nrow(x)
  xnorm <- as.matrix(dist(x, method = "euclidean", diag = TRUE, upper = TRUE))

  sigmay * exp(-xnorm^2/(2*sigmax^2))
}

# sample from asymmetric generalized normal distribution (kappa != 0)
ragnorm <- function(n, xi=0, alpha=1, kappa=0.5){
  return(xi + alpha/kappa*(1-exp(-kappa*qnorm(runif(n)))))
}

# 1) AN-s
sample_ANs = function(n, skewness_param, Ey_type, pure_noise = F){
  sample_generative_model(n, skewness_param, 1, 1, Ey_type, pure_noise)
}

# 2) LS-s
sample_LSs = function(n, skewness_param, Ey_type, pure_noise = F){
  sample_generative_model(n, skewness_param, 1, 3, Ey_type, pure_noise)
}

sample_generative_model<- function(n, skewness_param = 1, mech_type = 1, noise_type = 1, Ey_type=1,
                                   pure_noise = FALSE){
  # skewness_param - skewness parameter corresponding to the respective distribution
  #                  (for asymmetric gen. normal: skewness_param != 0)
  # mech_type - 1:injective (sigmoid), 2: non-injective
  # noise_type - 1:additive, 2:multiplicative, 3: ls
  # Ey_type - 1:normal, 2: asymmetric generalized normal. 3: skew-normal

  ran <- rnorm(n)
  noise_exp <- 1
  noise_var <- runif(n, 1, 2)
  noisetmp <- (sqrt(noise_var) * abs(ran))^(noise_exp) * sign(ran)
  x_pa <- noisetmp

  noise_var_ch <- runif(n, 1, 2)

  if(mech_type == 1){
    a.sig <- runif(n=1, min=-2, max=2)
    bern <- rbinom(1,1,0.5)
    b.sig <- bern*runif(n=1, min=0.5, max=2) + (1-bern)*runif(n=1, min=-2, max=-0.5)
    c.sig <- rexp(n=1,rate=4)+1
    x_child <- c.sig*(b.sig*(x_pa+a.sig))/(1+abs(b.sig*(x_pa +a.sig)))
  } else if(mech_type == 2){
    kern_pa <- gauss_kernel(x_pa, 1, 1)
    x_child <- mvrnorm(1, rep(0, n), kern_pa)
  }

  if(Ey_type == 1){ # normal distribution
    ran <- rnorm(n, 0, 1)
  } else if(Ey_type == 2){
    # asymmetric generalized normal distribution
    ran <- ragnorm(n, 0, 1, skewness_param)
  } else if(Ey_type == 3){
    # skewed normal distribution
    ran <- rskewnorm(n, 0, 1, skewness_param)
  } else{
    print("Distribution of Y not implemented yet")
  }
  if(!pure_noise){
    noisetmp <- (0.2 * sqrt(noise_var_ch) * abs(ran)) ^ (noise_exp) * sign(ran)
  } else{
    noisetmp <- (0.2 * abs(ran)) ^ (noise_exp) * sign(ran)
  }

  if (noise_type == 1) {
    x_child <- x_child + noisetmp
  } else if (noise_type == 2) {
    x_child <- x_child * runif(n)
  } else if (noise_type == 3) {
    ran <- rnorm(n)
    x_child <- x_child + (x_child - min(x_child))*noisetmp
  } else if (noise_type == 4) {
    ran <- rnorm(n)
    sd = (x_child - min(x_child))
    x_child <- (0.2 * sqrt(sd) * abs(ran)) ^ (noise_exp) * sign(ran)
  } else {
    print("model type not implemented")
  }
  # standardization:
  # x_pa = (x_pa-mean(x_pa))/sd(x_pa)
  # x_child = (x_child-mean(x_child))/sd(x_child)
  if(!pure_noise){
    return(cbind(x_pa, x_child, noisetmp/(0.2*sqrt(noise_var_ch))))
  } else{
    return(cbind(x_pa, x_child, noisetmp/0.2)) # since 0.2 is absorbed by g(x) anyway
  }
}

# generate 100 pairs of size n = 1000 for each configuration

# for AGN:
# extreme = -0.5 (skewness coeff = 1.750),
# strong = -0.31 (skewness coeff = 0.985),
# medium = 0.15 (skewness coeff = -0.456)
# skewness function is given by
# gamma_AGN = function(kappa){
#   (3*exp(kappa^2) - exp(3*kappa^2) - 2) / (exp(kappa^2) - 1)^(3/2) * sign(kappa)
# }

# for SN:
# strong = 20 (skewness coeff = 0.985),
# medium = -2 (skewness coeff = -0.454)
# skewness function is given by
# gamma_SN = function(alpha){
#   delta = alpha/sqrt(1+alpha^2)
#   return((4-pi)/2 * (delta * sqrt(2/pi))^3 / (1-2*delta^2/pi)^(3/2))
# }

AGN_ex = -0.5
AGN_str = -0.31
AGN_med = 0.15

SN_str = 20
SN_med = -2


# pure noise case, i.e. the noise corresponds exactly to SN/AGN
n = 1000
n_pairs = 100
set.seed(752)

# Redefine functions
# 1) SN:
# ANs with "mediocre" skewness
ANs_SN_med = function(n) {
  sample_ANs(n, skewness_param = SN_med, Ey_type = 3, pure_noise = T)
}
# LSs with "mediocre" skewness
LSs_SN_med = function(n) {
  sample_LSs(n, skewness_param = SN_med, Ey_type = 3, pure_noise = T)
}
# ANs with "strong" skewness
ANs_SN_strong = function(n) {
  sample_ANs(n, skewness_param = SN_str, Ey_type = 3, pure_noise = T)
}
# LSs with "strong" skewness
LSs_SN_strong = function(n) {
  sample_LSs(n, skewness_param = SN_str, Ey_type = 3, pure_noise = T)
}

# 2) AGN:
# ANs with "mediocre" skewness
ANs_AGN_med = function(n) {
  sample_ANs(n, skewness_param = AGN_med, Ey_type = 2, pure_noise = T)
}
# LSs with "mediocre" skewness
LSs_AGN_med = function(n) {
  sample_LSs(n, skewness_param = AGN_med, Ey_type = 2, pure_noise = T)
}
# ANs with "strong" skewness
ANs_AGN_strong = function(n) {
  sample_ANs(n, skewness_param = AGN_str, Ey_type = 2, pure_noise = T)
}
# LSs with "strong" skewness
LSs_AGN_strong = function(n) {
  sample_LSs(n, skewness_param = AGN_str, Ey_type = 2, pure_noise = T)
}
# ANs with "extreme" skewness
ANs_AGN_extreme = function(n) {
  sample_ANs(n, skewness_param = AGN_ex, Ey_type = 2, pure_noise = T)
}
# LSs with "extreme" skewness
LSs_AGN_extreme = function(n) {
  sample_LSs(n, skewness_param = AGN_ex, Ey_type = 2, pure_noise = T)
}

data_generator <- function(model, n_size, n_pairs, prefix="pure_"){
  path_to_store <- paste0("../data/skew/",prefix, model, "/")
  if (!dir.exists(paste0("../data/skew/",prefix, model))) {
    dir.create(paste0("../data/skew/",prefix, model), recursive=T)
  } else {
    message("Directory already exists.")
  }
  if (file.exists(paste0(path_to_store,"pairs_gt.txt"))) {
    file.remove(paste0(path_to_store,"pairs_gt.txt"))
    message("Pairs_gt will be overwritten")
  }
  func_generator <- switch(model,
                           "ANs_AGN_med" = ANs_AGN_med,
                           "ANs_SN_med" = ANs_SN_med,
                           "ANs_AGN_strong" = ANs_AGN_strong,
                           "ANs_SN_strong" = ANs_SN_strong,
                           "ANs_AGN_extreme" = ANs_AGN_extreme,
                           "LSs_AGN_med" = LSs_AGN_med,
                           "LSs_SN_med" = LSs_SN_med,
                           "LSs_AGN_strong" = LSs_AGN_strong,
                           "LSs_SN_strong" = LSs_SN_strong,
                           "LSs_AGN_extreme" = LSs_AGN_extreme
  )
  for(i in 1:n_pairs){
    pair <- func_generator(n_size)
    coin <- runif(1, 0, 1)
    if(coin < 0.5) {
      pair_out <-  data.frame("x_child" = pair[,2], "x_pa" = pair[,1], "noisetmp"=pair[,3])
    } else{
      pair_out <- data.frame("x_pa" = pair[,1], "x_child" = pair[,2], "noisetmp"=pair[,3])
    }
    pair_gt <- ifelse(coin > 0.5, 1, 0)
    write.table(pair_out,  paste0(path_to_store,"pair_",i,".txt"),
                sep = ",", col.names = NA,  qmethod = "double")
    write(as.matrix(pair_gt), paste0(path_to_store,"pairs_gt.txt"), append = T)
  }
}
# generate pairs for all pure noise benchmarks
models_to_generate <- list("ANs_AGN_med","ANs_SN_med","ANs_AGN_strong",
                           "ANs_SN_strong","ANs_AGN_extreme",
                           "LSs_AGN_med","LSs_SN_med","LSs_AGN_strong",
                           "LSs_SN_strong","LSs_AGN_extreme")
lapply(models_to_generate, function (m) data_generator(m, n, n_pairs))

