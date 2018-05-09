#######################################################################################
##### Supplementary R script for paper:                                          ######
##### Accounting for individual variation in cognitive testing in farmed species ######
#######################################################################################

# get relevant packages (install packages if not already loaded)
require(rstan)
require(brms)
require(lme4)
require(glmmTMB)
require(rethinking)
require(MASS)
require(ggplot2)
require(plyr)
require(dplyr)

# set your working directory to the 
setwd()

#----------------------------------------------------------------------------------------
# Function to generate multi-level data
#----------------------------------------------------------------------------------------
#List of parameters:
#                   * N_j = number of individuals
#                   * N_trials = number of trials in each condition
#                   * N_cond = number of conditions (only 2 allowed in this function)
#                   * alpha = the intercept parameter,
#                   * beta_trial = the influence of trial number (standardised)
#                   * beta_cond = the influence of condition
#                   * beta_tc = the interaction coefficient
#                   * gamma = a list of individual-level predictor coefficients
#                   * sigma_nu_1 = variance of the random intercept
#                   * sigma_nu_2 = variance of the random slope for trial
#                   * sigma_nu_3 = variance of the random slope for condition
#                   * sigma_nu_4 = variance of the random interction between trial and condition
#                   * r_cor_is1 = intercept-slope 1 (trial) correlation 
#                   * r_cor_is2 = intercept-slope 2 (condition) correlation
#                   * r_cor_isi = intercept-interaction correlation
#                   * r_cor_s1s2 = slope 1 - slope 2 correlation
#                   * r_cor_s1si = slope 1 - interaction correlation
#                   * r_cor_s2si = slope 2 - interaction correlation

generate_data <- function(N_j, N_trials, N_cond, 
                             alpha, beta_trial, beta_cond, beta_tc, gamma,
                             sigma_nu_1, sigma_nu_2, sigma_nu_3, sigma_nu_4,
                             r_cor_is1, r_cor_is2, r_cor_isi, 
                             r_cor_s1s2, r_cor_s1si, r_cor_s2si
                             )
{
  N <- N_j * N_trials * N_cond
  
  nu <- mvrnorm(
    n = N_j, mu = c(0,0,0,0), 
    Sigma = matrix(c(
      # random effects 
      sigma_nu_1^2, sigma_nu_1*sigma_nu_2*r_cor_is1, sigma_nu_1*sigma_nu_3*r_cor_is2, sigma_nu_1*sigma_nu_4*r_cor_isi,
      sigma_nu_1*sigma_nu_2*r_cor_is2, sigma_nu_2^2, sigma_nu_2*sigma_nu_3*r_cor_s1s2, sigma_nu_2*sigma_nu_4*r_cor_s1si,
      sigma_nu_1*sigma_nu_3*r_cor_is2, sigma_nu_2*sigma_nu_3*r_cor_s1s2, sigma_nu_3^2, sigma_nu_3*sigma_nu_4*r_cor_s2si,
      sigma_nu_1*sigma_nu_4*r_cor_isi, sigma_nu_2*sigma_nu_4*r_cor_s1si, sigma_nu_3*sigma_nu_4*r_cor_s2si, sigma_nu_4^2), 
      
      ncol = 4, byrow = T)
                      )
  j <- rep(1:N_j, each = N_trials*N_cond)
  x_trial <- rep(seq(1, N_trials, by = 1), N_j*N_cond)
  x_trial_z <- (x_trial - mean(x_trial))/sd(x_trial)
  x_cond_c <- rep( rep(-0.5:0.5, each=N_trials), N_j)
  j_pred <- sample( 0:1, N_j, replace = TRUE )
  j_pred_z <- (j_pred - mean(j_pred))/sd(j_pred)
  eta <- inv_logit( alpha + nu[j,1] + gamma[1] * j_pred_z[j] + 
                      (beta_trial + nu[j,2] + gamma[2] * j_pred_z[j]) * x_trial_z + 
                      (beta_cond + nu[j,3] + gamma[3] * j_pred_z[j]) * x_cond_c   +
                      (beta_tc   + nu[j,4] + gamma[4] * j_pred_z[j]) * x_trial_z * x_cond_c 
                    )
  y <- rbinom(n = N, size = 1, prob = eta)
  return(list(y = y, x_trial_z = x_trial_z, x_cond_c = x_cond_c, j_pred_z = j_pred_z, 
              j = j, N = N))
}

#------------------------------------------------------------------------------------------------------
# set the parameters and generate the data
#------------------------------------------------------------------------------------------------------

# Number of individuals
N_j <- 100
# Number of time points per individuals
N_trials <- 10
# Number of conditions
N_cond <- 2
# Overall intercept (w/h predictors centered/scaled)
alpha <- 0.2
# Time regression coefficient
beta_trial <- 0.2
# Condition regression coefficient
beta_cond <- -0.5
# Time * Condition regression coefficient
beta_tc <- -0.05
# Individual-level predictor
gamma <- c(0,0,0,-0.2)
# Standard deviation of intercept
sigma_nu_1 <- 0.4
# Standard deviation of slope across time
sigma_nu_2 <- 0.01
# Standard deviation of slope across conditions
sigma_nu_3 <- 0.01
# Standard deviation of slope 1 * slope 2 interaction
sigma_nu_4 <- 0.01
# Correlation between intercept and time slope 
r_cor_is1 <- 0.3
# Correlation between intercept and condition slope
r_cor_is2 <- 0.4
# Correlation between intercept and time*condition interaction
r_cor_isi <- 0.5
# Correlation between time slope and condition slope
r_cor_s1s2 <- 0.2
# Correlation between slope 1 and time*condition interaction
r_cor_s1si <- 0.1
# Correlation between slope 2 and time*condition interaction
r_cor_s2si <- -0.1

set.seed(1234)
d <- generate_data(N_j = N_j, N_trials = N_trials, N_cond = N_cond,  
                   alpha = alpha, beta_trial = beta_trial, 
                   beta_cond = beta_cond, beta_tc = beta_tc, gamma = gamma,
                   sigma_nu_1 = sigma_nu_1, sigma_nu_2 = sigma_nu_2, 
                   sigma_nu_3 = sigma_nu_3, sigma_nu_4 = sigma_nu_4, 
                   r_cor_is1 = r_cor_is1, r_cor_is2 = r_cor_is2, r_cor_isi = r_cor_isi,
                   r_cor_s1s2 = r_cor_s1s2, r_cor_s1si = r_cor_s1si, r_cor_s2si = r_cor_s2si
                  )

#------------------------------------------------------------------------------------------------------
# fit the model in Stan. The model code file should be saved and accessible in the working directory
#------------------------------------------------------------------------------------------------------

stan_data <- list(N = length(d$y), N_j = N_j, j = d$j,
                  x_trial_z = d$x_trial_z, x_cond_c = d$x_cond_c, y = d$y, j_pred_z = d$j_pred_z)

fit_stan <- stan("Bushby-et-al_Bernoulli-model.stan", data = stan_data, 
            chains = 4, cores = 4, warmup = 2500, iter = 5000)

print(fit_stan, pars = c("alpha","beta","gamma","Sigma","Rho"), 
      probs=c(0.025,0.975))

#------------------------------------------------------------------------------------------------------
# fit the same model in brms using default priors/MCMC iterations
#------------------------------------------------------------------------------------------------------
fit_brms <- brm(
  y ~ x_trial_z * x_cond_c * j_pred_z[j] +
     (x_trial_z * x_cond_c | j ),
  data = d, family = "bernoulli"
)

summary(fit_brms)

#------------------------------------------------------------------------------------------------------
# fit the frequentist model using lme4
#------------------------------------------------------------------------------------------------------
# single model; struggles to converge
fit_lme4 <- glmer(
  y ~ x_trial_z * x_cond_c * j_pred_z[j] +
      (x_trial_z * x_cond_c | j ),
  data = d, family = binomial(link="logit")
  )

fit_lme4_2 <- glmer(
  y ~ x_trial_z + x_cond_c + j_pred_z[j] + 
  x_trial_z * x_cond_c    + 
  x_trial_z * j_pred_z[j] +
  x_cond_c  * j_pred_z[j] + 
  x_trial_z * x_cond_c * j_pred_z[j] + 
  (x_trial_z + x_cond_c + x_trial_z * x_cond_c | j ),
  data = d, family = binomial(link="logit")
  )

summary(fit_lme4)

#### fit the model using non-parametric bootstrapping to get uncertainty intervals on the parameters

# make a data frame of the data
d_df <- data.frame(
  y = d$y, 
  x_trial_z = d$x_trial_z,
  x_cond_c = d$x_cond_c,
  j_pred_z = d$j_pred_z[d$j],
  j = d$j
)

# set number of boots and an empty matrix to store parameter estimates
N_boots <- 100
param_estims <- matrix(0, ncol = N_boots, nrow=18)

# do the bootstrapping - for 100, it takes around 5-10 minutes
for(i in 1:N_boots){
  # sample with replacement
  sampled_j <- sample(1:length(unique(d_df$j)), length(unique(d_df$j)), replace=TRUE)
  # new sample of data
  d_sample <- do.call(rbind, lapply(sampled_j, function(z) d_df[d_df$j %in% z, ]))
  # fit the model
  fit <- glmmTMB(
    y ~ x_trial_z * x_cond_c * j_pred_z +
      (x_trial_z * x_cond_c | j ),
    data = d_sample, family = binomial(link="logit")
  )
  fixed <- as.vector( fixef(fit)[c(1:3,5,4,6:8)]$cond)
  std_cor <- c(as.vector(attr(VarCorr(fit)$cond$j, 'stddev')),
               as.vector(attr(VarCorr(fit)$cond$j, 'correlation'))[c(2:4,7:8,12)])
  param_estims[,i] <- c(fixed, std_cor)
}

#------------------------------------------------------------------------------------------------------
# post-process the results, mainly from the Stan model but including some comparisons between the models
#------------------------------------------------------------------------------------------------------

ps_stan <- as.matrix(fit_stan)
ps_brms <- as.matrix(fit_brms)

x_trial <- seq(min(d$x_trial_z), max(d$x_trial_z), length.out = N_trials)
x_cond_c <- rep(-0.5:0.5, each=N_trials)
x_tc <- cbind(rep(x_trial,2), x_cond_c)

nu <- ps_stan[,grep("nu", colnames(ps_stan))]
nu_1 <- nu[,1:N_j]; nu_2 <- nu[,(N_j+1):(N_j*2)]; 
nu_3 <- nu[,(N_j*2+1):(N_j*3)]; nu_4 <- nu[,(N_j*3+1):(N_j*4)]
alpha <- ps_stan[,"alpha"]; beta_trial <- ps_stan[,"beta[1]"]; beta_cond <- ps_stan[,"beta[2]"]; 
beta_tc <- ps_stan[,"beta[3]"]

# graphs the probability correct across trials on the probability scale

# condition one 
pred_avg_tc1 <- sapply(x_trial, 
                       function(x_trial) 
                         inv_logit(alpha + beta_trial * x_trial + beta_cond * -0.5 + 
                                     beta_tc * x_trial * -0.5)
)
pred_mu_tc1 <- apply(pred_avg_tc1, 2, mean )
pred_hdi_tc1 <- apply(pred_avg_tc1, 2, function(z) HPDI(z, 0.89))

pred_list_tc1 <- rep(list(list()), N_j)
for(i in 1:N_j){
  pred_list_tc1[[i]] <- sapply(
    x_trial, 
    function(x_trial)
      inv_logit( alpha + nu_1[,i] + (beta_trial + nu_2[,i]) * x_trial + 
                  (beta_cond + nu_3[,i]) * -0.5 + 
                  (beta_tc + nu_4[,i]) * x_trial * -0.5)
  )
}

pred_tc1_mu <- lapply(pred_list_tc1, function(z) apply(z, 2, mean) )
pred_tc1_hdi <- lapply(pred_list_tc1, function(z) apply(z, 2, function(x) HPDI(x,0.89)) )

# condition 2
pred_avg_tc2 <- sapply(x_trial, 
                       function(x_trial) 
                         inv_logit( alpha + beta_trial * x_trial + beta_cond * 0.5 + 
                                      beta_tc * x_trial * 0.5)
)
pred_mu_tc2 <- apply(pred_avg_tc2, 2, mean )
pred_hdi_tc2 <- apply(pred_avg_tc2, 2, function(z) HPDI(z, 0.89))

pred_list_tc2 <- rep(list(list()), N_j)
for(i in 1:N_j){
  pred_list_tc2[[i]] <- sapply(
    x_trial, 
    function(x_trial)
      inv_logit(alpha + nu_1[,i] + (beta_trial + nu_2[,i]) * x_trial + 
      (beta_cond + nu_3[,i]) * 0.5 + 
      (beta_tc + nu_4[,i]) * x_trial * 0.5)
  )
}

pred_tc2_mu <- lapply(pred_list_tc2, function(z) apply(z, 2, mean) )
pred_tc2_hdi <- lapply(pred_list_tc2, function(z) apply(z, 2, function(x) HPDI(x,0.89)) )

png("Sim-model.png", width=1600, height=800, res=200)
par(mfrow=c(1,2))
# plot 1 
plot(pred_mu_tc1 ~ x_tc[1:N_trials,1], type="n", ylim = c(0,1), xlim=c(-1.5,1.5), 
     xlab="Trials", ylab = "Probability correct", main = "Initial task",
     xaxt="n")
new.xaxt <- rep(1:N_trials, 2)
at <- x_tc[,1]
axis( side = 1 , at=at , labels = new.xaxt, cex.axis =1)
for(i in 1:N_j){
  lines(pred_tc1_mu[[i]] ~ x_tc[1:N_trials,1], col = col.alpha(rangi2, 0.2))
}
shade(pred_hdi_tc1, x_tc[1:N_trials,1], col = col.alpha("gray",0.7))
lines(pred_mu_tc1 ~ x_tc[1:N_trials,1], lwd=2)
mtext("(a)", side = 1, at = -2, line = -14, cex = 1.2)
# plot 2 
plot(pred_mu_tc2 ~ x_tc[1:N_trials,1], type="n", ylim = c(0,1), xlim=c(-1.5,1.5), 
     xlab="Trials", ylab = "Probability correct", main = "Reversal task",
     xaxt="n")
new.xaxt <- rep(1:N_trials, 2)
at <- x_tc[,1]
axis( side = 1 , at=at , labels = new.xaxt, cex.axis =1)
for(i in 1:N_j){
  lines(pred_tc2_mu[[i]] ~ x_tc[(N_trials+1):(N_trials*2),1], col = col.alpha(rangi2, 0.2))
}
shade(pred_hdi_tc2, x_tc[(N_trials+1):(N_trials*2),1], col = col.alpha("gray",0.7))
lines(pred_mu_tc2 ~ x_tc[(N_trials+1):(N_trials*2),1], lwd=2)
mtext("(b)", side = 1, at = -2, line = -14, cex = 1.2)
dev.off()

#------------------------------------------------------------------------------------------
# plot parameter estimates across the models and against the values to simulate the data

want_stan <- c(1:12,14:16,19:20,24)
stan_mu <- apply(ps_stan[,want_stan], 2, mean)
stan_hdi <- apply(ps_stan[,want_stan], 2, function(z) HPDI(z, 0.89) )

want_brms <- c(1,2,3,5,4,6:14,16,15,17,18)
brms_mu <- apply(ps_brms[,want_brms], 2, mean)
brms_hdi <- apply(ps_brms[,want_brms], 2, function(z) HPDI(z, 0.89) )

glmmTMB_mu <- apply(param_estims, 1, function(z) mean(z, na.rm=TRUE))
glmmTMB_hdi <- apply(param_estims, 1, function(x) HPDI(x, 0.89))

param_names <- c("Intercept", "b1", "b2", "b3", "g1", "g2", "g3", "g4", 
                 "SD_1", "SD_2", "SD_3", "SD_4", 
                 "Cor1", "Cor2", "Cor3", "Cor4", "Cor5", "Cor6"
                 )

sims <- c(0.2, 0.2, -0.5, -0.05, 0,0,0,-0.2, 0.4, 0.01, 0.01, 0.01, 0.3, 0.4, 
          0.5, 0.2, 0.1, -0.1)

results_df <- data.frame(
  parameter = rep(param_names, 4 ) ,
  model = rep(c("Stan","brms","glmmTMB","sim"), each=length(param_names) ),
  mu = c(stan_mu, brms_mu, glmmTMB_mu, sims),
  CI_low = c(stan_hdi[1,], brms_hdi[1,], glmmTMB_hdi[1,], sims),
  CI_high = c(stan_hdi[2,], brms_hdi[2,], glmmTMB_hdi[2,], sims)
)

results_df$parameter <- factor(results_df$parameter, levels = (unique(results_df$parameter)))
results_df$model <- factor(results_df$model, levels = c("sim","glmmTMB","brms","Stan"))

ggplot(results_df, aes(x = parameter, mu, color=model)) + 
  scale_fill_brewer(palette = "Dark2") + 
  geom_point(position = position_dodge(width=0.5)) + 
  geom_errorbar(aes(x=parameter, ymin=CI_low, ymax=CI_high), 
                width=0.1, position = position_dodge(width = 0.5)) + 
  labs(y = "Value", x="Parameter") + 
  theme(axis.title=element_text(size=15),
        legend.text = element_text(size=18),
        legend.position = "top",
        legend.title = element_blank()) + 
  coord_flip()

ggsave("Parameter-comparison.png", last_plot(), width=7, height=8)

########################################################################################################
####################################      end   ########################################################
########################################################################################################
