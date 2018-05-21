  data{
    int<lower=1> N; 							// the total number of rows in the data
    int<lower=1> N_j;							// the total number of individuals
    int<lower=1> j[N];							// id vector for individuals
    vector[N] x_trial_z;						// the trial number variable
    vector[N] x_cond_c;							// binary variable for reversal (1) or not (0)
    vector[N_j] j_pred_z; 						// the diet condition (individual-level predictor)
    int<lower=0, upper=1> y[N];						// the binary y variable
  }

  parameters{
    real alpha; 							// intercept parameter
    vector[3] beta; 							// vector of observation-level regression coefficients
    vector[4] gamma;							// vector of individual-level predictor coefficients
    vector<lower=0>[4] Sigma;						// standard deviations of random effects (RE)
    cholesky_factor_corr[4] L_Rho;					// cholesky factor of the RE correlation matrix
    matrix[4, N_j] Z;							// scaled (unit normal) RE parameters (for efficiency)
  }

  transformed parameters{
    matrix[4,4] Rho;							// RE correlation matrix
    matrix[N_j,4] nu;							// unscaled RE parameters
    nu = (diag_pre_multiply(Sigma, L_Rho) * Z)';			// transform RE parameters from scaled to unscaled
    Rho = L_Rho * L_Rho';						// transform from lower cholesky matrix to correlation matrix
  }

  model{
    vector[N] eta;				// local variable to hold linear predictor

    // prior distributions
    to_vector(Z) ~ normal(0, 1);		// vector of unit normals for the non-centered parameterisation
    alpha ~ normal(0, 10);
    beta ~  normal(0, 1);
    gamma ~ normal(0, 1);
    L_Rho ~ lkj_corr_cholesky(2.0);
    Sigma ~ normal(0, 1);

    // linear predictor
    for(i in 1:N){
      eta[i] = alpha + nu[j[i],1] + gamma[1] * j_pred_z[j[i]]
            + (beta[1] + nu[j[i],2] + gamma[2] * j_pred_z[j[i]]) * x_trial_z[i]
            + (beta[2] + nu[j[i],3] + gamma[3] * j_pred_z[j[i]]) * x_cond_c[i]
            + (beta[3] + nu[j[i],4] + gamma[4] * j_pred_z[j[i]]) * x_trial_z[i] * x_cond_c[i];

    }

     // Bernoulli likelihood statement
      y ~ bernoulli_logit(eta);
  }
