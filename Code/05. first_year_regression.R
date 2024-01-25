library(tidyverse)
library(lme4)
library(data.table) 
library(estimatr)
library(MASS)
library(ggpubr)
library(reticulate)
#####################################################################
# FUNCTIONS

generate_formula_nn = function(network_vars,k){
    # Generates a string out of network variables names that can be used in the formula
    
    ending = paste(k,"nn",sep="")
    network_vars_nn = c()
    for (i in 1:length(network_vars)){
      network_vars_nn[i] = paste(network_vars[i],ending,sep="_")
    } 
    network_vars_nn_concat = paste(network_vars_nn,collapse=" + ")
   return(network_vars_nn_concat)
}

rename_output_vars = function(vars_output, network_vars,k){
  # Adds an ending _knn to predictors
  
  var_renamed = c()
  for (v in vars_output){
    if(endsWith(v,paste("_",k,"nn",sep=""))){
      v = substr(v,1,nchar(v)-nchar(as.character(k))-3)
      v = paste(v,"nn",sep="_")
    }
    var_renamed = c(var_renamed,v)
  }
  return(var_renamed)
}

run_lin_regr = function(data, y, FE, X, network_vars, k_vec){
  anova_p_values = list()
  output_list = list() 
  
  regressors = paste(FE,X,sep=" + ")
  f <- as.formula(paste(y, regressors, sep = " ~ "))
  mod0 = lm(formula = f, data = data)
  
  
  for (k in k_vec){
    if (k!=0){
      network_vars_nn_concat = generate_formula_nn(network_vars,k)
      regressors = paste(FE,paste(X,network_vars_nn_concat,sep=" + "),sep=" + ")
    }
    else{
      regressors = paste(FE,X,sep=" + ")
    }
    
    f <- as.formula(paste(y, regressors, sep = " ~ "))
    
    # run the model
    lin_regr = lm(formula = f, data = data)
    lin_regr_robust = lm_robust(formula = f, data = data)
    regr_output = data.frame(summary(lin_regr_robust)$coefficients)[,-c(3,7)]
    model_name = paste("gpa_FE",k,sep="_")
    anova_p_values[paste("k_",k,sep="")] = anova(mod0,lin_regr)[2,"Pr(>F)"]
    
    # model diagnostics
    par(mfrow = c(2, 2))
    plot(lin_regr,main=model_name)
    #############
    
    row.names(regr_output) = rename_output_vars(row.names(regr_output), network_vars,k)
    colnames(regr_output) = c(paste("Estimate",model_name,sep="_"),
                              paste("StdError",model_name,sep="_"),
                              paste("p_value",model_name,sep="_"),
                              paste("CI_Lower",model_name,sep="_"),
                              paste("CI_Upper",model_name,sep="_"))
    
    regr_output["R Squared",] = c(summary(lin_regr_robust)$r.squared,NA,NA,NA,NA)
    regr_output["Adj R Squared",] = c(summary(lin_regr_robust)$adj.r.squared,NA,NA,NA,NA)
    regr_output["N",] = c(lin_regr_robust$nobs,NA,NA,NA,NA)
    regr_output$var_name = row.names(regr_output)
    output_list[[model_name]] = regr_output
  }
  return(list(output = output_list,anova = anova_p_values))
}
###############################################################

path = ".../Data/"
path_analysis = ".../Analysis/"

pd = import("pandas")
year_data = pd$read_pickle(paste(path,"KNN/KNN_YEAR_METRICS_201192+.pkl",sep=""))
df2_info = pd$read_pickle(paste(path,"for_network/df2_info.pkl",sep=""))
demo = pd$read_pickle(paste(path,"for_network/demo.pkl",sep=""))

# Cohort 2016
demo = demo[(demo$admitdate=="F16") & (demo$application_status %in% c("Freshmen","Sophomore")),] 
# Only graduated
grad_2019 = df2_info[df2_info$last_term %in%c(201903,201914),]$graduated
sum(grad_2019)

# Keeping only graduated and from 2016 cohort
# 6553 -> 4889: 26%
cohort_2016_grad = df2_info[(df2_info$mellon_id %in% demo$mellon_id)
                            &(df2_info$graduated == 1),]

table(year_data$year)

first_year_vars = year_data[year_data$year==2016,]
first_year_vars = first_year_vars[, -which(names(first_year_vars) %in% c("major_change"))]
cohort_2016_grad =  merge(cohort_2016_grad,first_year_vars,on=c("mellon_id"))

# delete useless columns
cohort_2016_grad = cohort_2016_grad[,!startsWith(colnames(cohort_2016_grad),"major_name_1_")]
cohort_2016_grad = cohort_2016_grad[,!startsWith(colnames(cohort_2016_grad),"year_nn_")]

# factors
cohort_2016_grad$major_school_name_1 = as.factor(unlist(cohort_2016_grad$major_school_name_1))
cohort_2016_grad$major_name_1 = as.factor(unlist(cohort_2016_grad$major_name_1))

# 4880 -> 4700 : 3.689%
# list-wise deletion of missings
cohort_2016_grad = cohort_2016_grad[complete.cases(cohort_2016_grad),]
# Scaling uc_total_score
cohort_2016_grad$uc_total_score = scale(cohort_2016_grad$uc_total_score)
cohort_2016_grad$uc_total_score_2nn = scale(cohort_2016_grad$uc_total_score_2nn)
cohort_2016_grad$uc_total_score_4nn = scale(cohort_2016_grad$uc_total_score_4nn)
cohort_2016_grad$uc_total_score_8nn = scale(cohort_2016_grad$uc_total_score_8nn)
cohort_2016_grad$uc_total_score_16nn = scale(cohort_2016_grad$uc_total_score_16nn)
##########################################################################
# GPA prediction
#########################################################################
par(mfrow = c(1, 1))
hist(cohort_2016_grad$last_gpa_cum,main="Distribution of Dependent Variable")

# Variables
y = "last_gpa_cum"
# fixed effect
FE = "major_school_name_1"
X = c("hs_gpa","uc_total_score","US_citizen","major_stem_1","female","low_income_desc","URM")
# knn variables
network_vars = c("hs_gpa","gpa_cumulative","major_stem_1","US_citizen","same_major","female","low_income_desc","uc_total_score","URM")
X = paste(X,collapse=" + ")
k_vec = c(16,8,4,2,0)
########################

# remove outliers based on leverage (4700->4661)
regressors = paste(FE,X,sep=" + ")
f <- as.formula(paste(y, regressors, sep = " ~ "))
mod = lm(formula=f, data=cohort_2016_grad)
cohort_2016_grad$leverage = hatvalues(mod)
hist(cohort_2016_grad$leverage)
cohort_2016_grad = cohort_2016_grad[cohort_2016_grad$leverage<0.02,]

# Run fixed effect linear regression for all k
output_list = run_lin_regr(cohort_2016_grad, y, FE, X, network_vars, k_vec)
anova_p_values = output_list[['anova']]
print(anova_p_values)
output_list = output_list[['output']]
 
output_gpa = output_list %>% reduce(full_join,"var_name")
row.names(output_gpa) = output_gpa$var_name
output_gpa = output_gpa[,-c(6)]
model_names = c("5. 16 neighbors", "4. 8 neighbors",
                "3. 4 neighbors","2. 2 neighbors",
                "1. no neighbors")
# R squared table
metrics = output_gpa[c(30,31),]
metrics = metrics[,c(1,6,11,16,21)]
colnames(metrics) = model_names
metrics = metrics[,order(colnames(metrics))]
write.csv(metrics, paste(path_analysis,"ols_metrics_1.csv"), row.names=TRUE)


# PLOT THE RESULTS

estimates = output_gpa[-c(2:13),]
estimates = estimates[-c(18:20),]
estimates = estimates[order(row.names(estimates)), ]
row.names(estimates) = c("(Intercept)","Female","Female(neighbors)",
                         "GPA(neighbors)","High School GPA","High School GPA(neighbors)",
                         "Low Income","Low Income(neighbors)",
                         "STEM","STEM(neighbors)","Same major(neighbors)",
                         "Admission Score","Admission score(neighbors)",
                         "URM","URM(neighbors)",
                         "US citizen","US citizen(neighbors)")
estimates = estimates[-1,]
var_name = row.names(estimates)
estimates_long = data.frame(var_name=character(),estimate=numeric(),stdError=numeric(),
                            p_value=numeric(),ci_2_5=numeric(),ci_97_5=numeric(),
                            model_name=character())
i = 1
for (nn in k_vec){
  subset_nn = estimates[,endsWith(colnames(estimates),as.character(nn))]
  subset_nn = cbind(row.names(estimates),subset_nn)
  subset_nn$model_name = model_names[i]
  colnames(subset_nn) = c("var_name","estimate","stdError","p_value","ci_2_5","ci_97_5","model_name")
  estimates_long = rbind(estimates_long,subset_nn)
  i = i+1
}
estimates_long = estimates_long[complete.cases(estimates_long),]

lin_regr_plot = 
  ggplot(data = estimates_long,
       aes(x = estimate, y = var_name,color=model_name)) +
  geom_vline(xintercept = 0, linetype = 2) +
  geom_point() + 
  geom_errorbarh(aes(xmin = ci_2_5, xmax = ci_97_5, height = 0.1)) + 
  labs(title = "OLS Regression of Final GPA",
       x = "Coefficient Estimate",
       y = NULL,
       color='Model specification'
  )+
  scale_color_manual(values=c("#000000", "#009E73","#0072B2", "#D55E00", "#CC79A7"))+
  scale_y_discrete(limits=rev)+
  theme_bw()



###
# GPA CUM in the first year included

y = "last_gpa_cum"
FE = "major_school_name_1"
X = c("gpa_cumulative","uc_total_score","US_citizen","major_stem_1","female","low_income_desc","URM")
network_vars = c("gpa_cumulative","major_stem_1","US_citizen","same_major","female","low_income_desc","uc_total_score","URM")
k_vec = c(16,8,4,2,0)
X = paste(X,collapse=" + ")


# Run fixed effect linear regression for all k
output_list = run_lin_regr(cohort_2016_grad, y, FE, X, network_vars, k_vec)
anova_p_values = output_list[['anova']]
output_list = output_list[['output']]

output_gpa = output_list %>% reduce(full_join,"var_name")
row.names(output_gpa) = output_gpa$var_name
output_gpa = output_gpa[,-c(6)]

metrics = output_gpa[c(29,30),]
metrics = metrics[,c(1,6,11,16,21)]
colnames(metrics) = model_names
metrics = metrics[,order(colnames(metrics))]
write.csv(metrics, paste(path_analysis,"ols_metrics_2.csv"), row.names=TRUE)

# PLOT THE RESULTS

estimates = output_gpa[-c(2:13),]
estimates = estimates[-c(17,18),]
estimates = estimates[-17,]
estimates = estimates[order(row.names(estimates)), ]
row.names(estimates) = c("(Intercept)","Female","Female(neighbours)",
                         "GPA","GPA(neighbours)","Low Income","Low Income(neighbours)",
                         "STEM","STEM(neighbours)","Same major(neighbours)",
                         "UC total score","UC total score(neighbours)",
                         "URM","URM(neighbours)",
                         "US citizen","US citizen(neighbours)")
estimates = estimates[-1,]
var_name = row.names(estimates)
estimates_long = data.frame(var_name=character(),estimate=numeric(),stdError=numeric(),
                            p_value=numeric(),ci_2_5=numeric(),ci_97_5=numeric(),
                            model_name=character())
model_names = c("5. 16 neighbours", "4. 8 neighbours",
                "3. 4 neighbours","2. 2 neighbours",
                "1. no neighbours")
i=1
for (nn in k_vec){
  subset_nn = estimates[,endsWith(colnames(estimates),as.character(nn))]
  subset_nn = cbind(row.names(estimates),subset_nn)
  subset_nn$model_name = model_names[i]
  colnames(subset_nn) = c("var_name","estimate","stdError","p_value","ci_2_5","ci_97_5","model_name")
  estimates_long = rbind(estimates_long,subset_nn)
  i = i+1
}
estimates_long = estimates_long[complete.cases(estimates_long),]

ggplot(data = estimates_long,
       aes(x = estimate, y = var_name,color=model_name)) +
  geom_vline(xintercept = 0, linetype = 2) +
  geom_point() + 
  geom_errorbarh(aes(xmin = ci_2_5, xmax = ci_97_5, height = 0.1)) + 
  labs(title = "OLS Regression of GPA",
       x = "Coefficient Estimate",
       y = "Predictors",
       color='Model specification',
       caption = "Error bars show the 95% confidence interval, robust standard errors.")+
  scale_color_manual(values=c("#000000", "#009E73","#0072B2", "#D55E00", "#CC79A7"))+
  scale_y_discrete(limits=rev)+
  theme_bw()


###########
# GPA_neighbours is not included

y = "last_gpa_cum"
FE = "major_school_name_1"
X = c("hs_gpa","uc_total_score","US_citizen","major_stem_1","female","low_income_desc","URM")
network_vars = c("hs_gpa","major_stem_1","US_citizen","same_major","female","low_income_desc","uc_total_score","URM")
X = paste(X,collapse=" + ")

k_vec = c(16,8,4,2,0)


# Run fixed effect linear regression for all k
output_list = run_lin_regr(cohort_2016_grad, y, FE, X, network_vars, k_vec)
anova_p_values = output_list[['anova']]
output_list = output_list[['output']]


output_gpa = output_list %>% reduce(full_join,"var_name")
row.names(output_gpa) = output_gpa$var_name
output_gpa = output_gpa[,-c(6)]
model_names = c("5. 16 neighbours", "4. 8 neighbours",
                "3. 4 neighbours","2. 2 neighbours",
                "1. no neighbours")
# R squared table

metrics = output_gpa[c(29,30),]
metrics = metrics[,c(1,6,11,16,21)]
colnames(metrics) = model_names
metrics = metrics[,order(colnames(metrics))]
write.csv(metrics, paste(path_analysis,"ols_metrics_3.csv"), row.names=TRUE)
anova_p_values

# PLOT THE RESULTS

estimates = output_gpa[-c(2:13),]
estimates = estimates[-c(17:19),]
estimates = estimates[order(row.names(estimates)), ]
row.names(estimates) = c("(Intercept)","Female","Female(neighbours)",
                         "High School GPA","High School GPA(neighbours)",
                         "Low Income","Low Income(neighbours)",
                         "STEM","STEM(neighbours)","Same major(neighbours)",
                         "UC total score","UC total score(neighbours)",
                         "URM","URM(neighbours)",
                         "US citizen","US citizen(neighbours)")
estimates = estimates[-1,]
var_name = row.names(estimates)
estimates_long = data.frame(var_name=character(),estimate=numeric(),stdError=numeric(),
                            p_value=numeric(),ci_2_5=numeric(),ci_97_5=numeric(),
                            model_name=character())
i = 1
for (nn in k_vec){
  subset_nn = estimates[,endsWith(colnames(estimates),as.character(nn))]
  subset_nn = cbind(row.names(estimates),subset_nn)
  subset_nn$model_name = model_names[i]
  colnames(subset_nn) = c("var_name","estimate","stdError","p_value","ci_2_5","ci_97_5","model_name")
  estimates_long = rbind(estimates_long,subset_nn)
  i = i+1
}
estimates_long = estimates_long[complete.cases(estimates_long),]

ggplot(data = estimates_long,
       aes(x = estimate, y = var_name,color=model_name)) +
  geom_vline(xintercept = 0, linetype = 2) +
  geom_point() + 
  geom_errorbarh(aes(xmin = ci_2_5, xmax = ci_97_5, height = 0.1)) + 
  labs(title = "OLS Regression of Final GPA",
       x = "Coefficient Estimate",
       y = "Predictors",
       color='Model specification',
       caption = "Error bars show the 95% confidence interval, robust standard errors.")+
  scale_color_manual(values=c("#000000", "#009E73","#0072B2", "#D55E00", "#CC79A7"))+
  scale_y_discrete(limits=rev)+
  theme_bw()


##########################################################################
# Major Change Prediction
#########################################################################
table(cohort_2016_grad$major_change)

y = "major_change"
FE = "major_school_name_1"
X = c("gpa_cumulative","uc_total_score","US_citizen","major_stem_1","female","low_income_desc","URM")
network_vars = c("gpa_cumulative","major_stem_1","US_citizen","same_major","female","low_income_desc","uc_total_score","URM")
k_vec = c(16,8,4,2,0)
X = paste(X,collapse=" + ")



regressors = paste(FE,X,sep=" + ")
f <- as.formula(paste(y, regressors, sep = " ~ "))
mod0 = glm(formula = f, data = cohort_2016_grad,family="binomial")
lrt_p_values = list()
output_list = list() 
for (k in k_vec){
  if (k!=0){
    network_vars_nn_concat = generate_formula_nn(network_vars,k)
    regressors = paste(FE,paste(X,network_vars_nn_concat,sep=" + "),sep=" + ")
  }
  else{
    regressors = paste(FE,X,sep=" + ")
  }
  
  f <- as.formula(paste(y, regressors, sep = " ~ "))
  
  # run the model
  logistic_regr = glm(formula = f, data = cohort_2016_grad,family="binomial")
  regr_output = exp(cbind(Odds_Ratio = coef(logistic_regr), confint(logistic_regr)))
  regr_output = data.frame(regr_output)
  model_name = paste("logistic",k,sep="_")
  lrt_p_values[paste("k_",k,sep="")] = anova(mod0,logistic_regr,test="LRT")[2,"Pr(>Chi)"]
  
  # model diagnostics
  h1 = hoslem.test(logistic_regr$y, fitted(logistic_regr), g = 10)
  print(h1) # H0: the model is correctly specified
  h2 = cbind(h1$expected, h1$observed) # well calibrated?
  # confusion table
  p = predict(logistic_regr, cohort_2016_grad, type = "response")
  Con_table = table(p > 0.5, cohort_2016_grad$major_change)
  print(Con_table)
  #print(vif(logistic_regr))
  
# AOC
  par(mfrow = c(1, 1))
  pred = prediction(p, cohort_2016_grad$major_change)
  roc = performance(pred, "tpr", "fpr")
  plot(roc, col = 'red',
       main = "ROC Curve")
  abline(a=0, b=1)
  auc = performance(pred, "auc")
  auc = unlist(slot(auc, "y.values"))
  auc = round(auc, 2)
  legend(.6, .2, auc, title = "Area under the Curve", cex = .75)
  
  
  row.names(regr_output) = rename_output_vars(row.names(regr_output), network_vars,k)
  colnames(regr_output) = c(paste("Odds_Ratio",model_name,sep="_"),
                            paste("CI_Lower",model_name,sep="_"),
                            paste("CI_Upper",model_name,sep="_"))
  
  regr_output["LogLik",] = c(logLik(logistic_regr)[1],NA,NA)
  regr_output["AIC",] = c(AIC(logistic_regr),NA,NA)
  regr_output["BIC",] = c(BIC(logistic_regr),NA,NA)
  regr_output$var_name = row.names(regr_output)
  output_list[[model_name]] = regr_output
}

output_change = output_list %>% reduce(full_join,"var_name")
row.names(output_change) = output_change$var_name
output_change = output_change[,-c(4)]

metrics = output_change[c(29,30,31),]
metrics = metrics[,c(1,4,7,10,13)]
colnames(metrics) = model_names
metrics = metrics[,order(colnames(metrics))]
write.csv(metrics, paste(path_analysis,"logistic_metrics_1.csv"), row.names=TRUE)

lrt_p_values

# PLOT THE RESULTS

estimates = output_change[-c(2:13),]
estimates = estimates[-c(17,18,19),]
estimates = estimates[order(row.names(estimates)), ]
row.names(estimates) = c("(Intercept)","Female","Female(neighbors)",
                         "GPA","GPA(neighbors)","Low Income","Low Income(neighbors)",
                         #"major_change(neighbours)",
                         "STEM","STEM(neighbors)","Same major(neighbors)",
                         "Admission score","Admission score(neighbors)",
                         "URM","URM(neighbors)",
                         "US citizen","US citizen(neighbors)")
estimates = estimates[-1,]
var_name = row.names(estimates)
estimates_long = data.frame(var_name=character(),estimate=numeric(),
                            ci_2_5=numeric(),ci_97_5=numeric(),
                            model_name=character())

model_names = c("5. 16 neighbors", "4. 8 neighbors",
                "3. 4 neighbors","2. 2 neighbors",
                "1. no neighbors")
i=1
for (nn in k_vec){
  subset_nn = estimates[,endsWith(colnames(estimates),as.character(nn))]
  subset_nn = cbind(row.names(estimates),subset_nn)
  subset_nn$model_name = model_names[i]
  colnames(subset_nn) = c("var_name","estimate","ci_2_5","ci_97_5","model_name")
  estimates_long = rbind(estimates_long,subset_nn)
  i = i+1
}
estimates_long = estimates_long[complete.cases(estimates_long),]

logit_regr_plot = 
  ggplot(data = estimates_long,
       aes(x = estimate, y = var_name,color=model_name)) +
  geom_vline(xintercept = 1, linetype = 2) +
  geom_point() + 
  geom_errorbarh(aes(xmin = ci_2_5, xmax = ci_97_5, height = 0.1)) + 
  labs(title = "Logistic Regression of Major Change",
       x = "Odds Ratio Estimate",
       y = NULL,
       color='Model specification')+
  #scale_shape_manual(values=c(0, 15,16,17,18))+
  scale_color_manual(values=c("#000000", "#009E73","#0072B2", "#D55E00","#CC79A7"))+
  scale_y_discrete(limits=rev)+
  theme_bw()

#############################################################################################

# Plot estimates of the selected models
ggarrange(lin_regr_plot, logit_regr_plot, ncol=2, nrow=1, common.legend = TRUE, legend="right")

ggarrange(lin_regr_plot, logit_regr_plot, ncol=2, nrow=1, common.legend = TRUE, legend="bottom")

