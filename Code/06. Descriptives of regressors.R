library(reticulate)

path = "C:/Honors Project/Social Network Analysis/Data/"
path_analysis = "C:/Honors Project/Social Network Analysis/Analysis/"

pd = import("pandas")
year_data = pd$read_pickle(paste(path,"KNN/KNN_YEAR_METRICS_201192+.pkl",sep=""))
df2_info = pd$read_pickle(paste(path,"for_network/df2_info.pkl",sep=""))
df2 = pd$read_pickle(paste(path,"for_network/df2.pkl",sep=""))

#########
#Original
#########
y = "last_gpa_cum"
FE = "major_school_name_1"
X = c("uc_total_score","US_citizen","major_stem_1","female","low_income_desc","URM")
network_vars = c("gpa_cumulative","major_stem_1","US_citizen","same_major","female","low_income_desc","uc_total_score","URM")
X = paste(X,collapse=" + ")

cohort_2016_grad = df2_info[(df2_info$freshmen_201692 == 1)
                            &(df2_info$graduated == 1),]

first_year_vars = year_data[year_data$year==2016,]
first_year_vars = first_year_vars[, -which(names(first_year_vars) %in% c("major_change"))]
cohort_2016_grad =  merge(cohort_2016_grad,first_year_vars,on=c("mellon_id"))

cohort_2016_grad = cohort_2016_grad[,!startsWith(colnames(cohort_2016_grad),"major_name_1_")]
cohort_2016_grad = cohort_2016_grad[,!startsWith(colnames(cohort_2016_grad),"year_nn_")]
cohort_2016_grad$major_school_name_1 = as.factor(unlist(cohort_2016_grad$major_school_name_1))
cohort_2016_grad$major_name_1 = as.factor(unlist(cohort_2016_grad$major_name_1))
cohort_2016_grad = cohort_2016_grad[complete.cases(cohort_2016_grad),]

# remove outliers (3275->3248)
regressors = paste(FE,X,sep=" + ")
f <- as.formula(paste(y, regressors, sep = " ~ "))
mod = lm(formula=f, data=cohort_2016_grad)
cohort_2016_grad$leverage = hatvalues(mod)
cohort_2016_grad = cohort_2016_grad[cohort_2016_grad$leverage<0.02,]

##################
#Descriptive stats
y = c("last_gpa_cum","major_change")
FE = "major_school_name_1"
X = c("gpa_cumulative","uc_total_score","US_citizen","major_stem_1","female","low_income_desc","URM")
network_vars = c("gpa_cumulative","major_stem_1","US_citizen","same_major","female","low_income_desc","uc_total_score","URM")

network_vars_nn = c()
counter=1
for(k in c(2,4,8,16)){
  ending = paste(k,"nn",sep="")
  for (i in 1:length(network_vars)){
    network_vars_nn[counter] = paste(network_vars[i],ending,sep="_")
    counter = counter+1
  } 
}

all_vars = c(y,X,network_vars_nn)

means_orig = colMeans(cohort_2016_grad[,colnames(cohort_2016_grad) %in% all_vars])
means_transf = colMeans(transf_2017_grad[,colnames(transf_2017_grad) %in% all_vars])
sd_orig = rapply(cohort_2016_grad[,colnames(cohort_2016_grad) %in% all_vars],sd)
sd_transf = rapply(transf_2017_grad[,colnames(transf_2017_grad) %in% all_vars],sd)

ds = cbind(means_orig, means_transf = means_transf[names(means_orig)],
      sd_orig = sd_orig[names(means_orig)],sd_transf = sd_transf[names(means_orig)])

write.csv(ds, paste(path_analysis,"descriptives_of_vars_compared.csv"), row.names=TRUE)

