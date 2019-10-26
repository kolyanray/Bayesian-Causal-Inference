# initial project environment and package dependency
library("bartCause")
library("grf")
library("Matching")
library("ATE")
library("balanceHD")
library("glmnet")
library("bcf")




itnumber<-200 # number of times we run the simulation
#datamatrix<- read.csv(file.choose()) # loading the ihdp.csv data set



############################################# Initialization   ###############################################################



# Causal BART: using the "bartCause" package
errorvectBart<-rep(0,itnumber) # difference between estimator and ATE
errormeanBart<-0 # average difference between the estimator and ATE in the simulations
meanBart<-0 # average estimator for ATE
errorsdBart<-0 # std of errorvectBart
sizeCIvectBart<-rep(0,itnumber) # size of confidence interval for ATE
sizeCImeanBart<-0 # average size of confidence interval for ATE
sizeCIsdBart<-0 # std ofsizeCIvectBart
coverageBart<-0 # (frequentist) coverage of confidence/credible interval
typeIIBart<-0 # type II error


# bcf:  using the "bcf" package
errorvectbcf<-rep(0,itnumber)
errormeanbcf<-0
meanbcf<-0
errorsdbcf<-0
sizeCIvectbcf<-rep(0,itnumber)
sizeCImeanbcf<-0
sizeCIsdbcf<-0
coveragebcf<-0
typeIIbcf<-0


# Causal BART propr score estimates:  using the "bartCause" package (with prop. score estimate)
errorvectBartPC<-rep(0,itnumber)
errormeanBartPC<-0
meanBartPC<-0
errorsdBartPC<-0
sizeCIvectBartPC<-rep(0,itnumber)
sizeCImeanBartPC<-0
sizeCIsdBartPC<-0
coverageBartPC<-0
typeIIBartPC<-0

# Causal forrests  using the "grf" package (with AIPW)
errorvectForest<-rep(0,itnumber)
errormeanForest<-0
meanForest<-0
errorsdForest<-0
sizeCIvectForest<-rep(0,itnumber)
sizeCImeanForest<-0
sizeCIsdForest<-0
coverageForest<-0
typeIIForest<-0


# Causal forrests  using the "grf" package (with TMLE)
errorvectForest2<-rep(0,itnumber)
errormeanForest2<-0
errorsdForest2<-0
meanForest2<-0
sizeCIvectForest2<-rep(0,itnumber)
sizeCImeanForest2<-0
sizeCIsdForest2<-0
coverageForest2<-0
typeIIForest2<-0


# Ordinary least squares (OLS)
errorvectOLS<-rep(0,itnumber)
errormeanOLS<-0
meanOLS<-0
errorsdOLS<-0
sizeCIvectOLS<-rep(0,itnumber)
sizeCImeanOLS<-0
sizeCIsdOLS<-0
coverageOLS<-0
typeIIOLS<-0



# Balancing the covariates: using the "ATE " package
errorvectCovB<-rep(0,itnumber)
errormeanCovB<-0
errorsdCovB<-0
meanCovB<-0
sizeCIvectCovB<-rep(0,itnumber)
sizeCImeanCovB<-0
sizeCIsdCovB<-0
coverageCovB<-0
typeIICovB<-0


# Balancing the covariates IPW: using the "balanceHD" package
errorvectIPW<-rep(0,itnumber)
errormeanIPW<-0
errorsdIPW<-0
meanIPW<-0
sizeCIvectIPW<-rep(0,itnumber)
sizeCImeanIPW<-0
sizeCIsdIPW<-0
coverageIPW<-0
typeIIIPW<-0



# Propensity score mathcing: using the "Matching" package

errorvectM<-rep(0,itnumber)
errormeanM<-0
errorsdM<-0
meanM<-0
sizeCIvectM<-rep(0,itnumber)
sizeCImeanM<-0
sizeCIsdM<-0
coverageM<-0
typeIIM<-0



for (j in 1:itnumber){ # we run itnumber amount of simulations (with new data sets)
  
  #################  Generating the data #################
  ############################# Syntetic data set #######################
  
  # function used for the treatment assignment
  gT=function(x){
    return(x[1]-0.5+(x[2]-0.5)^2+x[3]^2-1/3+2*sin(2*x[4])+exp(-x[5])-exp(-1)-1)
  }
  
  # function used for the response surface
  gf=function(x){
    return(exp(-x[1])+x[2]^2+x[3]+ifelse(x[4]>0,1,0)+cos(x[5]))
  }
  
  
  n<-1000L # sample size
  p<-100 # number of features
  ATEtrue<-1 # ATE value
  # initialization of treatment, observation and feature matrix
  t<-rep(0,n)
  Y<-rep(0,n)
  x<-matrix(rep(0,n*p),nrow=n)
  # generating the treatment, observation and feature matrix using the above functions
  for(i in 1:n){
    x[i,]<-rnorm(p,0,1)
    t[i]<-ifelse(gT(x[i,1:5])>0,1,0)
    # heterogenous treatment effect
    #Y[i]<-rnorm(1,0,1)+gf(x[i,1:5])+t[i]*(1+2*x[i,2]*x[i,5])
    # homogenous treatment effect
    Y[i]<-rnorm(1,0,1)+gf(x[i,1:5])+t[i]
  }
 
  ############################# Semi-syntatic data set #######################
  
  
  
  #p=25 # number of features
  #t<-datamatrix[,2]  # treatment
  #n=length(t) # data size
  #xdf<-datamatrix[,3:(p+2)] 
  #x<-data.matrix(xdf, rownames.force = NA)  # feature matrix
  #for (i in 1:p){ 
  #  x[,i]<-(x[,i]-mean(x[,i]))/sd(x[,i])
  #} # standardizing the feature matrix
  #intercept<-rep(1,n) # including intercept
  #x<-cbind(intercept,x) 
  #ATEtrue<-4 # assigning ATE value
  #W<-matrix(rep(0.5,(p+1)*n),nrow=n) # random matrix used 
  
  # generating beta:
  #prob<-c(0.6,0.1,0.1,0.1,0.1) # probability
  #beta<-sample(x=c(0,0.1,0.2,0.3,0.4), p+1, replace=T, prob)
  
  ##Generating the Y0:
  #mu0<-exp((x+W)%*%beta) 
  #Y0<-mvrnorm(n=1,mu0, Sigma=diag(n))
  
  ##Generating the Y1
  #omega<- (sum(x%*%beta)-sum(mu0))/n-4
  #mu1<- x%*%beta-omega
  #Y1<-mvrnorm(n=1,mu1, Sigma=diag(n))
  #Y<-rep(0,n)
  #for(i in 1:n){
  #  Y[i]<-t[i]*Y1[i]+(1-t[i])*Y0[i]
  #}
  
  ############## bcf #######################
  dat<-data.frame(x=x, Y=Y, t=t) # putting the data in data.frame format
  
  logit.ps <- glm(t ~ x, data = dat, family = binomial) # estimating the propensity score using logistic regression
  pihat<-fitted(logit.ps)
  nsim<-2000 # number of simulations in the BCF after the burn-in periode  
  fitbcf <- bcf(Y, t, x, x, pihat, nburn=2000, nsim=nsim) # calling the bcf function
  postATEbcf<-rep(0,nsim) # initialzing posterior draws of ATE
  for(i in 1:nsim){
    postATEbcf[i]<-mean(fitbcf$tau[i,]) # posterior draws of ATE
  }
  for(i in 1:nsim){ # ordering posterior draws of ATE
    for(k in 1:(nsim-1)){
      if(postATEbcf[k]>postATEbcf[k+1]){
        help<-postATEbcf[k]
        postATEbcf[k]<-postATEbcf[k+1]
        postATEbcf[k+1]<-help
      }
    }
  }
  credlow_bcf<-postATEbcf[nsim*0.025] # 2,5% quantile for credible set for ATE
  credup_bcf<-postATEbcf[nsim*0.975] # 97,5% quantile for credible set for ATE
  meanbcf <- meanbcf+mean(fitbcf$tau)/itnumber # average of posterior means
  errorvectbcf[j] <- abs(mean(fitbcf$tau)-ATEtrue) # difference between posterior mean (average of 2000 draws) and true ATE
  sizeCIvectbcf[j] <- credup_bcf-credlow_bcf # size of credible interal
  coveragebcf <- coveragebcf+ifelse( credup_bcf >=ATEtrue && credlow_bcf<=ATEtrue  ,1,0) # frequentist coverage of credible interval
  typeIIbcf <- typeIIbcf+ifelse( credup_bcf>=0 && credlow_bcf<=0  ,1,0) # type II error (If credible interval contains zero, then H) is retained and type II error was committed )

 
  
  
  
  ############## Causal Bart #######################
  fitBart <- bartc(Y, t, x, n.samples = n, method.rsp="bart",method.trt="bart") # use bartc function of "bartCause" package
  meanBart<-meanBart+summary(fitBart)$estimate[[1]]/itnumber # average posterior mean
  errorvectBart[j]<-abs(summary(fitBart)$estimate[[1]]-ATEtrue) # difference between posterior mean and true ATE
  sizeCIvectBart[j]<- summary(fitBart)$estimate[[4]]-summary(fitBart)$estimate[[3]] # size of confidence interval
  coverageBart<-coverageBart+ifelse( summary(fitBart)$estimate[[4]]>=ATEtrue && summary(fitBart)$estimate[[3]]<=ATEtrue  ,1,0) # frequentist coverage of confidence interval
  typeIIBart<-typeIIBart+ifelse( summary(fitBart)$estimate[[4]]>=0 && summary(fitBart)$estimate[[3]]<=0  ,1,0) # type II error
  
  
  
  ############## Causal Bart prop score #######################
  fitBartPC <- bartc(Y, t, x, n.samples = n, method.rsp="p.weight",method.trt="glm") # use bartc function of "bartCause" package with prop. score weights
  meanBartPC<-meanBartPC+summary(fitBartPC)$estimate[[1]]/itnumber # average posterior mean
  errorvectBartPC[j]<-abs(summary(fitBartPC)$estimate[[1]]-ATEtrue) # difference between posterior mean and true ATE
  sizeCIvectBartPC[j]<- summary(fitBartPC)$estimate[[4]]-summary(fitBartPC)$estimate[[3]] # size of confidence interval
  coverageBartPC<-coverageBartPC+ifelse( summary(fitBartPC)$estimate[[4]]>=ATEtrue && summary(fitBartPC)$estimate[[3]]<=ATEtrue  ,1,0) #frequentist coverage of confidence interval
  typeIIBartPC<-typeIIBartPC+ifelse( summary(fitBartPC)$estimate[[4]]>=0 && summary(fitBartPC)$estimate[[3]]<=0  ,1,0) # type II error
  

  ############## Causal Forest AIPW #######################
  c.forest = causal_forest(x, Y, t) # running causal_forest function of "grf" package
  ateForest<-average_treatment_effect(c.forest, target.sample = "all",  method ="AIPW") # running average_treatment_effect function of "grf" package
  meanForest<-meanForest+ateForest[[1]]/itnumber  # average posterior mean
   errorvectForest[j]<-abs(ateForest[[1]]-ATEtrue) # difference between posterior mean and true ATE
  sizeCIvectForest[j]<- ateForest[[2]]*2*1.96 # size of confidence interval 2*z_{0.975}*sigma
  coverageForest<-coverageForest+ifelse( ateForest[[1]]+1.96*ateForest[[2]]>=ATEtrue&&  ateForest[[1]]-1.96*ateForest[[2]]<=ATEtrue  ,1,0) #frequentist coverage of CI
  typeIIForest<-typeIIForest+ifelse(ateForest[[1]]+1.96*ateForest[[2]]>=0 && ateForest[[1]]-1.96*ateForest[[2]]<=0 ,1,0) # type II error

  ############## Causal Forest TMLE #######################
  # same as above, but with TMLE method instead of AIPW
  c.forest = causal_forest(x, Y, t)
  ateForest2<-average_treatment_effect(c.forest, target.sample = "all",  method ="TMLE")
  errorvectForest2[j]<-abs(ateForest2[[1]]-ATEtrue)
  meanForest2<-meanForest+ateForest2[[1]]/itnumber
  sizeCIvectForest2[j]<- ateForest2[[2]]*2*1.96
  coverageForest2<-coverageForest2+ifelse( ateForest2[[1]]+1.96*ateForest2[[2]]>=ATEtrue&&  ateForest2[[1]]-1.96*ateForest2[[2]]<=ATEtrue  ,1,0)
  typeIIForest2<-typeIIForest2+ifelse(ateForest2[[1]]+1.96*ateForest2[[2]]>=0 && ateForest2[[1]]-1.96*ateForest2[[2]]<=0 ,1,0)
  
  
  ############## OLS ####################### 
  # simple OLS algotihm
  Y0<-Y[t==0]
  xwork<-x[t==0,]
  fitOLS0<-lm(Y0~xwork-1)
  xwork<-t(colMeans(x))
  Xfull<-data.frame(xwork)
  haty0<-predict(fitOLS0,Xfull, se.fit = T)
  Y1<-Y[t==1]
  xwork<-x[t==1,]
  fitOLS1<-lm(Y1~xwork-1)
  xwork<-t(colMeans(x))
  Xfull<-data.frame(xwork)
  haty1<-predict(fitOLS1,Xfull, se.fit = T)
  ateOLS<-haty1$fit-haty0$fit
  ateOLSsd<-sqrt(haty1$se.fit^2+haty0$se.fit^2)
  meanOLS<-meanOLS+ateOLS/itnumber
  errorvectOLS[j]<-abs(ateOLS-ATEtrue)
  sizeCIvectOLS[j]<-2*1.96*ateOLSsd
  coverageOLS<-coverageOLS+ifelse( ateOLS+1.96*ateOLSsd>=ATEtrue&&  ateOLS-1.96*ateOLSsd<=ATEtrue  ,1,0)
  typeIIOLS<-typeIIOLS+ifelse(ateOLS+1.96*ateOLSsd>=0 && ateOLS-1.96*ateOLSsd<=0 ,1,0)
  
  
  ################################# # Covariate Balancing  #################  ################# 
  # xnointer<-x[,-1] # in case of second example () where we have to remove intercept
  # xnointer<-x
  #fitCovBal<-ATE(Y,t,xnointer,max.iter = 10)
  #errorvectCovB[j]<-abs(fitCovBal$est[[3]]-ATEtrue)
  #sizeCIvectCovB[j]<-summary(fitCovBal)$Estimate[3,4]-summary(fitCovBal)$Estimate[3,3]
  #coverageCovB<-coverageCovB+ifelse( summary(fitCovBal)$Estimate[3,4]>= ATEtrue&&  summary(fitCovBal)$Estimate[3,3]<=ATEtrue  ,1,0)
  #typeIICovB<-typeIICovB+ifelse(summary(fitCovBal)$Estimate[3,4] && summary(fitCovBal)$Estimate[3,3]<=0 ,1,0)
  
  ################################# # Covariate Balancing IPW  #################  ################# 
  
  ATE<-ipw.ate(x, Y, t, target.pop = 0,fit.method = "none",eps.threshold = 1/20,prop.method = "elnet",prop.weighted.fit = T,
               targeting.method = "AIPW", estimate.se = T) # calling the ipw.ate function of "balanceHD" package.
  meanIPW<-meanIPW+ATE[1]/itnumber # average posterior mean
  errorvectIPW[j]<-abs(ATE[1] -ATEtrue) # difference between posterior mean and true ATE
  sizeCIvectIPW[j]<-ATE[2]*2*1.96  # size of confidence interval 2*z_{0.975}*sigma
  coverageIPW<-coverageIPW+ifelse( ATE[1]+1.96*ATE[2]>=ATEtrue&&  ATE[1]-1.96*ATE[2]<=ATEtrue  ,1,0)  # frequentist coverage of CI
  typeIIIPW<-typeIIIPW+ifelse(ATE[1]+1.96*ATE[2]>=0 && ATE[1]-1.96*ATE[2]<=0 ,1,0) # type II error
  
  ########### Propensity score mathcing  ########### ###########
  # Propensity score mathcing using the Match package
  ATE<-Match(Y, t, x, Z = x, estimand = "ATE", M = 1) # calling Match function
  meanM<-meanM+ATE$est/itnumber # average posterior mean
  errorvectM[j]<-abs(ATE$est -ATEtrue) # difference between posterior mean (average of 2000 draws) and true ATE
  sizeCIvectM[j]<-ATE$se*2*1.96 # size of CI:  2*z_{0.975}*sigma
  coverageM<-coverageM+ifelse( ATE$est+1.96*ATE$se>=ATEtrue&&  ATE$est-1.96*ATE$se<=ATEtrue  ,1,0) # frequentist coverage of CI
  typeIIM<-typeIIM+ifelse(ATE$est+1.96*ATE$se>=0 && ATE$est-1.96*ATE$se<=0 ,1,0) # type II error
  print(j)
  }




################################### Computing the summary statistics ################################### 

############## Causal Bart #######################3

errormeanBart<-mean(errorvectBart) # average error
errorsdBart<-sd(errorvectBart) ## standard deviation of error 
sizeCImeanBart<-mean(sizeCIvectBart) # average CI size
sizeCIsdBart<-sd(sizeCIvectBart) # standard deviation of CI size
typeIIBart<-typeIIBart/itnumber # average type II error
coverageBart<-coverageBart/itnumber # average covarage

# The above results (for printing on the consol)
meanBart
errormeanBart
errorsdBart
sizeCImeanBart
sizeCIsdBart
typeIIBart
coverageBart

############## bcf #######################3

errormeanbcf<-mean(errorvectbcf)
errorsdbcf<-sd(errorvectbcf)
sizeCImeanbcf<-mean(sizeCIvectbcf)
sizeCIsdbcf<-sd(sizeCIvectbcf)
typeIIbcf<-typeIIbcf/itnumber
coveragebcf<-coveragebcf/itnumber

meanbcf
errormeanbcf
errorsdbcf
sizeCImeanbcf
sizeCIsdbcf
typeIIbcf
coveragebcf


############## Causal Bart Prop score matching #######################3

errormeanBartPC<-mean(errorvectBartPC)
errorsdBartPC<-sd(errorvectBartPC)
sizeCImeanBartPC<-mean(sizeCIvectBartPC)
sizeCIsdBartPC<-sd(sizeCIvectBartPC)
typeIIBartPC<-typeIIBartPC/itnumber
coverageBartPC<-coverageBartPC/itnumber

meanBartPC
errormeanBartPC
errorsdBartPC
sizeCImeanBartPC
sizeCIsdBartPC
typeIIBartPC
coverageBartPC
############## Causal Forest AIPW #######################3


errormeanForest<-mean(errorvectForest)
errorsdForest<-sd(errorvectForest)
sizeCImeanForest<-mean(sizeCIvectForest)
sizeCIsdForest<-sd(sizeCIvectForest)
typeIIForest<-typeIIForest/itnumber
coverageForest<-coverageForest/itnumber

meanForest
errormeanForest
errorsdForest
sizeCImeanForest
sizeCIsdForest
typeIIForest
coverageForest


############## Causal Forest TMLE #######################3


errormeanForest2<-mean(errorvectForest2)
errorsdForest2<-sd(errorvectForest2)
sizeCImeanForest2<-mean(sizeCIvectForest2)
sizeCIsdForest2<-sd(sizeCIvectForest2)
typeIIForest2<-typeIIForest2/itnumber
coverageForest2<-coverageForest2/itnumber

meanForest2
errormeanForest2
errorsdForest2
sizeCImeanForest2
sizeCIsdForest2
typeIIForest2
coverageForest2


############## OLS #######################3

meanOLS
errormeanOLS<-mean(errorvectOLS)
errorsdOLS<-sd(errorvectOLS)
sizeCImeanOLS<-mean(sizeCIvectOLS)
sizeCIsdOLS<-sd(sizeCIvectOLS)
typeIIOLS<-typeIIOLS/itnumber
coverageOLS<-coverageOLS/itnumber

meanOLS
errormeanOLS
errorsdOLS
sizeCImeanOLS
sizeCIsdOLS
typeIIOLS
coverageOLS

############## covariate ballancing #######################3

errormeanCovB<-mean(errorvectCovB)

errorsdCovB<-sd(errorvectCovB)
sizeCImeanCovB<-mean(sizeCIvectCovB)
sizeCIsdCovB<-sd(sizeCIvectCovB)
typeIICovB<-typeIICovB/itnumber
coverageCovB<-coverageCovB/itnumber

meanCovB
errormeanCovB
errorsdCovB
sizeCImeanCovB
sizeCIsdCovB
typeIICovB
coverageCovB
############## covariate ballancing #######################3

errormeanIPW<-mean(errorvectIPW)

errorsdIPW<-sd(errorvectIPW)
sizeCImeanIPW<-mean(sizeCIvectIPW)
sizeCIsdIPW<-sd(sizeCIvectIPW)
typeIIIPW<-typeIIIPW/itnumber
coverageIPW<-coverageIPW/itnumber


meanIPW
errormeanIPW
errorsdIPW
sizeCImeanIPW
sizeCIsdIPW
typeIIIPW
coverageIPW


############## propensity score matching #######################3

errormeanM<-mean(errorvectM)

errorsdM<-sd(errorvectM)
sizeCImeanM<-mean(sizeCIvectM)
sizeCIsdM<-sd(sizeCIvectM)
typeIIM<-typeIIM/itnumber
coverageM<-coverageM/itnumber

meanM
errormeanM
errorsdM
sizeCImeanM
sizeCIsdM
typeIIM
coverageM


