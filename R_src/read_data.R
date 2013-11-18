library(tm)
library(kernlab)
library(Rstem)
library(Snowball) 

#ALSO NOTE********
#Snowball package may have problems installing due to Java Virtual Machine issues. During installation, R froze,
#I shut R down and was then able to install. May also need to install Rstem, which is located on http://www.omegahat.org/
#go to packages > select repositories>omegahat, and then you should be able to install from there
#In recent versions R, can use following options to install Rstem:
#options(CRAN = c(getOption("CRAN"), "http://www.omegahat.org/R"))
#install.packages("Rstem")

#Set Directory where Text is located
dir <- "E:\\Github\\webmining\\data"
setwd(dir)

#--------------tm package-------------------#

train_raw <- read.table("train.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
head(train_raw)