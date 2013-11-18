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

train_tweet <- train_raw[,1:4] #id, tweet, state, location
train_label <- train_raw[,5:28] #sentiment, when, weather

#train <- Corpus(x=train_tweet, readerControl=list(language="eng"))
train <- Corpus(VectorSource(train_tweet))

label_senti <- train_label[,1:5] #sentiment
label_when <- train_label[,6:9] #when
label_weather <- train_label[,10:24] #weather

## Here needs to convert binary-classification to multi-classification
#senti <- label_senti
#when <- label_when
#weather <- label_weather
