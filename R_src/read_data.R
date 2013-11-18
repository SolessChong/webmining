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


#--------------Transformations---------------#
#fixing up the data nice and good like

train <- tm_map(train,stripWhitespace) #strip white space
train <- tm_map(train, tolower)

#Remove Stopwords
train <- tm_map(train, removeWords,stopwords("english"))
tm_map(train,stemDocument) 

#Create Document Term Matrix
dtm <- DocumentTermMatrix(train)
inspect(dtm[1:10,1:10])

findFreqTerms(dtm, 50) #View terms that appear atleast 50 times
findAssocs(dtm,"sunny",0.8) #what words show up in correlation with the term "sunny". It appears many negative terms


#dtm_mat <- DocumentTermMatrix(makeChunks(rs,500),list(weighting=weightBin)) 
