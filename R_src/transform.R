
#--------------Transformations---------------#
#fixing up the data nice and good like

train <- tm_map(train,stripWhitespace) #strip white space
train <- tm_map(train, tolower)

#Remove Stopwords
train <- tm_map(train, removeWords,stopwords("english"))
tm_map(train,stemDocument) #The stemmer native to tm is the stemDocument function, and you can call it with

#Create Document Term Matrix

dtm <- DocumentTermMatrix(train)
dtm2 <- DocumentTermMatrix(train,  #dtm2 with weightTfIdf
                          control = list(weighting =
                                         function(x)
                                         weightTfIdf(x, normalize =
                                                     FALSE),
                                         stopwords = TRUE,
										 removePunctuation = TRUE))
dtm2 <- DocumentTermMatrix(train,  #dtm3 with weightTfIdf and normalized
                          control = list(weighting =
                                         function(x)
                                         weightTfIdf(x, normalize =
                                                     TRUE),
                                         stopwords = TRUE,
										 removePunctuation = TRUE))
inspect(dtm[1:10,1:10])

findFreqTerms(dtm, 50) #View terms that appear atleast 50 times
findAssocs(dtm,"sunny",0.8) #what words show up in correlation with the term "sunny". It appears many negative terms

#dtm_mat <- DocumentTermMatrix(makeChunks(rs,500),list(weighting=weightBin)) 
