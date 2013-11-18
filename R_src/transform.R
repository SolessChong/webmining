
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
