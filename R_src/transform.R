
#--------------Transformations---------------#
#fixing up the data nice and good like

corpus <- tm_map(corpus,stripWhitespace) #strip white space
corpus <- tm_map(corpus, tolower)

#Remove Stopwords
#train <- tm_map(train, removeWords,stopwords("english"))
#tm_map(train,stemDocument) #The stemmer native to tm is the stemDocument function, and you can call it with

#Create Document Term Matrix
dtm <- DocumentTermMatrix(corpus,
control = list(
            stemming=TRUE,
            stopwords=TRUE,
            minWordLength=3,
            removeNumbers=TRUE,
            removePunctuation=TRUE))
#clean dtm matrix
dtm_matrix_clean <- function(dtm_matrix) {
    out <- dtm_matrix[,grep('^[A-Za-z]+$',colnames(dtm_matrix))]
    return(out)
}
inspect(dtm)
dtm_matrix_clean(dtm)
inspect(dtm)

removeSparseTerms(dtm, 0.8)

tfidf <- tapply(dtm$v/row_sums(dtm)[dtm$i],
   dtm$j, mean)* log2(nDocs(dtm)/col_sums(dtm > 0))
   
dtm <- dtm[, tfidf > 0.1]
dtm <- dtm[row_sums(dtm) > 0, ]

#as.numeric(as.matrix(dtm)))


			
#dtm <- DocumentTermMatrix(train)
#dtm2 <- DocumentTermMatrix(train,  #dtm2 with weightTfIdf
#                          control = list(weighting =
#                                         function(x)
#                                         weightTfIdf(x, normalize =
#                                                     FALSE),
#                                         stopwords = TRUE,
#										 removePunctuation = TRUE))
#dtm3 <- DocumentTermMatrix(train,  #dtm3 with weightTfIdf and normalized
#                          control = list(weighting =
#                                         function(x)
#                                         weightTfIdf(x, normalize =
#                                                     TRUE),
#                                         stopwords = TRUE,
#										 removePunctuation = TRUE))
inspect(dtm[1:4,1:20])#Doc-1:4,#Terms-1:129909



#removeSparseTerms(dtm, 0.8)
#removeSparseTerms(dtm2, 0.8)
#removeSparseTerms(dtm3, 0.8)



findFreqTerms(dtm, 50) #View terms that appear atleast 50 times
findAssocs(dtm,"sunny",0.8) #what words show up in correlation with the term "sunny". It appears many negative terms

#dtm_mat <- DocumentTermMatrix(makeChunks(rs,500),list(weighting=weightBin)) 
