
rm(list=ls())

setwd("D:\\Data Science - Course\\Text and Naive Bayes")
# read the sms data into the sms data frame
sms_raw = read.csv("sms_spam.csv", stringsAsFactors = FALSE)

# examine the structure of the sms data
str(sms_raw)

head(sms_raw)

# convert spam/ham to factor.
sms_raw$type <- factor(sms_raw$type)

prop.table(table(sms_raw$type))

# examine the type variable more carefully
str(sms_raw$type)
prop.table(table(sms_raw$type))

head(sms_raw)

# build a corpus using the text mining (tm) package
library(tm)
library(NLP)
sms_corpus <- Corpus(VectorSource(sms_raw$text))

# examine the sms corpus
print(sms_corpus)

?removeNumbers

?tolower



# clean up the corpus using tm_map()
corpus_clean <- tm_map(sms_corpus, tolower)
corpus_clean =  tm_map(corpus_clean, PlainTextDocument)
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords("english"))
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stripWhitespace)

# examine the clean corpus
inspect(sms_corpus[1:3])

class(corpus_clean)


# create a document-term sparse matrix

sms_dtm <- DocumentTermMatrix(corpus_clean)


# creating training and test datasets
sms_raw_train <- sms_raw[1:4169, ]
sms_raw_test  <- sms_raw[4170:5559, ]

sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test  <- sms_dtm[4170:5559, ]

sms_corpus_train <- corpus_clean[1:4169]
sms_corpus_test  <- corpus_clean[4170:5559]





head(sms_raw_train)


# # check that the proportion of spam is similar
# prop.table(table(sms_raw_train$type))
# prop.table(table(sms_raw_test$type))
#   
# library(e1071)
# 
# sms_classifier <- naiveBayes(sms_raw_train, sms_raw_train$type) # Type is a target variable 
# ## naivebayes function with two arguments, First is raw text, sencond is the label(target) 
# 
# ## Step 4: Evaluating model performance ----
# sms_test_pred <- predict(sms_classifier, sms_raw_test)
# 
# head(sms_raw_test$type)
# 
# head(sms_test_pred)
# 
# library(gmodels)
# 
# CrossTable(sms_raw_test$type, sms_test_pred)


library(e1071)
# word cloud visualization

library(wordcloud)

wordcloud(sms_corpus_train, min.freq = 50, random.order = FALSE)

# subset the training data into spam and ham groups
spam <- subset(sms_raw_train, type == "spam")
ham  <- subset(sms_raw_train, type == "ham")

wordcloud(spam$text, max.words = 40  )
wordcloud(ham$text, max.words = 40)

## only spam records corpus 

head(spam)


spam_corpus <- Corpus(VectorSource(spam$text))

spam_clean <- tm_map(spam_corpus, tolower)
spam_clean =  tm_map(spam_clean, PlainTextDocument)
spam_clean <- tm_map(spam_clean, removeNumbers)
spam_clean <- tm_map(spam_clean, removeWords, stopwords("english"))

spam_clean <- tm_map(spam_clean, removePunctuation)
spam_clean <- tm_map(spam_clean, stripWhitespace)


inspect(spam_clean)

spam_dtm = TermDocumentMatrix(spam_clean)

m <- as.matrix(spam_dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 25)

wordcloud(words = d$word, freq = d$freq, min.freq = 40,
          max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(6, "Dark2"))

mywords = as.character(d[1:20,1])

findAssocs(spam_dtm, terms = "call",corlimit = 0.2 )

findAssocs(spam_dtm, terms = "free",corlimit = 0.2 )

findAssocs(spam_dtm, terms = c("free","prize"),corlimit = 0.2 )
mywords


# indicator features for frequent words

myTerms <- c("amount", "free", "claim", "prize", "won", "draw", "latest", "win", "stop",  "contact", "send")
##inspect(DocumentTermMatrix(crude, list(dictionary = myTerms)))
##sms_dict <- Dictionary(findFreqTerms(sms_dtm_train, 5)) - function not available
sms_train <- DocumentTermMatrix(sms_corpus_train, list(dictionary = myTerms))
sms_test  <- DocumentTermMatrix(sms_corpus_test, list(dictionary = myTerms))



# convert counts to a factor
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
}

# apply() convert_counts() to columns of train/test data
sms_train <- apply(sms_train, MARGIN = 2, convert_counts)
sms_test  <- apply(sms_test, MARGIN = 2, convert_counts)



## Step 3: Training a model on the data ----
library(e1071)
sms_classifier <- naiveBayes(sms_train, sms_raw_train$type)
sms_classifier


## Step 4: Evaluating model performance ----
sms_test_pred <- predict(sms_classifier, sms_test)

sms_test_pred
table(sms_raw_test$type, sms_test_pred)

98/(98+30)
98/(98+85)

2*0.76*0.53/(0.76+0.53)


### create a new set of myterms 
myTerms2 = c( "claim", "prize", "won", "draw", "latest", "win", "stop", "contact", "send", "text", "txt", "you", "call" ,"reply")

sms_train2 <- DocumentTermMatrix(sms_corpus_train, list(dictionary = myTerms2))
sms_test2  <- DocumentTermMatrix(sms_corpus_test, list(dictionary = myTerms2))

## Convert counts to flags 

sms_train2 <- apply(sms_train2, MARGIN = 2, convert_counts)
sms_test2  <- apply(sms_test2, MARGIN = 2, convert_counts)



sms_classifier2 <- naiveBayes(sms_train2, sms_raw_train$type)
sms_classifier2


sms_test_pred2 <- predict(sms_classifier2, sms_test2)

## Step 4: Evaluating model performance of model2 ----

table(sms_raw_test$type, sms_test_pred2)

98/108

2*0.9*0.53/(0.53+0.9)

head(sms_raw_test$type)

?naiveBayes

### applying laplace smooting 

sms_classifier2 <- naiveBayes(sms_train2, sms_raw_train$type, laplace = 50)

sms_test_pred2 <- predict(sms_classifier2, sms_test2)


sms_classifier2 <- naiveBayes(sms_train2, sms_raw_train$type, laplace = 50)

## Step 4: Evaluating model performance of model2 ----

table(sms_raw_test$type, sms_test_pred2)

98/103
library(gmodels)

CrossTable( sms_raw_test$type, sms_test_pred)

CrossTable( sms_raw_test$type, sms_test_pred2)

table(sms_raw_test$type, sms_test_pred2)


117/(117+15)
117/(117+66)

2* 0.89*0.64/(0.89+0.64) 

 head(sms_test_test)

install.packages("gmodels")

library(gmodels)
CrossTable(sms_raw_test$type,sms_test_pred,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

?naiveBayes

## Step 5: Improving model performance ----
sms_classifier2 <- naiveBayes(sms_train, sms_raw_train$type, laplace = 1)
sms_test_pred2 <- predict(sms_classifier2, sms_test)
CrossTable(sms_raw_test$type,  sms_test_pred2)
         
