# The Unredactor
## Author

### Bhavya Reddy Kanuganti
Email: bhavya.reddy.kanuganti-1@ou.edu

## Project Description
The data that has been shared with public has few sensitive details that has to be redacted. 
The sensitive data may include names, phone numbers, addressses and other data that may reveal the identity of a person.
In this project the redacted data has to be unredacted. The unredactor will take redacted documents and return the most 
likely candidates to fill in the redacted location.
The unredactor unredacts people names.

# Packages Installed
The following packages were used in the project:
- glob
- io
- os
- pdb
- re
- sys
- numpy 
- pandas
- nltk
  - sent_tokenize
  - word_tokenize
  - pos_tag
  - ne_chunk
  - stopwords
- sklearn
  - train_test_split
  - LogisticRegression
  - precision_score,recall_score,f1_score,accuracy_score
  - GaussianNB
- from textblob import TextBlob

# Function Description

## get_entity(text):

The get_entity function takes one argument as input. The argument is a string 
for which the features are to be generated. Firstly using the following command
're.findall(re.compile(r'\█+\s?\█\s?\█+\D█+'),text)' the redacted word is found.
Now, for the features I calculated the length of the redacted word, length of the sentences 
without stopwords and sentiment score. The function returns the dictionary of features.

## doextraction(glob_text)

In doextraction function the file is taken as input. First we add header to the file and
then seperate the data at tab space. Now if the second column in the data frame is training 
type then the corresponding sentence is sent to get_entity() function. The output of the get entity 
function is appended to a list and also the redacted names from dataframe are appended to 
another list. The list of names and features of training data are returned by doextraction function.

## get_entity_validation(text):

The get_entity_validation() function is similar to get_entity() function.As input, function accepts one argument.
The argument is a string that will be used to produce the features. To begin, use the command 
're.findall(re.compile(r'\█+\s?\█\s?\█+\D█+'),text)'. The redacted word has been identified.
Now, on to the features. I computed the length of the redacted word as well as the length of the sentences.
without stopwords and sentiment score. The feature dictionary is returned by the function.

## doextraction_validation(glob_text):

The doextraction_validation() function is similar to doextraction() function, the only difference is that
if the second column in the data frame is validation type instead of training
type then the corresponding sentence is sent to get_entity() function. The list of feature dictionaries are returned.

## get_entity_test(text):

The get_entity_test() function is similar to get_entity() function.As input, function accepts one argument.
The argument is a string that will be used to produce the features. To begin, use the command 
're.findall(re.compile(r'\█+\s?\█\s?\█+\D█+'),text)'. The redacted word has been identified.
Now, on to the features. I computed the length of the redacted word as well as the length of the sentences.
without stopwords and sentiment score. The feature dictionary is returned by the function.

## doextraction_test(glob_text):

The doextraction_validation() function is similar to doextraction() function, the only difference is that
if the second column in the data frame is testing type instead of training
type then the corresponding sentence is sent to get_entity() function. The list of feature dictionaries are returned.

## '__main__':
Here we call the doextraction, doextraction_validation and doextraction_test functions.
Using dict vectorizer we need to fit_transform the training features(x_train)
Now using GaussianNB() the training of features and names(Y_train) is done.
After performing  vec_features_validation=vec.fit_transform(x_validation) and 
vec_features_test = vec.fit_transform(x_test) we need to predict the output for validation data
using output=model.predict(vec_features_validation) and testing data using output1=model.predict(vec_features_test).
The precision, recall and f1-score is calculated for validation data and testing data.
The redacted names and also the precision, recall and f1 score are given as output.
### Logic to get latest tsv file
To get the latest tsv file we may use the url if the file and then decode it and create a dataframe, and perform the above mentioned functions

## Assumptions and Bugs

Only the given regex pattern can be found. 

If more features were given it was taking longer to run the code, so reduced the features and eventually
the precision, recall, and f1 score are reduced.

In the unredactor.tsv file there were errors after line 4286, so I trained till that line  


## Execution

The following command has been used after necessary installlations to run the code:
 
pipenv run python extractor.py unredactor.tsv

## External Links used

https://analyticsindiamag.com/how-to-obtain-a-sentiment-score-for-a-sentence-using-textblob/ --helped with sentiment score

https://www.geeksforgeeks.org/removing-stop-words-nltk-python/ --helped with stopwords

https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html --helped understand dict vectorizer




