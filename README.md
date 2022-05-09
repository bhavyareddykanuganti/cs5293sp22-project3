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
- from textblob import TextBlob
# Function Description
## get_entity(text):
The get_entity function takes one argument as input. The argument is a string 
for which the features are to be generated. Firstly using the following command
're.findall(re.compile(r'\█+\s?\█\s?\█+\D█+'),text)' the redacted word is found.
Now, for the features I calculated the lenght of the redacted word, length of the sentences 
without stopwords and sentiment score. The function returns the dictionary of features.
## doextraction(glob_text)
## Assumptions and Bugs
## Execution
## External Links used




