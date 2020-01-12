# data definitions
doc_trump =  "Mr. Trump became president after winning the political election. Though he lost the support of some republican friends, Trump is friends with President Putin."
doc_elections = "President Trump says Putin had no political interference is the election outcome. He says it was a witchhunt by political parties. He claimed President Putin is a friend who had nothing to do with the election."
doc_putin = "Post elections, Vladimir Putin became President of Russia. President Putin had served as the Prime Minister earlier in his political career."

documents = [doc_elections, doc_putin, doc_trump]
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# create the document term matrix
count_vectorizer = CountVectorizer(stop_words= 'english')
count_vectorizer = CountVectorizer()
sparse_matrix = count_vectorizer.fit_transform(documents)

# convert to pandas to see word frequencies
doc_term_matrix = sparse_matrix.todense()
df = pd.DataFrame(
    doc_term_matrix,
    columns=count_vectorizer.get_feature_names(),
    index = ['doc_elections', 'doc_putin', 'doc_trump']
)
print(df)

from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity(df, df))

