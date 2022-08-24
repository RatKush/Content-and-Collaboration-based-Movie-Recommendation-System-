from sklearn.feature_extraction.text import CountVectorizer
text1= ["London Paris London", "Paris Paris London"]
cv= CountVectorizer()
count_matrix= cv.fit_transform(text1)
#print (count_matrix.vocabulary)
print (count_matrix.toarray())
