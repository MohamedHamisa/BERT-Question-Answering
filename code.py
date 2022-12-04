!pip install cdqa  #CLosed Domain Quastions & Answers is an easy-to-use python package to implement a QA pipeline

import os
import pandas as pd
from ast import literal_eval
#one of the helper functions that helps traverse an abstract syntax tree. 
#This function evaluates an expression node or a string consisting of a Python literal or container display.

from cdqa.utils.converters import pdf_converter
from cdqa.pipeline import QAPipeline
from cdqa.utils.download import download_model

download_model(model='bert-squad_1.1', dir='./models')

!ls models

!mkdir docs

!wget -P ./docs/ https://s2.q4cdn.com/299287126/files/doc_financials/2020/q3/AMZN-Q3-2020-Earnings-Release.pdf
!wget -P ./docs/ https://s2.q4cdn.com/299287126/files/doc_financials/2020/q2/Q2-2020-Amazon-Earnings-Release.pdf
!wget -P ./docs/ https://s2.q4cdn.com/299287126/files/doc_financials/2020/Q1/AMZN-Q1-2020-Earnings-Release.pdf
!wget -P ./docs/ https://s2.q4cdn.com/299287126/files/doc_news/archive/Amazon-Q4-2019-Earnings-Release.pdf
!wget -P ./docs/ https://s2.q4cdn.com/299287126/files/doc_news/archive/Q3-2019-Amazon-Financial-Results.pdf

df = pdf_converter(directory_path='./docs/')
df.head()

pd.set_option('display.max_colwidth', -1)
df.head()

cdqa_pipeline = QAPipeline(reader='./models/bert_qa.joblib', max_df=1.0)

cdqa_pipeline.fit_retriever(df=df)

import joblib 
# transparent disk-caching of functions and lazy re-evaluation (memoize pattern) easy simple parallel computing
# Joblib is a set of tools to provide lightweight pipelining in Python
joblib.dump(cdqa_pipeline, './models/bert_qa_custom.joblib')

cdqa_pipeline=joblib.load('./models/bert_qa_custom.joblib')
cdqa_pipeline

query = 'How much is increase in operating cash flow?'
prediction = cdqa_pipeline.predict(query, 3)

cdqa_pipeline

prediction

query = 'What is latest earnings per share?'
cdqa_pipeline.predict(query)

query = 'How many jobs are created in 2020?'
prediction = cdqa_pipeline.predict(query)

print('query: {}'.format(query))
print('answer: {}'.format(prediction[0]))
print('title: {}'.format(prediction[1]))
print('paragraph: {}'.format(prediction[2]))

query = 'How many full time employees are on Amazon roll?'
prediction = cdqa_pipeline.predict(query)

print('query: {}'.format(query))
print('answer: {}'.format(prediction[0]))
print('title: {}'.format(prediction[1]))
print('paragraph: {}'.format(prediction[2]))

query = 'General Availability of which AWS services were announced?'
prediction = cdqa_pipeline.predict(query, n_predictions=5)

prediction

query = 'What is the impact of COVID on business?'
prediction = cdqa_pipeline.predict(query, n_predictions=5)

print('query: {}'.format(query))
print('answer: {}'.format(prediction[0]))
print('title: {}'.format(prediction[1]))
print('paragraph: {}'.format(prediction[2]))

from google.colab import drive
drive.mount('/content/drive')

!cp ./models/bert_qa_custom.joblib '/content/drive/MyDrive/Colab Notebooks/models/'
