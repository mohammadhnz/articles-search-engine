# Articles Search Engine
In this project, we used multiple information retrieval methods to design search engine for scintific articles.
## How to Run!
first you need to install requirements, run elasticsearch and build elasticsearch indexes. simply you can run this commands:
```
pip install -r requirments.txt
python manage.py search_index --rebuild
```

Beside these command, you need to download models and put them in results.
Then you have to initialize elasticsearch with your data. 
#### Example
```
import json
from search_engine.documents import Article

with open("data.json") as file:
    data = json.load(file)
for item in data:
    Article.create(item['paperId'], item['title'], item['authors'], item['abstract']).save()
    
```
## Code Explanation
In search_engine app we have the services package. which implemented each retrievla method in single class. such as ElasticQueryService, TransformerQueryService, ...

### ElasticQueryService
In this query service, we used elasticsearch to search among our articles. in this search we use fuzziness to search between titles, abstracts and authors name and we return related articles.

#### Code Example
```
from search_engine.services import elastic_query_service

elastic_query_service.query(
    query="Transformers nlp",
    result_count=10
)
```
### BooleanQueryService
In this class we implemnted boolean retrival. in this case we ,created a linked list of articles ids for each word. and then for every query we seperate the query with conditional words (and, or). then we apply a boolean search between titles and authors' names.
#### Code Example
```
from search_engine.services import boolean_query_service

boolean_query_service.query(
    query="nlp and transformers",
    result_count=10
)
```
### ClassificationService
After training our BERT model using a trainer and saving the results in a directory named "./results/checkpoint-500" ; in this class we:
<br>•   First,  load our model and tokenizer using AutoTokenizer.from_pretrained 
<br>•  Second, we define a TextClassificationPipeline using our model and tokenizer
<br>•  Third, by giving that pipeline our input text, we can get the predicted class
### ClusterService
In this class, we first load both:
<br>•  Our model (which we trained before and used 4000 paper's abstracts to do so) 
<br>•  Our vectorizer (which we used to vectorize our training data)
Then, we then use vectorizer.transform on our input text and then use model.predict to get our predicted cluster for that input. 

### TFIDFQueryService
We calculate each article's score based on the tf_idf data we gathered earlier and return articles with the highest scores.
#### Code Example
```
from search_engine.services import tf_ids_query_service

tf_ids_query_service.query(
    query="expressive neural encoders",
    result_count=10
)
```
### TransformerQueryService
We used CrossEncoder with "cross-encoder/ms-marco-TinyBERT-L-2-v2". In this case we use CrossEncoder predict method to check if query is related to document or not.
#### Code Example
```
from search_engine.services import transformer_query_service

transformer_query_service.query(
    query="trained classifier",
    result_count=10
)
```
### WORDEMBEDDINGQueryService
We calculate the query's average vector based on the model previously trained, then calculate the cosine similarity of the query and all articles and return most similar articles.
#### Code Example
```
from search_engine.services import word_embedding_query_service

word_embedding_query_service.query(
    query="trained classifier",
    result_count=10
)
```
## Query Explination
First, we sort articles based on their cosine similarity from query. then we name the top 10 results as good articles and the rest as bad articles. next, we add good articles average vector to query and subtract bad articles average vector from query and search again with new vector.
