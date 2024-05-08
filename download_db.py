from pymongo import MongoClient
import json

#if pulling the db from mongo client
# client = MongoClient('mongodb://jerry:iloveaiengineering@localhost:27778/article_content_profile?directConnection=true')

# db = client['article_content_profile']
# collection = db['articles']


# documents = collection.find().sort('_id', -1).limit(2000)


batch_size = 500
batch_index = 0

for i, doc in enumerate(documents):
    if i % batch_size == 0:  
        if i > 0:  
            file.close()
        file = open(f'section{batch_index+1}.json', 'w')  
        batch_index += 1
    file.write(json.dumps(doc, default=str) + '\n')

file.close()
#client.close()



