import sys
import os
import json
from tqdm import trange

from pymongo import MongoClient

from dotenv import load_dotenv
load_dotenv()
MONGO_URI = os.environ['MONGO_URI']

# Connect to the MongoDB client
client = MongoClient(MONGO_URI)
db = client['article_content_profile']


input_file = sys.argv[1]

# 'results.json'
with open(input_file) as fin:
    print('Loading data blob')
    data = json.load(fin)

    for i in trange(len(data)):
        doc = data[i]
        db['articles'].insert_one(doc['_source'])
    # end for
# end with