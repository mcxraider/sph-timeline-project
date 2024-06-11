

def split_batches(timeline, max_batch_size=30):
    n = len(timeline)
    if n <= max_batch_size:
        return [timeline]
    
    num_batches = n // max_batch_size
    remainder = n % max_batch_size
    
    if remainder > 0 and remainder < max_batch_size // 2:
        num_batches -= 1
        remainder += max_batch_size

    batches = []
    start = 0
    for i in range(num_batches):
        end = start + max_batch_size
        batches.append(timeline[start:end])
        start = end
    
    if remainder > 0:
        batches.append(timeline[start:start + remainder])
    
    return batches


def load_mongodb():
    print("Fetching article data from MongoDB...\n")
    
    # Connect to the MongoDB client
    try:
        db = mongo_client[config["database"]["name"]]
        train_documents = db[config["database"]["train_collection"]].find()
        print("Train data successfully fetched from MongoDB\n")
    except Exception as error: 
        print(f"Unable to fetch train data from MongoDB. Check your connection the database...\nERROR: {error}\n")
        sys.exit()   
    try:
        test_docs = db[config["database"]["test_collection"]].find()
        print("Test data successfully fetched from MongoDB\n")
    except:
        print(f"Unable to fetch test data from MongoDB. Check your connection the database...\nERROR: {error}\n")
        sys.exit()
    return train_documents, test_docs
