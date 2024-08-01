# Imports the Google Cloud client library
from google.cloud import language_v1

# Instantiates a client
client = language_v1.LanguageServiceClient()


def extract_sections(client, document):
    """
    Extracts and classifies text sections from the document.

    Args:
        client (language_v1.LanguageServiceClient): The LanguageService client instance.
        document (language_v1.Document): The document to classify.

    Returns:
        list: A list of tuples containing category names and their confidence scores, 
              sorted by confidence score in descending order.
    """
    # Classify text sections
    sections = client.classify_text(request={"document": document})
    
    # Filter and sort sections with confidence greater than 0.5
    sections_scores = sorted(
        [(cat.name, cat.confidence) for cat in sections.categories if cat.confidence > 0.5], 
        reverse=True, key=lambda x: x[1]
    )
    
    return sections_scores
# end def 

def extract_categories(client, document):
    """
    Extracts and categorizes the content of the document.

    Args:
        client (language_v1.LanguageServiceClient): The LanguageService client instance.
        document (language_v1.Document): The document to categorize.

    Returns:
        list: A list of tuples containing category names and their confidence scores,
              sorted by confidence score in descending order.
    """
    # Moderate text categories
    categories = client.moderate_text(request={"document": document})
    
    # Filter and sort categories with confidence greater than 0.5
    categories_scores = sorted(
        [(cat.name, cat.confidence) for cat in categories.moderation_categories if cat.confidence > 0.5],
        reverse=True, key=lambda x: x[1]
    )
    
    return categories_scores
# end def 

def analyse_entities(client, document):
    """
    Analyzes entities in the document and returns the most salient ones.

    Args:
        client (language_v1.LanguageServiceClient): The LanguageService client instance.
        document (language_v1.Document): The document to analyze.

    Returns:
        list: A list of tuples containing entity names and their salience scores,
              sorted by salience score in descending order.
    """
    # Analyze entities in the document
    entity = client.analyze_entities(request={"document": document})
    
    # Collect raw entities categorized by their type
    raw_entities = {}
    for en in entity.entities:
        type_ = str(en.type_)[5:].lower()
        if type_ not in raw_entities:
            raw_entities[type_] = []
        raw_entities[type_].append(en)
    # end for
    
    # Sort entities within each category by salience
    entities = {}
    for tag_category, tags in raw_entities.items():
        tag_ls = [(en.name, en.salience) for en in raw_entities[tag_category]]
        tag_ls.sort(reverse=True, key=lambda x: x[1])
        entities[tag_category] = tag_ls
    # end for
    
    # Filter and select the top entities
    seen_tags = []
    final_tags = []
    for tag_category, tag_ls in entities.items():
        filtered_ls = [(key, value) for key, value in tag_ls if value > 0.01]
        if len(filtered_ls) > 4:
            filtered_ls = filtered_ls[:4]
        # end if
        
        for tag, sal in filtered_ls:
            if tag not in seen_tags:
                seen_tags.append(tag)
                final_tags.append((tag, sal))
            else:
                # Find tag index
                for i in range(len(final_tags)):
                    if final_tags[i][0] == tag:
                        existing_salience = final_tags[i][1]
                        break
                # end for
                if sal > existing_salience:
                    final_tags.pop(i)
                # end if
        # end for
    # end for
    
    # Sort final tags by salience
    final_tags.sort(key=lambda x: x[1], reverse=True)
    
    return final_tags
# end def 


def extract_entities(text):
    """
    Extracts entities, categories, and sections from the given text.

    Args:
        text (str): The input text to analyze.

    Returns:
        dict: A dictionary containing the extracted entities, categories, and sections.
    """
    # Create a document object from the input text
    document = language_v1.types.Document(
        content=text, type_=language_v1.types.Document.Type.PLAIN_TEXT
    )

    # Analyze entities in the document
    cleaned_entities = analyse_entities(client, document)
    
    # Extract categories from the document
    categories = extract_categories(client, document)
    
    # Extract sections from the document
    sections = extract_sections(client, document)
    
    # Compile the results into a properties dictionary
    properties = {}
    properties['entities'] = cleaned_entities
    properties['categories'] = categories
    properties['sections'] = sections
    
    print(properties)
    return properties
# end def extract_entities


sample_1 = '''\
GENEVA – The remains of a climber discovered in the Swiss Alps in 2022 have been identified as \
those of a British mountaineer who went missing 52 years ago, local police said on Thursday.\
It is the latest in a series of discoveries of remains of long-missing climbers revealed as the Alps’ glaciers melt and recede because of global warming.\
The climber was reported missing in July 1971, \
but search teams at the time turned up nothing, said police in the canton of Valais, south-west Switzerland.Then on Aug 22, 2022,\
two climbers found human remains on the Chessjengletscher glacier near Saas-Fee,\
an Alpine village in the Saas Valley.It took a year to identify the person as experts worked their \
way through the case files of missing climbers.Finally, with the help of Interpol Manchester and the police in Scotland,\
a relative was found and a DNA sample allowed them to identify the British mountaineer\
, \the Swiss police said in a statement.\
The climber was formally identified on Aug 30.Increasing numbers of human remains, some of them of climbers missing for decades,\
have been discovered in recent years as glaciers in the Alps melt because of global warming.In late July, \
the remains of a German climber who went missing in 1986 were discovered on another Swiss glacier. AFP      ",'''

sample_2 = """\
EU countries are still discussing the idea of a humanitarian ceasefire in the war between Israel and Hamas but there are different ways to get much-needed aid to Palestinians in Gaza,\
Swedish foreign minister Tobias Billstrom said on Monday."The discussions are ongoing, but the question really isn't about a ceasefire, \
but about how to bring aid forward and that can be done in very many different ways," he told reporters after a meeting of EU foreign ministers in Luxembourg.\
He said Sweden preferred a U.N. proposal for a humanitarian corridor.Earlier on Monday, \
EU foreign policy chief Josep Borrell voiced support for a "humanitarian pause" but some of the bloc's foreign ministers expressed reservations about the idea. REUTERS"""


if __name__=="__main__":
    sample_1_entities = extract_entities(sample_1)
    sample_2_entities = extract_entities(sample_2)