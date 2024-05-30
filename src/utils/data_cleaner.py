import json
import re
from json.decoder import JSONDecodeError

def clean_llm_score(output):
    text = output.parts[0].text.replace("```", '').replace('json','')
    result = json.loads(text)
    return result

def clean_output(output):
    try:
        updated_timeline = json.loads(output)
        return updated_timeline
    except JSONDecodeError:
        #try 1: Ensuring that the string ends with just the open and close lists brackets
        try:
            new_output = re.search(r'\[[^\]]*\]', output).group(0)
        except AttributeError:
            new_output = re.search(r'\{.*?\}', output, re.DOTALL).group(0)  
        updated_timeline = json.loads(new_output)
        return updated_timeline

def clean_sort_timeline(timelines, df_retrieve):  
    generated_timeline = []
    for idx, line in timelines.items():
        indiv_timeline = clean_output(line)
        if type(indiv_timeline) == list:
            for el in indiv_timeline:
                generated_timeline.append(el)
        else:
            generated_timeline.append(indiv_timeline)
    unsorted_timeline = []

    for event in generated_timeline:
        article_index = event["Article"] - 1
        event["Article_id"] = df_retrieve.iloc[article_index].id
    for event in generated_timeline:
        del event["Article"]
        unsorted_timeline.append(event)  
        
    timeline = sorted(unsorted_timeline, key=lambda x:x['Date'])
    timeline = [event for event in timeline if event['Date'].lower()!= 'nan']
    for event in timeline:
        date = event['Date']
        if date.endswith('-XX-XX'):
            event['Date'] = date[:4]
        elif date.endswith('-XX'):
            event['Date'] = date[:7]
    return timeline
