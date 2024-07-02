# Prompt to determine timeline heading of an article:

I would like to create a timeline of events based on the title of an article Given this article title below, what would be a generalised, suitable name for for a timeline for such an article that will provide a reader contextual information about a timeline of events regarding the article.

article title: Japan to provide $88 million in additional humanitarian aid to Palestinians. 
only return the fill in the blank part 
Answer: Timeline of <fill in this blank>



# Prompt to try to determine need for headline:
You are a highly intelligent AI tasked with analyzing articles to determine whether generating a timeline of events leading up to the key event in the article would be beneficial. 
Consider the following factors to make your decision:
    1. **Significance of the Event**:
       - Does the event have a significant impact on a large number of people, industries, or countries?
       - Are the potential long-term consequences of the event important?

    2. **Controversy or Debate**:
       - Is the event highly controversial or has it sparked significant debate?
       - Has the event garnered significant media attention and public interest?

    3. **Complexity**:
       - Does the event involve multiple factors, stakeholders, or causes that make it complex?
       - Does the event have deep historical roots or is it the culmination of long-term developments?

    4. **Personal Relevance**:
       - Does the event directly affect the reader or their community?
       - Is the event of particular interest to the reader due to economic implications, political affiliations, or social issues?

    5. Educational Purposes:
       - Would a timeline provide valuable learning or research information?

    Here is the information for the article:
    Title:{title}
    Text: {text}
    

    Based on the factors above, decide whether generating a timeline of events leading up to the key event in this article would be beneficial. 
    Your answer will include the need for this article to have a timeline with a score 1 - 5, 1 means unnecessary, 5 means necessary. It will also include the main reason for your choice.
    {format_instructions}    
    ANSWER:


# Prompt to generate tags for each article:
Task Description: Given the following news article, identify and suggest 8 most relevant tags that best categorize the geographical locations mentioned, main events, topics, entities.
The tags should be concise, informative, and reflect the content accurately to facilitate effective searching and organization within a database.
Do not use abbreviations in the tags. (Example: COE should be Certificate of Entitlement)
The tags should also be ranked in decreasing order of importance and relevance in the article.
Combined Title and Summaries:
{text}
    
    Formatting convention: List the tags to me in this example format:
    Singapore, Big family, climbing, Baby, crying, hungry
    
    Ensure that the tags generated follow the formatting convention very closely, and the response should not include any "-" or backslash n.
    Generated tags:
    
    Check again that the format follows the formatting convention stated above




# Prompt to generate tineline:
Given an article, containing a publication date, title, and content, your task is to construct a detailed timeline of events leading up to the main event described in the first article.
Begin by thoroughly analyzing the title, content, and publication date of the article to understand the main event in the first article. 
the dates are represented in YYYY-MM-DD format. Identify events, context, and any time references such as "last week," "last month," or specific dates. 
The article could contain more or one key events. 
If the article does not provide a publication date or any events leading up to the main event, return NAN in the Date field, and 0 i the Article Field


Construct the Timeline:
Chronological Order: Organize the events chronologically, using the publication dates and time references within the articles.
Detailed Descriptions: Provide detailed descriptions of each event, explaining how it relates to the main event of the first article.
Contextual Links: Use information from the articles to link events together logically and coherently.
Handle Ambiguities: If an article uses ambiguous time references, infer the date based on the publication date of the article and provide a clear rationale for your inference.

Contextual Links:
External Influences: Mention any external influences (e.g., global conflicts, economic trends, scientific discoveries) that might have indirectly affected the events.
Internal Issues: Highlight any internal issues or developments (e.g., political changes, organizational restructuring, societal movements) within the entities involved that might have impacted the events.
Efforts for Improvement: Note any indications of efforts to improve the situation (e.g., policy changes, strategic initiatives, collaborative projects) despite existing challenges.

Be as thorough and precise as possible, ensuring the timeline accurately reflects the sequence and context of events leading to the main event.

Article:
{text}

{format_instructions}
Check and ensure again that the output follows the format instructions above very strictly. 




