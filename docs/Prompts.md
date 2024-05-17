# Prompt to try to determine need for headline:
Given a news headline, determine whether headline requires a contextual background (for example, in the form of a timeline). 
Guideline:
Needs timeline:  (1) Complext article (2) Specific details or referring to individuals or entities (3) Important and impactful article
Does not need timeline: (1) Explainer article (2) Forum, commentaries, podcast, inforgrahics (3) Accidents or independent events
Then reply in term of the need with a score 1 - 5, 1 means unnecessary, 5 means necessary, in JSON format, for example {"score": 3}.

Title: Swedish criminal gangs laundering money through Spotify: Report
Answer: 


# Prompt to generate tags for each article:
Task Description: Given the combined title and text of a news article below, focus on identifying the most relevant tags that categorize the main   topic, people, locations or entities mentioned in the article. 
Article:
{Text}

Generate Tags:




# Prompt to generate tineline:
Given a series of articles, each containing a publication date, title, and content, your task is to construct a detailed timeline of events leading up to the main event described in the first article.
Analyze the First Article: Begin by thoroughly analyzing the title, content, and publication date of the first article to understand the main event.
Use Subsequent Articles: For each following article, examine the title, content, and publication date. Identify events, context, and any time references such as "last week," "last month," or specific dates.

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
Series of Articles: 
{text}
Now, I want you to carefully analyse the timeline and cross reference that the information is correct, and that it corresponds with the correct date.
Then, check carefully to ensure that the dates are arranged in chronological order.
FORMAT FOR OUTPUT: Return them as a JSON object that the keys are Dates and Events the explanation, respectively.
This is a good example of how to return the output (Do not include this in the timeline):
{{"Date":"2000-05-22",
"Event": "Reports emerge about the successful evacuation of thousands of residents, and the government begins planning long-term recovery efforts.",
"Article": 1}},
{{"Date":"2023-08-01",
"Event": "Major flood hits the southern region of Country X, causing significant damage and displacing thousands of residents.",
"Article": 1}}

Check again that the output is formatted as a JSON object that the keys are Dates and Event and Article.






