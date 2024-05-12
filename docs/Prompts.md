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
