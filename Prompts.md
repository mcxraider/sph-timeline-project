# Prompt to try to determine need for headline:
Given a news headline, determine whether headline requires a contextual background (for example, in the form of a timeline). 
Guideline:
Needs timeline:  (1) Complext article (2) Specific details or referring to individuals or entities (3) Important and impactful article
Does not need timeline: (1) Explainer article (2) Forum, commentaries, podcast, inforgrahics (3) Accidents or independent events
Then reply in term of the need with a score 1 - 5, 1 means unnecessary, 5 means necessary, in JSON format, for example {"score": 3}.

Title: Swedish criminal gangs laundering money through Spotify: Report
Answer: 


# Prompt to generate tags for each article:
Given the text of a news article below, identify and list the most relevant tags that categorize the main themes, topics, or entities mentioned in the article. Focus on extracting key aspects like notable events, people, locations, and any significant issues discussed.

Article:
{Text}

Generate Tags:






