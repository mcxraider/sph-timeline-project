# News StoryTimeline Project
This Github Repo contains all the code that use for the creation of 2 versions of the timeline.

This Repo contain following structure:
```
├── notebooks
│   └── hierarchical_clustering_experiment.ipynb
├── public
│   ├── index.html
│   └── st_logo.ico
├── server
│   ├── node_modules/
│   ├── package-lock.json
│   ├── package.json
│   └── server.js
├── src
│   ├── components
│   │   ├── StoryTimeline_official.vue
│   │   ├── Version1_timeline.vue
│   │   └── Version2_timeline.vue
│   ├── utils
│   │   ├── __init__.py
│   │   ├── google_nlp_tag_generation.py
│   │   ├── graio_interface.py
│   │   ├── main.js
│   │   ├── verison_2_timeline.py
│   │   └── version_1_timeline.py
│   ├── App.vue
│   └── main.js
├── .env
├── .gitignore
├── babel.config.js
├── config.yaml
├── env_template.txt
├── jsconfig.json
├── package-lock.json
├── package.json
├── README.md
├── requirements.txt
└── vue.config.js

```

# MongoDB configuration:
- Adjust config.yaml file for proper connection to local MongoDB.
- Change the Mongo client string
- Change the name of the collections in the database accordingly


# Documentation
<br>
Check this page for more experiments detail and results: [Documentation]([URL](https://sph.atlassian.net/wiki/spaces/DTM/pages/1792934620/News+StoryTimeline+Project))



