
// Connect to MongoDB
// mongosh mongodb://jerry:iloveaiengineering@localhost:27778/article_content_profile?directConnection=true

// Creating a user in 
db.createUser({
  user: "jerry",
  pwd: "iloveaiengineering",
  roles: [
     { role: "dbAdmin", db: "article_content_profile" },
  ] 
})


db.grantRolesToUser("jerry", [ 
    { role: "dbOwner", db: "article_content_profile" },
]);

// How to connect into a mongodb database that is running on docker
mongodb://[<username>:<password>@]hostname0<:port>[,hostname1:<port1>][,hostname2:<port2>][...][,hostnameN:<portN>]







