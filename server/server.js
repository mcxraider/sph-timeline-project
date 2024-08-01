// server/server.js

const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');

const app = express();
const port = 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Connect to MongoDB
const mongoConnectionString = "mongodb://127.0.0.1:27017/timeline-project";

mongoose.connect(mongoConnectionString, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

const db = mongoose.connection;
db.on('error', console.error.bind(console, 'connection error:'));
db.once('open', () => {
  console.log('Connected to MongoDB');
});

//Define a schema and model

// Schema below for version 1 of timeline
const timelineSchema = new mongoose.Schema({
    Article_id: String,
    Article_Title: String,
    Timeline: String,
    Timeline_header: String
  }, {collection: 'generated_timelines'});


// Schema below for version 2 of timeline
// const timelineSchema = new mongoose.Schema({
//     Article_id: String,
//     Article_Title: String,
//     Timeline: String,
//     Timeline_header: String
//   }, {collection: 'generated_hybrid_timelines'});


const TimelineEntry = mongoose.model('TimelineEntry', timelineSchema);

// Define a route to fetch data
app.get('/', async (req, res) => {
    try {
      const timelineEntry = await TimelineEntry.findOne();
      res.json(timelineEntry);
    } catch (error) {
      res.status(500).send(error);
    }
  });

// Start the server
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
