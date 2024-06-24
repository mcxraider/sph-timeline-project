<template>
  <div class="timeline-container">
    <q-timeline :layout="layout" color="secondary" class="timeline">

      <q-timeline-entry heading>
        <strong class="timeline-heading">{{ timelineHeading }}</strong>
        <br>
      </q-timeline-entry>

      <q-timeline-entry 
        v-for="(item, index) in timelineData" 
        :key="index"
        :title="item.Event_Summary ? item.Event_Summary : item.Event"
        :subtitle="item.Date"
        :side="index % 2 === 0 ? 'right' : 'left'"
        class="timeline-entry"
        :header-inset-level="0" 
      >

        <div v-if="item.Event_Summary && countWords(item.Event) > 30" class="event-details">
          <strong>{{ item.Event }}</strong>
        </div>
        
        <q-expansion-item
          expand-separator
          class="article-links"
          :label="'Explore relevant articles'"
          :switch-toggle-side="index % 2 === 1"
          :header-inset-level="0" 
          :content-inset-level="1"
        >
          
          <div v-for="(link, linkIndex) in item.Article_URL" :key="linkIndex" class="article-link">
              <a :href="link.url" target="_blank">{{ link.title }}</a>
          </div>
        </q-expansion-item>

      </q-timeline-entry>

    </q-timeline>
  </div>
</template>


<script setup>
// Import necessary functions and libraries
import { ref, onMounted, computed } from "vue";
import axios from "axios";
import { useQuasar, QTimeline, QTimelineEntry, QExpansionItem } from 'quasar';

// Initialize Quasar utilities
const $q = useQuasar();

// Define a computed property for responsive layout
const layout = computed(() => {
  // Return 'dense' layout for small screens, 'comfortable' for medium screens, and 'loose' for large screens
  return $q.screen.lt.sm ? "dense" : $q.screen.lt.md ? "comfortable" : "loose";
});

// Create reactive references to hold the timeline data and header
const timelineData = ref([]);
const timelineHeading = ref("");

// Function to fetch timeline data from the server
const fetchTimelineData = async () => {
  try {
    console.log("Fetching timeline data...");
    // Make a GET request to fetch the JSON data
    const response = await axios.get("http://localhost:3000");
    console.log("Data fetched:", response.data);
    
    // Set the fetched data to the reactive references
    const timelineString = response.data.Timeline;
    console.log(`This is the timeline String: ${timelineString}`);
    const parsedTimeline = JSON.parse(timelineString);
    console.log(`This is the parsed timeline: ${parsedTimeline}`);
    timelineData.value = parsedTimeline;

    // Update the reactive reference for the timeline header
    const timelineHeader= response.data.Timeline_header;
    timelineHeading.value = timelineHeader
    console.log(`This is the parsed timeline heading: ${timelineHeader}`);
    
  } catch (error) {
    console.error("Error fetching timeline data:", error);
  }
};

// Fetch timeline data when the component is mounted
onMounted(() => {
  fetchTimelineData();
});

const countWords = (text) => {
  return text.split(/\s+/).length;
};
</script>


<style scoped>

.timeline-container{
  margin: 20px;
}

</style>
