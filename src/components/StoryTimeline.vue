<template>
  <div>
    <div class="q-px-lg q-py-md">
      <q-timeline :layout="layout" color="secondary">
        <q-timeline-entry heading>
          <strong>Timeline heading</strong>
        </q-timeline-entry>

        <hr />

        <q-timeline-entry v-for="(entry, index) in timelineData" :key="index">
          <div class="entry-Date"><strong>Date: </strong>{{ entry.Date }}</div>

          <div class="entry-Event">
            <strong>Event: </strong>{{ entry.Event }}
          </div>

          <div class="entry-Context">
            <strong>Event Context: </strong>{{ entry.Contextual_Annotation }}
          </div>

          <div class="entry-Article">
            <strong>Article ID: </strong> {{ entry.Article }}
          </div>

          <hr />
        </q-timeline-entry>
      </q-timeline>
    </div>
  </div>
</template>

<script setup>
// Import necessary functions and libraries
import { useQuasar } from "quasar";
import { ref, onMounted, computed } from "vue";
import axios from "axios";

// Initialize Quasar utilities
const $q = useQuasar();

// Define a computed property for responsive layout
const layout = computed(() => {
  // Return 'dense' layout for small screens, 'comfortable' for medium screens, and 'loose' for large screens
  return $q.screen.lt.sm ? "dense" : $q.screen.lt.md ? "comfortable" : "loose";
});

// Create a reactive reference to hold the timeline data
const timelineData = ref([]);

// Define an async function to fetch data from the JSON file
const fetchTimelineData = async () => {
  try {
    console.log("Fetching timeline data...");
    // Make a GET request to fetch the JSON data
    const response = await axios.get("/data_upload/Timeline.json");
    console.log("Data fetched:", response.data);
    // Set the fetched data to the reactive reference
    timelineData.value = response.data;
  } catch (error) {
    // Log any errors that occur during the fetch operation
    console.error("Error fetching timeline data:", error);
  }
};

// Fetch the timeline data when the component is mounted
onMounted(() => {
  console.log("Component mounted");
  fetchTimelineData();
});
</script>

<style scoped>
.timeline {
  margin: 10px;
}
</style>
