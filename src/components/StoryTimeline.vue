<template>
  <div>
      <q-timeline :layout="layout" color="secondary" class="timeline">
        
        
        <q-timeline-entry heading>
          <strong>Timeline heading</strong>
          <hr>
          <br>
        </q-timeline-entry>


        <q-timeline-entry v-for="(item, index) in timelineData" :key="index"
          :title="item.Event"
          :subtitle="item.Date"
          :side="index % 2 === 0 ? 'right' : 'left'"
        >
        <div>
          {{item.Contextual_Annotation}}
        </div>
        </q-timeline-entry>  
        
        
      </q-timeline>
  </div>
</template>

<script setup>
// Import necessary functions and libraries
import { ref, onMounted, computed } from "vue";
import axios from "axios";
import { useQuasar, QTimeline, QTimelineEntry } from 'quasar';

// Initialize Quasar utilities
const $q = useQuasar();

// Define a computed property for responsive layout
const layout = computed(() => {
  // Return 'dense' layout for small screens, 'comfortable' for medium screens, and 'loose' for large screens
  return $q.screen.lt.sm ? "dense" : $q.screen.lt.md ? "comfortable" : "loose";
});

// Create a reactive reference to hold the timeline data
const timelineData = ref([]);

const fetchTimelineData = async () => {
        try {
        console.log("Fetching timeline data...");
        // Make a GET request to fetch the JSON data
        const response = await axios.get("http://localhost:3000");
        console.log("Data fetched:", response.data);
        // Set the fetched data to the reactive reference
        const timelineString = response.data.Timeline;
        console.log(`This is the timeline String: ${timelineString}`);
        const parsedTimeline = JSON.parse(timelineString);
        console.log(`This is the parsed timeline: ${parsedTimeline}`)
        // Set the parsed data to the reactive reference
        timelineData.value = parsedTimeline;
        } catch (error) {
        console.error("Error fetching timeline data:", error);
        }
        };
        
onMounted(() => {
      fetchTimelineData();
    });


</script>

<style scoped>
</style>
