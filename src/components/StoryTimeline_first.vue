<template>
  <div>
    <div class="q-px-lg q-py-md">
      <q-timeline color="secondary">
        <q-timeline-entry heading> Timeline heading </q-timeline-entry>

        <q-timeline-entry
          v-for="(entry, index) in timelineData"
          :key="index"
          :title="entry.title"
          :subtitle="entry.date"
          :icon="entry.icon"
        >
          <div>{{ entry.description }}</div>
        </q-timeline-entry>
      </q-timeline>
    </div>

    <hr />

    <q-timeline :layout="layout" color="secondary" class="timeline">
      <q-timeline-entry heading>
        Timeline heading
        <br />
        ({{
          $q.screen.lt.sm ? "Dense" : $q.screen.lt.md ? "Comfortable" : "Loose"
        }}
        layout)
      </q-timeline-entry>

      <q-timeline-entry
        v-for="(item, index) in timelineData"
        :key="index"
        :title="item.title"
        :subtitle="item.date"
        :side="index % 2 === 0 ? 'right' : 'left'"
        :icon="item.icon"
      >
        <div>
          {{ item.description }}
        </div>
      </q-timeline-entry>
    </q-timeline>
  </div>
</template>

<script setup>
// This block is a special feature that allows you to use composition API functions directly and more concisely.

/* Quasar is a UI framework that offers various UI component and utilities.
Using useQuasar allows you to utilise these features in your component, such as responsive design utilities.*/
import { useQuasar } from "quasar";

/* ref creates reaective referneces that can store data and trigger reactivity.
Needed to create reactive data properties like 'timelineData' which can change and update the template when the data changes*/
import { ref } from "vue";

// onMounted is used to fetch data from the JSON file when the component is first rendered.
import { onMounted } from "vue";

// Needed for the layout property to dynamically adjust based on the screen size.
import { computed } from "vue";

/* Imports the Axios library, which is used to make HTTP requests.
It is needed to fetch data from the Timeline.json file located on the server*/
import axios from "axios";

const $q = useQuasar();

/*Defines a computed property named layout that dynamically returns a layout value based on the screen size.
Reason: Adjusts the layout of the timeline entries depending on the screen size to provide a responsive design.*/
const layout = computed(() => {
  return $q.screen.lt.sm ? "dense" : $q.screen.lt.md ? "comfortable" : "loose";
});

/* Creates a reactive reference named timelineData that is initialised as an empty array
It holds the fetched dat from the Timeline.json. 
As it is a reactive reference, it ensures that any changes to the Timeline*/
const timelineData = ref([]);

// Define an async function to fetch data from the JSON file
const fetchTimelineData = async () => {
  try {
    // Make a GET request to fetch the JSON data
    const response = await axios.get("/data_upload/Timeline.json");
    // Set the fetched data to the reactive reference
    timelineData.value = response.data;
  } catch (error) {
    // Log any errors that occur during the fetch operation
    console.error("Error fetching timeline data:", error);
  }
};

// Fetch the timeline data when the component is mounted
onMounted(fetchTimelineData);
</script>

<style scoped>
.timeline {
  margin: 10px;
}
</style>
