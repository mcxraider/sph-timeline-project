<template>
  <div id="app">
    <StoryTimeline />
  </div>
</template>

<script>
import StoryTimeline from "./components/StoryTimeline_hybrid.vue";

export default {
  name: "App",
  components: {
    StoryTimeline,
  },
};
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>


