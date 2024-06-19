import { createApp } from 'vue';
import App from './App.vue';
import { Quasar } from 'quasar';
import 'quasar/dist/quasar.css';

// Create the Vue application instance
const app = createApp(App);

// Use Quasar with its configuration
app.use(Quasar, {
  config: {},
});

// Mount the Vue application
app.mount('#app');
