<template>
  <div>

    <div class="q-px-lg q-py-md">
      <q-timeline color="secondary">
        <q-timeline-entry heading>
          Timeline heading
        </q-timeline-entry>

        <q-timeline-entry
          title="Event Title"
          subtitle="February 22, 1986"
        >
        <div>
            Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
          </div>
        </q-timeline-entry>

        <q-timeline-entry
          title="Event Title"
          subtitle="February 21, 1986"
          icon="delete"
        >
          <div>
            Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
          </div>
        </q-timeline-entry>

        <q-timeline-entry heading>
          November, 2017
        </q-timeline-entry>

        <q-timeline-entry
          title="Event Title"
          subtitle="February 22, 1986"
          avatar="https://cdn.quasar.dev/img/avatar2.jpg"
        >
          <div>
            Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
          </div>
        </q-timeline-entry>

        <q-timeline-entry
          title="Examine me"
          subtitle="February 22, 1986"
          avatar="https://cdn.quasar.dev/img/avatar2.jpg"
        >
          <div>

            <q-expansion-item
              expand-separator
              icon="zoom_in"
              label="Expand list of events"
              caption="53 events"
              header="Account settings"
            
            >
              <q-timeline-entry
                title="Event Title"
                subtitle="February 22, 1986"
              >
                <div>
                  Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
                </div>
              </q-timeline-entry>

              <q-timeline-entry
                title="Event Title"
                subtitle="February 22, 1986"
                color="orange"
                icon="done_all"
              >
                <div>
                  Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
                </div>
              </q-timeline-entry>

              <q-timeline-entry
                title="Event Title"
                subtitle="February 22, 1986"
              >
                <div>
                  Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
                </div>
              </q-timeline-entry>
            </q-expansion-item>
          </div>
        </q-timeline-entry>



        <q-timeline-entry
          title="Event Title"
          subtitle="February 22, 1986"
        >
          <div>
            Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
          </div>
        </q-timeline-entry>
      </q-timeline>
    </div>

    <hr>

    <q-timeline :layout="layout" color="secondary" class="timeline">
      <q-timeline-entry heading>
        Timeline heading
        <br>
        ({{$q.screen.lt.sm ? 'Dense' : ($q.screen.lt.md ? 'Comfortable' : 'Loose')}} layout)
      </q-timeline-entry>

      <q-timeline-entry v-for="(item, index) in timeline_data" :key="index"
        :title="item.title"
        :subtitle="item.subtitle"
        :side="index % 2 === 0 ? 'right' : 'left'"
        :icon="item.icon"
      >
        <div>
          {{item.text}}
        </div>
      </q-timeline-entry>

      <q-timeline-entry heading>Republic Era</q-timeline-entry>

      <q-timeline-entry v-for="(item, index) in timeline_data_2" :key="index"
        :title="item.title"
        :subtitle="item.subtitle"
        :side="index % 2 === 0 ? 'right' : 'left'"
        :icon="item.icon"
        :body="item.text"
      >
      </q-timeline-entry>

      
    </q-timeline>
  </div>
</template>




<script setup>
import { useQuasar } from 'quasar'
import { computed } from 'vue'

const $q = useQuasar()
const layout = computed(() => {
    return $q.screen.lt.sm ? 'dense' : ($q.screen.lt.md ? 'comfortable' : 'loose')
});


const timeline_data = [
 {'title': 'Qing conquest of China',
  'subtitle': '1644',
  'text': 'The Qing dynasty, originally Manchu tribes, defeated the Ming dynasty and established the Qing dynasty in Beijing.',
  'icon': "flag"
},
 {'title': 'Xinhai Revolution',
  'subtitle': '1911-1912',
  'text': 'The Xinhai Revolution, also known as the 1911 Revolution, overthrew the Qing dynasty and established the Republic of China.',
  'icon': "star"
}
];

const timeline_data_2 = [{'title': '1911',
  'subtitle': 'Xinhai Revolution',
  'text': 'Beginning of the republican revolution that overthrew the Qing dynasty, led by Dr. Sun Yat-sen. '},
 {'title': '1912',
  'subtitle': 'Establishment of the Republic of China',
  'text': 'The first president of the republic was Dr. Sun Yat-sen, but his presidency was short-lived, as Yuan Shikai took over the following year. '},
 {'title': '1921-1927',
  'subtitle': 'May Fourth Movement',
  'text': 'A nationalist and anti-imperialist movement inspired by the Russian Revolution, demanding democratic reforms and political independence. '}]

</script>

<script>
export default {
  name: "StoryTimeline",
  components: {
  },
};
</script>



<style scoped>

.timeline {
  margin: 10px;
}

</style>


