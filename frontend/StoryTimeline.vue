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
            {{  }}
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
 {'title': "Emperor Kangxi's reign begins",
  'subtitle': '1661',
  'text': 'Emperor Kangxi, the second emperor of the Qing dynasty, began his long reign. He is considered one of the most capable and successful emperors of the Qing dynasty.',
  'icon': "mdi-crown"
},
 {'title': "Kangxi's reign: Westernization and exploration",
  'subtitle': '1661-1722',
  'text': 'Kangxi encouraged Western learning, invited Jesuit missionaries to teach mathematics, astronomy, and science, and established a shipbuilding industry. In addition, he sent several expeditions to explore the Pacific Ocean.',
  'icon': "mdi-ship-wheel"
},
 {'title': "Emperor Qianlong's reign begins",
  'subtitle': '1736',
  'text': "Emperor Qianlong, the sixth emperor of the Qing dynasty, began his reign. He was also a skilled leader and expanded the empire's territory, bringing Mongolia, Tibet, and Xinjiang under Qing control.",
  'icon': "mdi-crown"
},
 {'title': "Qianlong's reign: Cultural flourishing",
  'subtitle': '1736-1795',
  'text': 'Qianlong\'s reign saw a flourishing of Chinese culture. He supported the arts and the development of Confucianism, and in 1745, he moved the imperial court to Chengde, establishing the "Shengyang Palace" in the suburbs of Beijing.',
  'icon': "brush"
},
 {'title': 'Opium Wars',
  'subtitle': '1839-1842, 1856-1860',
  'text': "Britain declared war on China twice, known as the Opium Wars. The result was China's defeat, the loss of Hong Kong, and the forced opening of several treaty ports to British trade.",
  'icon': "mdi-sword"
},
 {'title': "Tongzhi's reign: Taiping Rebellion",
  'subtitle': '1861-1875',
  'text': 'Tongzhi, the thirteenth emperor of the Qing dynasty, faced the Taiping Rebellion, a major peasant revolt that lasted for more than a decade.',
  'icon': "food_bank"
},
 {'title': "Guangxu's reign: Modernization and reform",
  'subtitle': '1875-1908',
  'text': 'Emperor Guangxu, the twelfth emperor of the Qing dynasty, implemented a series of modernization and reform measures, including the establishment of a modern army, the expansion of railroads, and the founding of a modern navy.',
  'icon': "mdi-train"
},
 {'title': "Hundred Days' Reform",
  'subtitle': '1898',
  'text': "Empress Dowager Cixi orchestrated a brief but significant reform movement called the Hundred Days' Reform. The movement aimed to modernize the government and society, but Cixi ultimately abolished it due to conservative opposition.",
  'icon': "short_text"
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
  'text': 'A nationalist and anti-imperialist movement inspired by the Russian Revolution, demanding democratic reforms and political independence. '},
 {'title': '1927-1937',
  'subtitle': 'Northern Expedition',
  'text': 'Led by Chiang Kai-shek, a military campaign to unify China and expel foreign influences. The campaign successfully occupied Beiping (Beijing) in 1928. '},
 {'title': '1937-1945',
  'subtitle': 'Second Sino-Japanese War',
  'text': "China's resistance against Japanese aggression, which resulted in massive destruction and loss of life. "},
 {'title': '1949',
  'subtitle': "Establishment of the People's Republic of China",
  'text': 'Mao Zedong, the leader of the Chinese Communist Party, proclaimed the establishment of the new republic after defeating the nationalist government in the Chinese Civil War. '},
 {'title': '1949-1971',
  'subtitle': 'Cold War Era',
  'text': 'A period of hostility between China and the United States due to ideological differences and territorial disputes, culminating in the Cuban Missile Crisis in 1962. '},
 {'title': '1971',
  'subtitle': 'Re-establishment of Diplomatic Relations with the United States',
  'text': "President Richard Nixon's visit to China in 1972 initiated a rapprochement between the two nations, resulting in full diplomatic relations the following year. "},
 {'title': '1971',
  'subtitle': 'Establishment of the Republic of China (Taiwan)',
  'text': "Following the formation of the People's Republic of China, the Nationalist Party (KMT) government relocated to Taiwan and continued to claim sovereignty over the mainland. "},
 {'title': '1987',
  'subtitle': "Taiwan's Constitutional Amendment",
  'text': "The Constitutional Amendment of 1987 introduced a semi-presidential system, effectively giving power to the president, and was seen as a turning point in Taiwan's political development. "}]



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


