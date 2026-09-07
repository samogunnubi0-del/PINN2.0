// api/_data/defaultDrillPacks.js

export const defaultGrammarPack = {
  grammarDrills: [
    {
      id: 'g1',
      sentencePart1: 'Marine biologist Sylvia Earle has logged thousands of hours underwater',
      sentencePart2: 'she has witnessed firsthand the degradation of coral reef ecosystems across the globe.',
      c1True: 'INDEPENDENT',
      c2True: 'INDEPENDENT',
      rule: 'Rule 1: Indep + Indep &rArr; Semicolon (;), Period (.), or Comma + FANBOYS.',
      explanation: 'Both clauses have a subject and a conjugated verb and can stand alone as complete sentences. A lone comma would create a comma splice. A semicolon is correct.',
      options: [
        { text: '; she has witnessed', correct: true },
        { text: ', she has witnessed', correct: false },
        { text: ' she has witnessed', correct: false },
        { text: ': although she has witnessed', correct: false }
      ]
    },
    {
      id: 'g2',
      sentencePart1: 'Although the new telescope array was engineered to operate in freezing conditions',
      sentencePart2: 'the sensor calibration required several hours of fine adjustments at room temperature.',
      c1True: 'DEPENDENT',
      c2True: 'INDEPENDENT',
      rule: 'Rule 2: Dep + Indep &rArr; Comma (,). Never use a semicolon after a dependent clause.',
      explanation: 'Clause 1 begins with the subordinating conjunction "Although", rendering it dependent. When a dependent clause leads, it must be followed by a comma, not a semicolon or period.',
      options: [
        { text: 'conditions, the sensor', correct: true },
        { text: 'conditions; the sensor', correct: false },
        { text: 'conditions. The sensor', correct: false },
        { text: 'conditions: and the sensor', correct: false }
      ]
    },
    {
      id: 'g3',
      sentencePart1: 'Architect Zaha Hadid incorporated a singular design philosophy into the museum expansion',
      sentencePart2: 'sweeping fluid geometries that eliminate traditional right angles.',
      c1True: 'INDEPENDENT',
      c2True: 'DEPENDENT',
      rule: 'Rule 3: Indep + Explanation &rArr; Colon (:) or Single Dash (&mdash;).',
      explanation: 'Clause 1 is a complete independent clause. Clause 2 is a noun phrase elaborating on and defining the singular design philosophy. A colon or dash is required.',
      options: [
        { text: 'expansion: sweeping fluid', correct: true },
        { text: 'expansion, sweeping fluid', correct: false },
        { text: 'expansion; sweeping fluid', correct: false },
        { text: 'expansion. Sweeping fluid', correct: false }
      ]
    },
    {
      id: 'g4',
      sentencePart1: 'The newly discovered hydrothermal vent species thrive at extreme ocean depths',
      sentencePart2: 'their metabolic pathways utilize hydrogen sulfide instead of sunlight for energy.',
      c1True: 'INDEPENDENT',
      c2True: 'INDEPENDENT',
      rule: 'Rule 1: Indep + Indep &rArr; Semicolon or Comma + FANBOYS.',
      explanation: '"Their metabolic pathways utilize..." is a full independent clause. It cannot be joined with a comma. Semicolon or period is required.',
      options: [
        { text: 'depths; their metabolic', correct: true },
        { text: 'depths, their metabolic', correct: false },
        { text: 'depths their metabolic', correct: false },
        { text: 'depths, and their metabolic pathways which', correct: false }
      ]
    },
    {
      id: 'g5',
      sentencePart1: 'While analyzing centuries of demographic records in northern Scandinavia',
      sentencePart2: 'historians uncovered clear evidence of seasonal migration cycles.',
      c1True: 'DEPENDENT',
      c2True: 'INDEPENDENT',
      rule: 'Rule 2: Dep + Indep &rArr; Comma (,).',
      explanation: 'Clause 1 is an introductory dependent modifier beginning with "While". It requires a comma before the main clause.',
      options: [
        { text: 'Scandinavia, historians', correct: true },
        { text: 'Scandinavia; historians', correct: false },
        { text: 'Scandinavia: historians', correct: false },
        { text: 'Scandinavia. Historians', correct: false }
      ]
    }
  ],
  svDrills: [
    {
      words: [
        { text: 'The', type: 'other' },
        { text: 'constellation', type: 'subject', isSubject: true },
        { text: 'of', type: 'prep' },
        { text: 'automated', type: 'prep' },
        { text: 'weather', type: 'prep' },
        { text: 'stations', type: 'prep' },
        { text: 'scattered', type: 'prep' },
        { text: 'across', type: 'prep' },
        { text: 'the', type: 'prep' },
        { text: 'Mojave', type: 'prep' },
        { text: 'Desert', type: 'prep' }
      ],
      verbPrompt: '[transmit / transmits] atmospheric telemetry directly to NOAA headquarters.',
      subject: 'constellation (Singular)',
      correctVerb: 'transmits',
      options: [
        { text: 'transmits (Singular verb matching singular subject "constellation")', correct: true },
        { text: 'transmit (Plural verb erroneously matching "stations" or "Desert")', correct: false },
        { text: 'have transmitted (Plural verb)', correct: false },
        { text: 'are transmitting (Plural verb)', correct: false }
      ],
      explanation: 'The true grammatical subject is the singular noun "constellation." The intervening prepositional phrases "of automated weather stations" and "scattered across the Mojave Desert" are distractors. Singular subject requires "transmits."'
    },
    {
      words: [
        { text: 'Fluctuations', type: 'subject', isSubject: true },
        { text: 'in', type: 'prep' },
        { text: 'the', type: 'prep' },
        { text: 'abundance', type: 'prep' },
        { text: 'of', type: 'prep' },
        { text: 'phytoplankton', type: 'prep' },
        { text: 'near', type: 'prep' },
        { text: 'the', type: 'prep' },
        { text: 'ocean', type: 'prep' },
        { text: 'surface', type: 'prep' }
      ],
      verbPrompt: '[influences / influence] the migratory patterns of baleen whales.',
      subject: 'Fluctuations (Plural)',
      correctVerb: 'influence',
      options: [
        { text: 'influence (Plural verb matching plural subject "Fluctuations")', correct: true },
        { text: 'influences (Singular verb erroneously matching "surface")', correct: false },
        { text: 'has influenced (Singular verb)', correct: false },
        { text: 'is influencing (Singular verb)', correct: false }
      ],
      explanation: 'The subject is the plural noun "Fluctuations." All intervening words between the subject and verb are prepositional modifiers and must be ignored. Plural subject requires "influence."'
    },
    {
      words: [
        { text: 'The', type: 'other' },
        { text: 'restoration', type: 'subject', isSubject: true },
        { text: 'of', type: 'prep' },
        { text: 'historically', type: 'prep' },
        { text: 'degraded', type: 'prep' },
        { text: 'coastal', type: 'prep' },
        { text: 'salt', type: 'prep' },
        { text: 'marshes', type: 'prep' },
        { text: 'and', type: 'prep' },
        { text: 'barrier', type: 'prep' },
        { text: 'islands', type: 'prep' }
      ],
      verbPrompt: '[mitigates / mitigate] storm surge damage along the vulnerable shoreline.',
      subject: 'restoration (Singular)',
      correctVerb: 'mitigates',
      options: [
        { text: 'mitigates (Singular verb matching singular subject "restoration")', correct: true },
        { text: 'mitigate (Plural verb erroneously matching "marshes" or "islands")', correct: false },
        { text: 'have mitigated (Plural verb)', correct: false },
        { text: 'are mitigating (Plural verb)', correct: false }
      ],
      explanation: 'The true subject is the singular noun "restoration." The intervening compound prepositional phrases "of historically degraded coastal salt marshes and barrier islands" do not alter verb number. Singular subject requires "mitigates."'
    },
    {
      words: [
        { text: 'Advances', type: 'subject', isSubject: true },
        { text: 'in', type: 'prep' },
        { text: 'high-resolution', type: 'prep' },
        { text: 'cryo-electron', type: 'prep' },
        { text: 'microscopy', type: 'prep' },
        { text: 'developed', type: 'prep' },
        { text: 'by', type: 'prep' },
        { text: 'biochemical', type: 'prep' },
        { text: 'laboratories', type: 'prep' }
      ],
      verbPrompt: '[enables / enable] researchers to determine macromolecular protein structures at atomic resolution.',
      subject: 'Advances (Plural)',
      correctVerb: 'enable',
      options: [
        { text: 'enable (Plural verb matching plural subject "Advances")', correct: true },
        { text: 'enables (Singular verb erroneously matching "microscopy" or "laboratories")', correct: false },
        { text: 'has enabled (Singular verb)', correct: false },
        { text: 'is enabling (Singular verb)', correct: false }
      ],
      explanation: 'The grammatical subject is the plural noun "Advances." The prepositional modifiers "in high-resolution cryo-electron microscopy" and participial modifier "developed by biochemical laboratories" must be ignored. Plural subject requires "enable."'
    },
    {
      words: [
        { text: 'Each', type: 'subject', isSubject: true },
        { text: 'of', type: 'prep' },
        { text: 'the', type: 'prep' },
        { text: 'twelve', type: 'prep' },
        { text: 'archaeological', type: 'prep' },
        { text: 'field', type: 'prep' },
        { text: 'teams', type: 'prep' },
        { text: 'surveying', type: 'prep' },
        { text: 'the', type: 'prep' },
        { text: 'remote', type: 'prep' },
        { text: 'Andean', type: 'prep' },
        { text: 'valley', type: 'prep' }
      ],
      verbPrompt: '[relies / rely] on satellite lidar imaging to detect buried stone foundations.',
      subject: 'Each (Singular indefinite pronoun)',
      correctVerb: 'relies',
      options: [
        { text: 'relies (Singular verb matching singular indefinite pronoun "Each")', correct: true },
        { text: 'rely (Plural verb erroneously matching "teams" or "valley")', correct: false },
        { text: 'have relied (Plural verb)', correct: false },
        { text: 'are relying (Plural verb)', correct: false }
      ],
      explanation: 'The subject is the singular indefinite pronoun "Each." While the prepositional phrase "of the twelve archaeological field teams" contains plural nouns, "Each" is mathematically singular and requires "relies."'
    }
  ]
};

export const defaultEvidencePack = {
  evidenceDrills: [
    {
      id: 'ev1',
      domain: 'Textual Evidence',
      sentences: [
        { id: 's1', text: 'In deep hydrothermal vent communities, ecosystems exist in complete isolation from solar radiation.' },
        { id: 's2', text: 'Rather than relying on photosynthetic organisms as the primary trophic base, these ecosystems depend on chemolithoautotrophic bacteria.' },
        { id: 's3', text: 'These specialized bacteria metabolize dissolved hydrogen sulfide and methane emerging from geothermal fissures, synthesizing organic compounds.' },
        { id: 's4', text: 'Giant tube worms (Riftia pachyptila) harbor millions of these bacteria within a specialized organ called a trophosome, obtaining nourishment directly from bacterial endosymbionts without possessing a digestive tract.' }
      ],
      proofSentenceId: 's4',
      prompt: 'Which sentence from the passage provides the most direct textual evidence that certain vent organisms depend entirely on internal symbiotic bacteria for metabolic nutrition rather than ingestion?',
      options: [
        { text: 'Sentence 4 (explaining that tube worms lack a digestive tract and obtain food directly from bacterial endosymbionts)', correct: true },
        { text: 'Sentence 1 (stating ecosystems exist in isolation from sunlight)', correct: false },
        { text: 'Sentence 2 (noting ecosystems depend on chemotrophic bacteria)', correct: false },
        { text: 'Sentence 3 (describing bacterial metabolism of hydrogen sulfide)', correct: false }
      ],
      explanation: 'Sentence 4 explicitly specifies that the tube worms possess no digestive tract and derive nourishment directly from the internal bacteria inside the trophosome. Selecting any other sentence is an OUTSIDE_ASSUMPTION error.'
    },
    {
      id: 'ev2',
      domain: 'Quantitative Evidence',
      hasChart: true,
      sentences: [
        { id: 's1', text: 'Ecologists tracked annual nitrogen runoff and seasonal macroalgal biomass in estuaries along the Chesapeake Bay from 2018 to 2024.' },
        { id: 's2', text: 'Researchers hypothesized that wet spring seasons with heavy agricultural runoff directly correlate with elevated peak summer algal blooms.' },
        { id: 's3', text: 'The collected field data, recorded across twelve monitoring stations, demonstrates this dynamic with high fidelity.' }
      ],
      proofSentenceId: 's2',
      chartData: {
        title: 'Spring Runoff vs Summer Algal Biomass (2018-2022)',
        points: [
          { year: '2018', runoff: 42, algal: 18 },
          { year: '2019', runoff: 78, algal: 64 },
          { year: '2020', runoff: 55, algal: 36 },
          { year: '2021', runoff: 88, algal: 82 },
          { year: '2022', runoff: 32, algal: 14 }
        ],
        series: [
          { key: 'runoff', name: 'Runoff (kt)', color: '#C57B36' },
          { key: 'algal', name: 'Algal (g/m²)', color: '#256B4E' }
        ]
      },
      chartAssertionPrompt: 'As spring nitrogen runoff increases, peak summer algal biomass',
      expectedTrend: 'INCREASES',
      prompt: 'Which assertion best reflects the quantitative evidence shown in the data while supporting the researchers\' hypothesis?',
      options: [
        { text: 'In 2021, when spring nitrogen runoff reached a recorded peak of 88 kilotons, summer algal biomass also achieved its highest level at 82 g/m².', correct: true },
        { text: 'Algal biomass remained consistent regardless of spring runoff fluctuations.', correct: false },
        { text: 'In 2019, runoff decreased while algal biomass increased to 64 g/m².', correct: false },
        { text: 'Macroalgal blooms were highest in years with the lowest agricultural runoff.', correct: false }
      ],
      explanation: 'The chart demonstrates a positive correlation: years with higher runoff (2019, 2021) showed corresponding peaks in algal biomass. Choice A accurately cites the peak coordinates matching the hypothesis.'
    },
    {
      id: 'ev3',
      domain: 'Textual Evidence',
      sentences: [
        { id: 's1', text: 'Planetary scientists analyzing atmospheric samples collected by the Curiosity rover on Mars detected episodic spikes in background methane concentrations.' },
        { id: 's2', text: 'Because atmospheric methane is rapidly degraded by solar ultraviolet radiation within roughly three hundred years, any detected methane must originate from relatively recent processes.' },
        { id: 's3', text: 'Some researchers hypothesized that serpentinization reactions between subsurface olivine minerals and liquid water could generate abiotic methane without biological intervention.' },
        { id: 's4', text: 'However, isotopic mass spectrometry revealed an anomalous carbon-12 enrichment in the rover samples that closely mirrors the metabolic fractionation signatures characteristic of methanogenic archaea on Earth.' }
      ],
      proofSentenceId: 's4',
      prompt: 'Which sentence from the passage provides the most direct textual evidence that the Martian methane spikes could potentially reflect biological metabolic activity rather than purely abiotic geochemical reactions?',
      options: [
        { text: 'Sentence 4 (explaining that anomalous carbon-12 enrichment mirrors the metabolic signatures of terrestrial archaea)', correct: true },
        { text: 'Sentence 1 (stating the rover detected episodic spikes in methane)', correct: false },
        { text: 'Sentence 2 (noting methane degrades rapidly under ultraviolet radiation)', correct: false },
        { text: 'Sentence 3 (describing abiotic serpentinization reactions with olivine)', correct: false }
      ],
      explanation: 'Sentence 4 explicitly identifies carbon-12 isotope fractionation that mirrors terrestrial methanogenic archaea, directly supporting the biological origin hypothesis over abiotic mineral reactions.'
    },
    {
      id: 'ev4',
      domain: 'Quantitative Evidence',
      hasChart: true,
      sentences: [
        { id: 's1', text: 'Oceanographers deployed deep-sea conductivity, temperature, and depth (CTD) profilers to map thermal gradients across the bathypelagic zone.' },
        { id: 's2', text: 'Marine physical models predict that incoming solar warmth dissipates completely below the photic zone, creating a steep thermocline where water temperatures plummet before stabilizing near freezing.' },
        { id: 's3', text: 'Telemetry recorded across five distinct ocean depths demonstrates this thermal stratification profile.' }
      ],
      proofSentenceId: 's2',
      chartData: {
        title: 'Ocean Depth vs Water Temperature Profile',
        points: [
          { depth: '100m', temp: 22 },
          { depth: '300m', temp: 15 },
          { depth: '600m', temp: 8 },
          { depth: '1000m', temp: 4 },
          { depth: '2000m', temp: 2 }
        ],
        series: [
          { key: 'temp', name: 'Temperature (°C)', color: '#416B95' }
        ]
      },
      chartAssertionPrompt: 'As ocean depth increases from 100m to 2000m, water temperature',
      expectedTrend: 'DECREASES',
      prompt: 'Which assertion best reflects the quantitative evidence recorded by the profilers while verifying the oceanographers\' model?',
      options: [
        { text: 'Water temperature drops precipitously from 22°C at 100 meters to 4°C at 1000 meters, verifying the steep thermocline predicted by the model.', correct: true },
        { text: 'Water temperature remained constant at 15°C across all depths from 100m to 2000m.', correct: false },
        { text: 'Water temperature increased as depth extended beyond 600 meters.', correct: false },
        { text: 'The warmest recorded water temperature occurred at a depth of 2000 meters.', correct: false }
      ],
      explanation: 'The data demonstrates a steep negative gradient from 22°C down to 4°C between 100m and 1000m, directly validating the thermocline model stated in Sentence 2. Choice A accurately cites coordinates.'
    },
    {
      id: 'ev5',
      domain: 'Textual Evidence',
      sentences: [
        { id: 's1', text: 'Anthropologists excavating the Olorgesailie basin in Kenya recovered thousands of Acheulean stone handaxes dating from 1.2 million to 500,000 years ago.' },
        { id: 's2', text: 'For decades, scholars debated whether early hominins selected specific lithic raw materials purely based on immediate local availability or deliberate mechanical suitability.' },
        { id: 's3', text: 'Geochemical X-ray fluorescence sourcing revealed that hominins consistently transported durable, fine-grained quartzite and obsidian from volcanic outcrops over thirty kilometers away, bypassing the brittle basalt boulders located directly beside the riverbank campsite.' },
        { id: 's4', text: 'Furthermore, micro-wear edge analysis indicated that handaxes made from imported quartzite retained their sharp cutting edges through twice as many butchery cycles as basalt tools.' }
      ],
      proofSentenceId: 's3',
      prompt: 'Which sentence from the passage provides the strongest evidence that hominins deliberately expended significant effort to procure higher-quality toolstone rather than opportunistically using whatever rock was immediately at hand?',
      options: [
        { text: 'Sentence 3 (revealing hominins transported quartzite and obsidian from over thirty kilometers away while ignoring adjacent basalt boulders)', correct: true },
        { text: 'Sentence 1 (stating thousands of handaxes were recovered in the Olorgesailie basin)', correct: false },
        { text: 'Sentence 2 (stating scholars debated whether material selection was opportunistic or deliberate)', correct: false },
        { text: 'Sentence 4 (demonstrating quartzite handaxes retained sharp edges longer than basalt tools)', correct: false }
      ],
      explanation: 'Sentence 3 provides decisive behavioral evidence of intentional procurement effort: hominins transported preferred raw materials over 30 kilometers despite immediately accessible local basalt.'
    }
  ]
};
