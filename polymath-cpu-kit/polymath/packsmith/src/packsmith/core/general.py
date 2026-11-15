"""General-ability set for locality checks.

`general_qa`   short factual prompts with an expected substring (accuracy drop).
`general_text` neutral paragraphs for per-token KL and routing drift.
On the 5090, pass --general-text <file> to the harness to use a larger held-out set
(e.g. 200 paragraphs of WikiText-103 validation); these built-ins are a minimum for CI.
"""

GENERAL_QA = [
    ("The capital of France is", "Paris"), ("The capital of Japan is", "Tokyo"),
    ("Water is made of hydrogen and", "oxygen"), ("The largest planet in the solar system is", "Jupiter"),
    ("The opposite of hot is", "cold"), ("Two plus two equals", "4"),
    ("The chemical symbol for gold is", "Au"), ("The author of Hamlet is William", "Shakespeare"),
    ("The first month of the year is", "January"), ("A triangle has this many sides:", "3"),
    ("The currency of the United Kingdom is the pound", "sterling"), ("The freezing point of water in Celsius is", "0"),
    ("The longest river in Africa is the", "Nile"), ("Photosynthesis takes place in the", "chloroplast"),
    ("The speed of light is about 300,000 kilometres per", "second"), ("The capital of Italy is", "Rome"),
    ("Ten times ten equals", "100"), ("The planet known as the Red Planet is", "Mars"),
    ("The smallest prime number is", "2"), ("The language spoken in Brazil is", "Portuguese"),
]

GENERAL_TEXT = [
    "The river rose slowly through the spring, covering the low fields near the village. Farmers moved their animals to higher ground and waited for the water to fall.",
    "A compiler translates source code into machine instructions. Most modern compilers first build an intermediate representation, optimise it, and then generate code for a target processor.",
    "The museum opened a new gallery of ceramics last month. The collection includes bowls, tiles and storage jars from several centuries, arranged by region and period.",
    "Regular exercise improves cardiovascular health, strengthens muscles and can help with sleep. Experts recommend a mix of aerobic activity and strength training each week.",
    "The train left the station on time, but a signal fault near the junction delayed it by twenty minutes. Passengers were offered refreshments while engineers repaired the line.",
    "In mathematics, a prime number has exactly two divisors: one and itself. The study of how primes are distributed has occupied mathematicians for centuries.",
    "The committee reviewed the budget and asked each department to explain its spending. Several projects were postponed until the next financial year.",
    "Bees collect nectar from flowers and convert it into honey inside the hive. In the process they pollinate many plants, which is essential for agriculture.",
    "The novel follows a young engineer who returns to her home town to rebuild an old bridge. Along the way she reconnects with family and confronts her past.",
    "Cloud computing lets organisations rent storage and processing power instead of buying servers. Costs depend on usage, which can make budgeting harder.",
    "The mountain trail climbs steeply for the first two kilometres, then levels out along a ridge with views of the valley. Hikers should carry water and a map.",
    "A balanced diet includes vegetables, fruit, whole grains and protein. Reducing added sugar and salt is one of the simplest ways to improve health.",
]


# The anchor's questions (train_pack --anchor-qa default): the base model answers each in the question format the
# packs are trained in, and the KL anchor holds the pack to those answers, so a pack keeps the base's own answers to
# ordinary questions, not only its reading of the prose above. Varied subjects, short and long answers, a few that
# need steps. None is about a pack's subject.
GENERAL_QUESTIONS = [
    "What causes the seasons on Earth?", "Explain how a vaccine trains the immune system.",
    "Why does ice float on water?", "What is the difference between weather and climate?",
    "How does a bill become a law in a parliamentary democracy?", "What did the printing press change in Europe?",
    "Summarise the causes of the First World War in three points.", "What is inflation, and how do central banks respond to it?",
    "Write a Python function that returns the n-th Fibonacci number.", "What does a hash table do, and when is it slow?",
    "Explain recursion to a beginner with an example.", "What is the difference between TCP and UDP?",
    "A train leaves at 9:40 and arrives at 13:15. How long is the journey?", "If 3 pencils cost 45 rupees, what do 7 cost?",
    "Is 221 a prime number? Explain.", "What is 15% of 260?",
    "Give three tips for writing a clear email.", "How do I politely decline a meeting invitation?",
    "What is photosynthesis?", "Name the planets of the solar system in order.",
    "Who wrote Pride and Prejudice, and what is it about?", "What is the theme of a tragedy in literature?",
    "How should a beginner start learning to run?", "What is a balanced diet?",
    "Translate 'good morning, how are you?' into French.", "What is the past tense of 'swim'?",
    "Explain supply and demand with an example.", "What is the role of a judge in a trial?",
    "How do you change a flat tyre?", "What is the boiling point of water at sea level, and why does it change with altitude?",
    "Describe the water cycle.", "What are the main parts of a cell?",
    "What is machine learning, in two sentences?", "How does compound interest work?",
    "What is the capital of Australia?", "Why is the sky blue?",
    "List the steps to make a cup of tea.", "What is a metaphor? Give an example.",
    "How do I reverse a list in Python?", "What makes a good argument in an essay?",
]
