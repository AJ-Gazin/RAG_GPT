## Section 1

CLIENT: DAPPFORGE
INDUSTRY: BLOCKCHAIN
Supporting blockchain code development with a unique AI copilot


## Section 2

jump to
Project Overview
Challenges
Solution
Results
dAppForge
provides an AI-powered integrated development environment (IDE) plugin that reduces blockchain development time and lowers the barriers to entry in a complex industry.
dAppForge partnered with Neurons Lab to develop an AI-powered plugin that significantly speeds up and simplifies blockchain development.
Project Overview
Polkadot
developers face a learning curve and delays when building dApps, appchains, and parachains.
To support them, dAppForge partnered with Neurons Lab to develop a plugin that uses AI to streamline the development process. The MVP phase focused specifically on code autocompletion, concentrating on
Substrate
and FRAME.
We created a solution for fine-tuning and optimizing AI code suggestions and then building a
knowledge graph
(KG) to optimize the fine-tuned
LLM
response.
Feedback
“I received a new LLM version to test every two weeks, which was terrific. The speed and efficiency demonstrate how quickly you work and meet my needs. The team was incredibly responsive, even on weekends. I appreciate that I received prompt answers whenever I reached out on Slack.”
Christian Casini, dAppForge co-founder
Challenges
Help Polkadot developers write code faster
using the autocompletion AI tool. This will reduce development time, lower barriers to entry, and simplify the learning curve.
Build the first app that will not only be used as a co-pilot
but will also require less refactoring than the existing Polkadot co-pilot tool.
The tool will also allow work on error handling; the AI will explain the errors and how to fix them, so developers can learn to avoid them in the future.
Solution
Neurons Lab created a copilot for code assistance. The tech stack provided by
AWS Cloud
included AI infrastructure, foundational models, and tools leveraging two main forms of data:
Documentation from websites about how to write code, plus best practices
GitHub
code repository based on real projects
The stack extracts, processes, and feeds the data into the model and builds an AI orchestration, creating a retrieval augmented generation (
RAG
) system and a KG.
With context from actual data and projects, the vast KG powers the RAG to retrieve better contextual information
, enabling it to generate more accurate responses and reducing the frequency of AI hallucinations.
In addition, the application layer has the copilot itself with a code debugger and optimizer, capable of making suggestions and completing code.
Here is a presentation outlining the solution benefits from Christian Casini, dAppForge co-founder:
In terms of data collection:
From the GitHub code repository
, the solution parses the data, filters out specific file extension types, processes them, and creates document objects, removing any duplicates before creating a triplet. The triplet structure includes two connected nodes, a code description text, and sample code.
In the document process pipeline
, a website scraper detects URLs, parses text and code, converts it into document objects, removes duplicates, and feeds it into a triplet structure, a text-to-text relation with no code.
The code generation module has a KG construction process. The LLM quickly generates triplets based on preset prompts. Once generated, the module indexes the triplets and
converts them into the KG
.
After the user provides contextual code and submits it, in the search and retrieval phase, the system takes the query and searches the KG for the relevant information. Then it creates a sub-graph, a small representation of entities, nodes, and their relations relevant to the query.
Next, the code generation module generates the right code structure based on the sub-graph, dictated by a preset prompt to design the code so that it appears correctly. After generating the code, the system scores it:
If the score is good
: The system sends it to the user.
If the score is bad
: The system performs a more in-depth search of the KG, updates the context length, and repeats the search and retrieval until the score improves. When the code receives a good score, the system sends it back to the user.
This solution provides developers with more relevant and up-to-date code structures in a new industry with few best practices and examples widely available. It provides developers with better code, helps them to write code faster, and take inspiration from industry experts, growing the community.
Results
Neurons Lab extensively tested the performance of this KG+LLM model and saw truly remarkable results.
In these comprehensive assessments, participants interacted with the RAG application and provided feedback based on three distinct levels of performance, compared to an LLM model without a KG:
Success
:
Instances where the model excelled in delivering accurate results.
Failed
:
Scenarios where the model struggled to generate satisfactory outcomes.
Partial
:
Situations where the model’s performance was only partially correct.
The KG + LLM model increased the success rate by an incredible 34.75% compared to the isolated LLM approach.
It also showed a remarkable 54.62% decrease in partial successes, indicating more definitive outcomes. The failure rate also increased, demonstrating the model’s decisive nature in handling complex tasks.
KG + LLM:
Success Rate
: 41.18%
Failed
: 41.18%
Partial Success
: 17.65%
Vanilla LLM:
Success Rate
: 30.56%
Failed
: 27.78%
Partial Success
: 38.89%
For more information, learn all about
empowering RAG using a KG
written by Rahul Kumar, Head of AI Engineering at Neurons Lab.
About Neurons Lab
Neurons Lab is an AI consultancy that provides end-to-end services – from identifying high-impact AI applications to integrating and scaling the technology. We empower companies to capitalize on AI’s capabilities.
As an AWS Advanced Partner, our global team comprises data scientists, subject matter experts, and cloud specialists supported by an extensive talent pool of 500 experts. We solve the most complex AI challenges, mobilizing and delivering with outstanding speed to support urgent priorities and strategic long-term needs.
Ready to leverage AI for your business? Get in touch with the team
here
.
Get in touch
Leverage the power of AI for your business.
Related Stories
Penpot:
Elevating the world’s first open-source design platform with AI
Design
Explore story
pareIT:
Summarizing medical documents using artificial intelligence and machine learning
HealthTech, InsurTech
Explore story
Peak Defence:
Automating Business Operations with Generative AI in Cybersecurity
Cybersecurity
Explore story

