## Section 1

CLIENT: VISA
INDUSTRY: FINANCIAL SERVICES
Streamlining localized multi-channel marketing with an LLM constructor for VISA


## Section 2

“The Neurons Lab team demonstrated
deep expertise in GenAI
and worked diligently to build a tailored solution that has
significantly improved our marketing personalization
.”
Anna Melnychuk, Senior Marketing Manager VISA
Visa, the world’s leading global digital payments network, recently partnered with Neurons Lab. Together we empowered the marketing teams to craft localized, compelling messages for multiple channels faster and more efficiently.
Partner Overview
If you have a debit or credit card, there’s a good chance it’s one of the
4.3 billion cards issued
by
Visa
. As the
world’s leading digital payments network
, Visa connects businesses, financial institutions, governments, and individuals across 200 countries, processing 283 billion payment transactions globally in 2023 alone.
Visa facilitates a wide variety of payment types – from P2P and B2C to B2B and G2C. Visa’s global network includes banks, governments, businesses of all sizes, and individual consumers worldwide. Therefore, its messaging must always be
tailored to the target audience’s context and location
to deliver maximum impact.
Project Overview
To help marketing teams overcome these challenges,
Visa enlisted our help in creating a simplified constructor
powered by
Large Language Model (LLM)
technology. The constructor would be available to marketing managers to
streamline message creation, translation, and optimization
.
To
maximize marketers’ productivity
, the generative AI module needed to be seamlessly integrated with its Campaign Atlas portal and allow for customizable parameter input (e.g., target language, channel, text length, and tone of voice selection).
The new tool had to support
20 different languages.
This included
9 must-have languages
and
11 nice-to-have languages
, plus copy optimization for social media platforms (Facebook, Instagram, LinkedIn, Twitter), messengers (Viber, Telegram), email, and sales scripts.
Business Challenges
Before working with Neurons Lab
, Visa had to tailor its messaging to each location and leverage various outreach channels. It is a
very time-consuming
endeavor as marketing specialists have to grapple with the following challenges:
Translating marketing copies accurately
Adapting them to specific communication channels
Ensuring they remain consistent in tone and messaging with the global corporate guidelines
This
efficiency gap in marketing
was the key challenge Visa turned to us with. The company wanted to leverage GenAI to increase the marketing teams’ productivity by tackling:
Translation localization across 20 different languages
Quality assurance to prevent spelling and grammar mistakes
Copy optimization for each marketing channel
Tone and messaging consistency, especially for smaller bank marketing specialists
Solution
The solution to these challenges
was a simplified constructor that empowers marketing managers to
streamline and optimize marketing messages
using LLM technology.
The constructor includes pre-built templates and customizable parameters that users can leverage to generate effective messaging for various communication channels efficiently.
LLM architecture
The user inputs a query using the web app dashboard in the Campaign Atlas portal, including:
Query
Language selection
Tone selection
Channel selection
Next, the
natural language processing (NLP)
module built on
LangChain
receives the input and activates components for tone and channel augmentation, as well as translation.
Prompting options for tone include
informative, authoritative, enthusiastic, inspirational
, and so on. Channel choice includes social media, email, sales rep script, digital ad, and more.
Later in the chain, the NLP module handles translations. Initially, it leverages
GPT-4
, followed by custom models such as
MarianMT
and
MBart
from
Hugging Face
to
improve accuracy and quality
.
Near the end of the chain, the user receives several different intermediate output responses on the Campaign Atlas portal before choosing a text operation.
These post-processing text operations include requests to
make the text shorter or longer, add emojis, simplify the language
, improve the writing, etc.
This response enters the NLP module to perform the post-processing request, and the user receives the final output.
The platform stores a history of all intermediate and final outputs in an
extensive database
, allowing users to fetch past responses again in the future.
API and Cloud architecture
Neurons Lab implemented the LLM model using LangChain between the user inputs and the outputs within a
LangServe
wrapper—
a Python framework well suited to APIs
.
The main chain checks the input configuration, while the sub-chains handle
tone, channel augmentation, and text customization
. Suggestions are provided as the outputs.
To deploy the API solution, Neurons Lab used the
AWS Elastic Container Service (ECS)
alongside
LangSmith
for the prompt storage logs and evaluation.
This dashboard supports prompt management and the playground while providing
exceptional log and chain tracking
.
Results
The solution provides:
An integrated GenAI module
: for NLP requests in the Campaign Atlas portal.
Customizable parameters implemented within the constructor
: These included: translation to local languages, setting the tone of voice, and generating text for specific outreach channels.
In addition
: Fixing spelling and grammar, adjusting text length, and simplifying the language.
Support for all nine must-have languages
: English, Ukrainian, Kazakh, Moldovan (Romanian), Georgian, Belarusian, Arabic, French, and Russian.
Support for other languages including
: Uzbek, Turkmen, Tajik, Serbian, Montenegrin, Macedonian, Kyrgyz, Albanian, Azerbaijani, Armenian, and Bosnian.
Optimization of text generation for outreach channels
: Social media (Facebook, Instagram, LinkedIn, Twitter), Messengers (Viber, Telegram), email, and scripts for sales representatives.
About us: Neurons Lab
Neurons Lab is an AI consultancy that provides end-to-end services – from identifying high-impact AI applications to integrating and scaling the technology. We empower companies to capitalize on AI’s capabilities.
As an AWS Advanced Partner, our global team comprises data scientists, subject matter experts, and cloud specialists supported by an extensive talent pool of 500 experts. We solve the most complex AI challenges, mobilizing and delivering with outstanding speed to support urgent priorities and strategic long-term needs.
Ready to leverage AI for your business? Get in touch with the team
here
.
Get in touch
Leverage the power of AI for your business.
Related Stories
Peak Defence:
Automating Business Operations with Generative AI in Cybersecurity
Cybersecurity
Explore story
Penpot:
Elevating the world’s first open-source design platform with AI
Design
Explore story
Nection:
Leveraging AI-powered recommendations for personalized engagement
Customer Relationship Management
Explore story

