## Section 1

CLIENT: SOLAR MANAGER
INDUSTRY: CLEANTECH
Scaling support operations for a growing customer base with an AI assistant chatbot for Solar Manager


## Section 2

Partner Overview
Solar Manager
supports
households with solar power systems
to
use energy more efficiently and sustainably
. Its app, which runs on
Amazon Web Services (AWS)
, monitors how much energy a household consumes for everything from using small appliances to charging large electric vehicles. It is currently in more than 23,000 homes and the number is growing quickly!
The company
optimizes green energy usage for homes across Europe on AWS
, managing the distribution of energy based on algorithms and user preferences. Solar Manager uses machine learning to predict solar production and optimize energy usage.
Project Overview
Neurons Lab provided Solar Manager with a solution that leverages the
Anthropic Claude 3.5
large language model (LLM)
to increase customer support team productivity.
The objective was to implement a
chatbot assistant on AWS
infrastructure. This initiative aimed to streamline support ticket responses, improve efficiency, and maintain high customer satisfaction.
We connected Solar Manager’s data and knowledge base to the architecture, ensuring seamless integration and AI validation.
Our solution demonstrated the capabilities of generative AI and machine learning in optimizing customer support operations.
Solution
AWS architecture
Here is a visualization of the solution integrated with AWS services, including Neurons Lab GenAI accelerators:
Frontend layer:
This is the user interface with which the support agent interacts.
API layer:
This layer, powered by
AWS Fargate
running
FastAPI
, handles the API requests – under the hood it runs Langchain, working with the LLMs.
Data layer:
This layer contains the databases and storage services that house the data required for the AI to generate responses. It includes the
Amazon Aurora PostgreSQL
optimized for vector searches,
Amazon Neptune
for graph database queries, and
Amazon ElastiCache Redis
for caching data for quick access.
LLMs:
Claude 3.5 and Amazon Bedrock process the data to generate accurate responses in this layer.
Solution overview
The new process involves the following steps:
User:
Initiates the interaction by submitting a query.
Help Desk:
Passes the query to the support agent.
Customer support representative:
Inputs the ticket ID into the virtual AI assistant UI application.
Virtual AI assistant UI:
Forwards the ticket ID to the API layer.
API layer:
Processes the user question using a
Langchain
framework.
API:
Provides ticket-related information to Langchain.
Vector store/graph database:
Supplies the Langchain framework with company documentation, FAQs, and instruction data optimized for LLM and
retrieval augmented generation (RAG)
search and use.
LLM models:
Langchain generates a final answer using the information from the API and vector store, leveraging
Amazon Bedrock
with Claude 3.5.
Virtual AI assistant UI:
Receives the generated answer from the API layer and displays it to the support agent.
Customer support representative:
Answers the user quickly, supported by the additional information.
Results
The project delivered an advanced chatbot solution that included:
Knowledge graph integration:
We designed and implemented a schema for a
knowledge graph empowering RAG
. It provides knowledge from users, products, FAQs, and customer interactions – pulling relevant information for responses, ensuring users receive accurate and helpful information.
Multi-step state machine and LLM agent implementation:
The LLM effectively manages multi-turn interactions, providing the agent with coherent and contextually relevant conversations for a particular ticket and generating a proposed answer to the ticket.
Intuitive UI:
The customer support team can enter either a ticket or user ID and generate relevant answers, including relevant links for the documentation.
Incremental data updates:
A mechanism for regularly updating the knowledge graph with new data from all sources, keeping the AI’s knowledge base current.
Infrastructure as code deployment:
The application’s deployment via Infrastructure as Code enables cloud engineers to deploy the solution efficiently and securely, optimizing operational workflows.
Testing and accuracy metrics:
The AI undergoes comprehensive testing to ensure relevant and helpful responses, enhancing user satisfaction with the platform.
Results for the Solar Manager customer support team included:
Scaled support operations
to accommodate a growing customer base while still ensuring high customer satisfaction levels.
Facilitated knowledge transfer and training
for new support team members by leveraging AI-driven responses and insights.
Optimized the support team’s time allocation
, allowing staff to focus on more complex cases by automating responses to regular, routine inquiries.
In another case study, find out how we
developed machine learning models to predict renewable energy output and consumption
.
We have also used
predictive AI to prevent driving hazards and detect road damage for ASH Sensors
.
About Neurons Lab
Neurons Lab is an AI consultancy that provides end-to-end services, from identifying high-impact AI applications to integrating and scaling the technology. We empower companies to capitalize on AI’s capabilities.
As an AWS Advanced Partner, our global team comprises data scientists, subject matter experts, and cloud specialists supported by an extensive talent pool of 500 experts. We solve the most complex AI challenges, mobilizing and delivering with outstanding speed to support urgent priorities and strategic long-term needs.
Ready to leverage AI for your business? Get in touch with the team
here
.
Get in touch
Leverage the power of AI for your business.
Related Stories
CleanTech client:
Developing machine learning models to predict PV output and consumption
Cleantech
Explore story
ASH Sensors:
Detecting road damage to prevent driving hazards using predictive AI for ASH Sensors
Civil Engineering
Explore story
Magellan X:
Creating a predictive maintenance solution in shipping with Magellan X
Cleantech
Explore story

