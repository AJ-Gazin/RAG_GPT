## Section 1

CLIENT: XAUEN
INDUSTRY: CYBERSECURITY
Building an innovative voice and text-based cybersecurity chatbot for Xauen


## Section 2

Partner Overview
Xauen
is a spinoff of the University of Jaén in Andalucia, Spain. It was established in 2021 and formed by young engineers with experience in
cybersecurity and cutting-edge technology
. They are
experts in digital transformation
and
support the launch of new tech-first companies
such as FinTechs, MedTechs, EdTechs, and more.
Among its services, Xauen offers a
virtual CISO
(Chief Information Security Officer) evolving platform called
Laguun
. As part of this, users fill in a questionnaire about their company’s security practices, with around
80 questions
. After answering these questions, the system prepares a summary for the user, showing the strong and weak points.
Business Challenges
This project vision emerged from the challenges of
staying ahead in a rapidly evolving market
.
The initiative addresses the need for
innovation in security assessment methodologies
by modernizing an outdated questionnaire format and incorporating AI technology.
Xauen aimed to innovate how companies assess their security by leveraging AI for a conversational security assessment chatbot. The chatbot needed to offer an intuitive, human-like interaction experience
to help CISOs evaluate their company’s security practices
.
Project Overview
Xauen wanted a
conversational security assessment text and voice-based AI chatbot
. It needed to be capable of intelligently navigating through dozens of security questions,
engaging users in dynamic conversations tailored to their responses
. This approach enhances user experience and ensures thorough coverage of security aspects.
Neurons Lab developed an AI-powered conversational chatbot that
transformed Xauen’s cybersecurity questionnaire into an interactive text and voice-based assessment
. The chatbot provides a seamless and engaging user experience while ensuring thorough coverage of all security aspects.
Project objectives included:
Ensuring consistent logic
: The conversational AI module should follow the logic flawlessly without overlooking any aspect of the security questionnaire. It also needed to put the assessment back on track if the user asks additional questions not present in the questionnaire.
Supporting verbal conversations
: Feedback loops allow the system to evolve and refine its capabilities based on real-time user interactions and inputs.
Providing low latency
: A fast speed was essential for the user experience and to innovate compared to the traditional questionnaire format.
Another key objective was to
showcase the new solution at an AWS summit
. The chatbot offered a prime opportunity to demonstrate Xauen’s commitment to leveraging
advanced technology to address contemporary business challenges
.
Solution overview
The diagram above shows the main components of the Laguun questionnaire process and the parts that the Neurons Lab team augmented with
conversational generative AI
.
We used the following AWS architecture to achieve this:
The chatbot API
leveraged AWS services – including
Amazon Bedrock
and
AWS Fargate
.
The AI agent
, built with a
Large Language Model (LLM)
and
retrieval-augmented generation (RAG)
architecture with a vector database, consistently follows the dialogue flow to ask all required questions.
Dialog
consistency
and fact-checking guardrails
are in place for the LLM to ensure accurate responses.
Integration
with
Deepgram
and
Amazon Polly
enables voice-to-text and text-to-voice functionality, as well as low latency and high language accuracy.
The solution also required synchronization between
Amazon S3
storage buckets and a vector database.
This setup allows an administrator to add documents to or delete them from the knowledge base when needed.
Solution process
The questionnaire chatbot needed to both ask questions and also respond in turn to any questions from the user.
Users may want the chatbot to explain specific terminology or handle their follow-up queries, for example.
In the above diagram, the user starts a session, and the chatbot asks the first question from the questionnaire. Then, it processes the input, and there are
two options
for what happens next:
If it’s an answer to the question
: The chatbot processes the answer and asks another question. The cycle continues until it finishes asking all the questions and reaches the end of the session.
If it’s a question from the user
: For example, a query about terminology – it processes the question by using RAG.
Pre-implemented materials in the RAG cover key security topics. However, if after leveraging
RAG there is still not enough
context to answer the user’s question, more resources are required.
In this solution, the system uses a web search to find the answer. After answering the question, the cycle continues – processing the next input from the user until all the questions are complete.
Results
Developed by Neurons Lab during a tight timeframe of
only a few weeks
, the innovative chatbot was ready in time for an AWS Summit, where Xauen successfully showcased it.
Key metrics used to determine the success of the solution included:
Accuracy:
Definition
: The percentage of chatbot responses that accurately address user queries and provide relevant information based on the cybersecurity questionnaire
Measurement
: (Number of accurate chatbot responses) / (Total number of chatbot responses) x 100
Target
: Achieve and maintain a chatbot response accuracy of at least 90%
A high chatbot response accuracy is essential for ensuring users receive reliable and relevant information throughout the assessment process.
This metric directly influences user satisfaction, trust, and the chatbot’s overall effectiveness in conducting cybersecurity assessments.
By consistently providing accurate responses, Xauen can demonstrate the value of its AI-powered solution to potential clients and establish a competitive advantage in the market.
Latency:
Definition
: The chatbot’s end-to-end response time is measured from the moment a user submits a voice input until the chatbot provides a voice output response.
Measurement
: Average time taken for the chatbot to process voice input, generate a response, and deliver the voice output to the user, measured in seconds.
Target
: Maintain an average end-to-end latency of 30 seconds or less for 90% of voice input to output responses.
Low latency is crucial for providing a smooth and engaging user experience.
By minimizing the time users wait for the chatbot to respond, Xauen can ensure that the assessment process remains interactive and efficient.
Consistently meeting the latency target will help to maintain user satisfaction and encourage the adoption of the chatbot for cybersecurity assessments.
In another case study, find out how we
automated business operations with GenAI in cybersecurity for Peak Defence
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
Insurance client:
AI-driven automated medical insurance claim analysis to detect fraud
Insurance
Explore story
Peak Defence:
Automating Business Operations with Generative AI in Cybersecurity
Cybersecurity
Explore story
Solar Manager:
Scaling support operations for a growing customer base with an AI assistant chatbot for Solar Manager
CleanTech
Explore story

