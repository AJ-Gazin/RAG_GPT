## Section 1

Deep-dive into Peak Defence’s AI Transformation #2 with Roman Jasins and Artem Kobrin
Generative AI
AI Delivery
AWS
Apr 12, 2024
|
10 minutes


## Section 2

SHARE THE ARTICLE
In this interview, our Head of Cloud & Partner, Artem Kobrin, talks to Roman Jasins, Co-Founder of
PeakDefence
, about their journey in
automating cybersecurity processes with AI
.
In the
previous session
, we discussed the business impact of AI on security and audit. Now, Artem and Roman discuss the specifics of the solution implementation process and the technologies used.
Since its inception a couple of decades ago, PeakDefence has been focused on helping companies protect their assets and manage risks. Now, they are an AI-powered security guardian angel, securing clients assets.
While PeakDefence has consistently strived to scale the security expertise they have cultivated over the years, scaling anything that necessitates extensive knowledge and human resources was a formidable challenge till recently.
How did PeakDefence start its AI journey?
Security has historically been viewed as a disabler, something that closes doors and opportunities, depending heavily on a few skilled, experienced individuals who do a lot of legwork.
But PeakDefence looks at it differently. They were determined to figure out how to achieve an optimal level of security with minimal investment of time and resources and better serve their clients, both start-ups and bigger enterprises.
When AI started to gain traction in the last couple of years, PeakDefence thought, “Why don’t we take AI and see if we can build a prototype where the machines would do all the heavy lifting in terms of context understanding?”
Then, businesses could evaluate the most important information and make informed decisions about what they need to protect. This was when they reached out to Neurons Lab to bring in niche AI expertise to work on the prototype.
What were the main considerations when choosing what model to use?
For the PeakDefence AI platform, we used
Antropic
’s Claude model (first Claude2, then the newly-released Claude3) to experiment and save time otherwise needed to gather data and train the model. When selecting the model, there were four key criteria that impacted the decision:
Security
: PeakDefence’s context is highly sensitive, so we couldn’t just share the data with a commercial model to train it.
Performance and speed
: The model had to analyze the data, give results back quickly, and do so accurately and on point.
Flexibility
: We needed to have the option to add more models that are better at certain tasks in the future because everything is evolving so quickly in this space.
Price of the solution
: Hosting our own model would have been quite expensive at the time.
What were the main challenges with the initial data set?
While PeakDefebce had a couple of decades of experience in the security space, its data was not cataloged in a useful way. Another challenge was the usability of the data due to privacy issues.
To overcome this cold-start problem, we had to hand-pick data and run numerous tests to see how the model behaves and how it performs. This is an ongoing process.
Now, we have a machine-human hybrid, where AI does a lot of the heavy lifting for the PeakDefence team, and professionals review it, both to deliver quality and to help the model learn.
What were the main benefits of implementing a feedback loop?
For the PeakDefence AI platform, the feedback loop and reinforced learning were extremely important. That’s how we get better results for each particular user and also continuously improve the product. The feedback loop allows the model to generate more personalized results and achieve higher relevancy.
How did we tackle the challenge of using multilingual documentation?
Claude2
supports hundreds of languages out of the box, so that’s why we didn’t need to do any language-specific model training. We also used
Amazon Bedrock
to transform the documentation information into a digital vector space, which is also multilingual. For example, even if a question was asked in English, the model would generate an answer in the original language from the context.
Why was it important to have a “fail fast” approach for this project?
With any new product, any new problem you’re trying to solve, and especially with AI, there is a lot of experimentation needed because everything is still new and is changing rapidly.
There is no guarantee; all the experiments might fail. So, if you don’t have the right team that is ready to experiment and pivot based on the results, discarding whatever didn’t work quickly, you don’t stand a chance.
For this project, we used a well-tested combination of methodologies, moving in two-week intervals, sprints. The idea was to see if it could be done and learn from that. That’s exactly what helped us in this partnership with PeakDefence and allowed us to create a solution that already sees some traction with customers and an increasing demand after only 3 months into the development.
How did the current setup allow us to start using the platform this quickly?
Firstly, creating a quick prototype to check the feasibility of the idea helped us get a head start. Doing small experiments every day also allowed us to move quickly and find the right setup.
The next step was to create a product that would bring real value. For this, we needed to do some data processing, find the right model, find the database and infrastructure that can handle all of this.
We also needed to address the security, data privacy, and scalability issues. That’s why we made this platform serverless, with most of the heavy lifting done by
AWS
, which hosts instances and provides serverless services that can be easily integrated through the API.
Here is what the solution looks like now:
We have a
Streamlit
user interface (UI) that is client-facing. Here, you can submit your documentation, trigger the audit process, and download the report.
Under the UI, we have an API that is connected to
Chroma DB
, a vector store. We convert the documentation to vectors and store them there, and we search through this database.
AWS
StepFunction
and
Lambda functions
are used to do the actual processing. The StepFunction can parallelize the process, triggering hundreds of Lambda functions to answer each question of the audit in parallel. This allows us to scale almost without any cost, processing hundreds of questions within five minutes. This is a significant improvement from doing it in sequence, which may take up to a few hours to process and generate a report.
The next step would be to simplify this architecture within
LangGraph
further, putting decision-making and separation on the graph. That’s something we are currently experimenting with.
How did this setup help us achieve product goals?
Having many Lambda functions triggered in parallel speeds up the work of information security professionals and allows us to visualize what is going on in a particular business and what assets need to be protected.
The major benefit of an AI-powered approach is the ability to process a lot of data and get meaningful results with actionable recommendations without having to wait for hours or days. This is one of the things that PeakDefence clients value the most.
The combination of clear visibility and rapid analysis is a powerful capability that PeakDefense provides. It allows information security teams to be much more agile and responsive, which is crucial in today’s fast-paced threat landscape.
We are excited about continuing to build on this foundation and exploring new techniques like LangGraph to enhance the platform further, pushing the boundaries of what’s possible with AI in this domain.
About Neurons Lab
Neurons Lab is an AI consultancy that provides end-to-end services – from identifying high-impact AI applications to integrating and scaling the technology. We empower companies to capitalize on AI’s capabilities.
As an AWS Advanced Partner, our global team comprises data scientists, subject matter experts, and cloud specialists supported by an extensive talent pool of 500 experts. We solve the most complex AI challenges, mobilizing and delivering with outstanding speed to support urgent priorities and strategic long-term needs.
Ready to leverage AI for your business? Get in touch with the team
here
.
Get in touch
Leverage the power of AI for your business
SHARE THE ARTICLE
All Stories
Peak Defence:
Automating Business Operations with Generative AI in Cybersecurity
Cybersecurity
Explore story
Treatline:
Implementing Intelligent Document Processing to streamline the Prior Authorization Process in Healthcare
HealthTech, InsurTech
Explore story
Featured Articles
Deep-dive into Peak Defence’s AI Transformation #1 with Roman Jasins and Lilia Udovychenko
Mar 27, 2024
|
3 minutes
Generative AI
Intelligent Document Processing
7 Tips for Effective Risk Management in AI Delivery
Oct 19, 2023
|
7 mins
AI Delivery
Explore The Managed Capacity Model for successful AI Solution Development
Nov 6, 2023
AI Delivery
Neurons Lab Achieves the AWS Generative AI Competency
Mar 5, 2024
Generative AI
AWS
Top 5 Gen AI use cases for enterprises to consider in 2024
Feb 23, 2024
|
9 minutes
Generative AI
Unlocking Generative AI for Enterprises with Alex Honchar: Top Opportunities and Threats
Feb 14, 2024
|
8 min
AI Product Development
Generative AI
Unlocking IDP #1: A Guide to Intelligent Document Processing for the Modern Enterprise
Dec 11, 2023
|
5 mins
Generative AI
Intelligent Document Processing
Unlocking IDP #2: Maximizing the ROI of intelligent document processing for enterprises
Jan 26, 2024
|
7 mins
Generative AI
Intelligent Document Processing

