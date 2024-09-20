## Section 1

CLIENT: PEAK DEFENCE
INDUSTRY: CYBERSECURITY
Automating Business Operations with Generative AI in Cybersecurity


## Section 2

Partner Overview
Peak Defence
, which empowers businesses with professional cybersecurity management services, partnered with Neurons Lab to enhance their business operations using
generative AI technologies
.
Peak Defence provides
cutting-edge information security compliance consulting services
, helping clients meet standards and frameworks like
ISO 27001
, NIST, and SOC2. With increased demand, achieving compliance, managing information security systems, and responding to RFPs significantly drained employee time and resources, limiting the company’s scalability.
To evolve its offerings and unlock new capacity,
Neurons Lab worked with Peak Defence to implement an AI platform on Amazon Bedrock that leverages LLMs
(large language models) to automate manual compliance processes.
Early results show
the technology is already allowing Peak Defence to take on more customers without expanding its headcount.
Business Needs & Objectives
L
ike many fast-growing companies, Peak Defence sought to fulfill two fundamental business needs with their AI project:
market competitiveness and operational excellence.
Peak Defence wanted to remain competitive and innovative in the information security compliance market. By freeing up time spent on manual tasks such as compliance, information security management, and operations and responding to RFPs, they would be able to focus on doing what they do best: providing their clients with cutting-edge, leading solutions.
Secondly, Peak Defence aimed to achieve operational excellence by
minimizing the costs associated with the manual execution of compliance processes
. Automating manual tasks would allow the company to unlock new capacity and efficiency gains, enabling Peak Defence to scale and meet client needs faster.
Ultimately, targeting these two key needs would allow Peak Defence to remain a leader in their industry whilst also accelerating growth.
To fulfill their business needs, the following objectives were set for the project:
1.Streamline Compliance Audit:
Automate the evaluation process of suppliers’ compliance with standards like ISO 27001, ensuring quick and consistent analysis. Reduce audit document processing from 2 weeks to 3 days.
2.Enhance RFP Response Management:
Develop an AI-driven mechanism to analyze and respond to RFPs based on historical data and existing company policies. Reduce RFP response preparation from 2 weeks to 3 days.
3.Foster Continuous Improvement:
Implement feedback loops that enable the system to evolve and refine its capabilities based on environment and business changes, emerging threats and trends, real-time user interactions, and inputs.
Business Challenges
Data Limitations
Limitations on data samples was a key challenge faced at the start of the project. With access to only 20 or fewer examples initially, the AI model would likely face difficulties training for comprehensive accuracy across the full range of possible compliance scenarios.
Having more data samples would enable more robust training and improve the model’s ability to generalize to new and diverse cases. The small starting dataset posed a hurdle to developing an AI system that could automate audits with a high degree of precision.
Diverse Supplier Documentation
Another challenge stemmed from the wide variety of formats and structures found in suppliers’ documentation. Suppliers presented compliance data in different styles, making it difficult to standardize inputs for the AI system. This diversity in documentation could hinder the model’s ability to extract key information and evaluate adherence to standards in a consistent manner.
To automate audits effectively, the AI needed to be trained to handle diverse documentation structures and unstructured data formats.
Solution Space
To support Peak Defence in achieving automated and enhanced business processes, Neurons Lab implemented an AI platform on
Amazon Bedrock
that uses
Anthropic’s Claude
and other
LLMs
with two main features.
The first,
Audit AI SaaS Platform
, automates lengthy compliance processes like ISO 27001 compliance. The second,
RFP response preparation SaaS platform
, uses generative AI to provide answers to RFP requests based on company documentation and previously answered RFP requests.
The generative AI is built on a serverless architecture pattern that provides endless scalability and high-cost efficiency. The user interface is deployed with
AWS Fargate
, a fully managed serverless container service that allows the UI to scale up and down automatically based on demand. With the web UI, users can upload documents, view AI-generated analysis, and easily interact with the system.
Under the hood,
AWS Step Functions orchestrate the workflow and parallel processing
. It triggers
AWS Lambda
functions to execute the core business logic powered by the large language models. Thousands of questions can be processed concurrently by the Lambda functions to generate compliant audit reports, RFP responses, and other documentation in minutes.
This serverless architecture allows Peak Defence to leverage the power of Claude without provisioning their own AI infrastructure. The automation platform can expand and contract as needed to meet client demand.
Video Demo
For more information about our solution, check out the
AWS Partner Network (APN) blog – transforming cybersecurity audits with generative AI
.
Also, read our
deep-dive into Peak Defence’s AI transformation
.
Results
The AI automation platform delivered impressive early results for Peak Defence. Within 3 months, a minimum viable product was launched. Thanks to automated infrastructure provisioning through
GitHub Actions
and
AWS CDK
,
new tenants can now be deployed in as little as 30 minutes
.
Most importantly, the solution achieved the core goal of radically reducing the time required to produce deliverables like audit reports. For instance, the system can now generate compliance reports and responses to security RFPs
within minutes compared to the 2-3 weeks of manual effort previously required
.
Ultimately, compliance consultancy that once caused bottlenecks due to manual processes can now
scale efficiently to serve more customers
.
Feedback
“We needed a professional and flexible partner while going through a fundamental pivot. Neurons Labs helped us to add gen AI capabilities to an existing successful platform. This enabled Peak Defence to evolve and unlock new capabilities for its customers. Neurons Lab team brings experience, good organization and creativity to the projects. That combo makes it efficient and fun to work together. We recommend these guys without hesitation.”
Roman Jasins, Co-Founder & Board Member of Peak Defence
About Neurons Lab
Neurons Lab is an AI consultancy that provides end-to-end services – from identifying high-impact AI applications to integrating and scaling the technology. We empower companies to capitalize on AI’s capabilities.
As 1 of 18 certified AWS partners in Applied AI globally, our goal is to help businesses to unlock the full potential of AI technologies with support from our diverse and highly skilled team, made up of applied scientists and PhDs, industry experts, data scientists, AI developers, cloud specialists, user design experts and business strategists with international expertise from across a variety of industries.
Get in touch
with the team to discuss your next AI initiative.
Get in touch
Leverage the power of AI for your business.
Related stories
Treatline:
Implementing Intelligent Document Processing to streamline the Prior Authorization Process in Healthcare
HealthTech, InsurTech
Explore story
Humanity Inc.:
Enhancing Biomedical Datasets with Generative AI
HealthTech
Explore story
Posthumously:
Creating a Digital AI Twin to connect with loved one’s memories and work through the grief stages
Other
Explore story
Magellan X:
Creating a predictive maintenance solution in shipping with Magellan X
Cleantech
Explore story

