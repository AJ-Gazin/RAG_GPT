## Section 1

CLIENT: Treatline
INDUSTRY: HealthTech, InsurTech
Implementing Intelligent Document Processing to streamline the Prior Authorization Process in Healthcare


## Section 2

jump to
Partner Overview
Challenge
Solution
Results
Treatline partnered with Neurons lab to reduce the burden of administration on healthcare providers and insurance companies with a digital prior authorization platform and intelligent document processing.
Partner Overview
Prior authorization is an insurance review process required to approve medical coverage that puts significant strain on medical professionals resulting in care delays, negative clinical outcomes, and an average cost of $35 billion annually. Treatline, an innovative InsurTech platform, wanted to tackle that problem hands on.
The platform facilitates the exchange of prior authorization requests through automation and improved payer/provider communication. It also provides analytical tools to measure treatment, clinician, facility, and organizational level performance and comply with recent changes in legislation. Treatline’s mission is to reduce the burden of prior authorization on healthcare providers and insurance companies, ensuring that patients receive timely and appropriate care while streamlining the overall process.
The Treatline team is led by co-founder Natalie Dranchuk, who has extensive experience in health policy and management, healthcare consulting, AI product development, and financial consulting. With a strong background in both healthcare and technology, the Treatline team is uniquely positioned to address the challenges faced by medical staff, patients, and insurance companies alike.
By focusing on improving patient outcomes, reducing administrative burden, and streamlining the healthcare delivery process, Treatline is committed to making a significant impact on the healthcare and insurance industries. Their innovative platform bridges the gap between healthcare providers and insurance companies, promoting greater efficiency and collaboration to ensure the best possible outcomes for all parties involved.
Challenge
Combining complex technologies to solve inefficiencies
The time-consuming pre approval process required by some insurance companies leads to staff burnout and negative impacts on patient care. Physicians spend an average of 13 hours per week on prior authorization tasks, submitting around 40 requests per week. Additionally, 86% of physicians describe the burden associated with prior authorization as high or extremely high, leading to annual costs of $35 billion.
Patients also suffer, with 93% of physicians reporting care delays, and 34% stating that prior authorization has led to serious adverse events for their patients, such as hospitalization, life-threatening events, or even death.
Insurance companies are also affected by the inefficiencies in the prior authorization process. Delays in claim approvals and the complex, manual nature of the process contribute to increased operational costs and dissatisfied customers.
Treatline set out to solve these inefficiencies by creating a streamlined solution that would benefit all parties involved.
Solution
Automation and AI-Powered Platform for Insurance and Healthcare
Treatline’s platform, developed in partnership with Neurons Lab, leverages cutting-edge AI technology and AWS services to streamline the prior authorization process for insurance companies and healthcare providers. The platform consists of three main components: the web app, Intelligent Document Processing (IDP), and the Generative AI Criteria Matching System.
Web App: User Interface and Secure Data Storage for Insurance and Healthcare Professionals
The web app provides insurance and healthcare providers with a user-friendly interface, hosted on Amazon S3 and served by Amazon CloudFront for fast and reliable access. User authentication is managed by Amazon Cognito, while the backend utilizes Amazon API Gateway, AWS Lambda, and Amazon DynamoDB for secure data storage and management.
Intelligent Document Processing (IDP): Advanced Data Extraction and Analysis for Insurance Claims
IDP is a key component of the Treatline platform, responsible for extracting, collecting, and interpreting data from medical records and documents, as well as insurance claim forms. By utilizing advanced techniques, such as natural language processing (NLP), optical character recognition (OCR), and computer vision, IDP ensures accurate and efficient data processing for both insurance and healthcare professionals. This significantly reduces the time spent on administrative tasks, allowing them to focus on patient care and claim processing.
Advanced Text and Structural Data Extraction
IDP leverages Amazon Textract to extract both text and structural information from medical and insurance documents, such as tables and key-value pairs. This powerful OCR technology enables the platform to recognize and process complex documents with a high degree of accuracy, ensuring that all relevant information is captured and utilized in the prior authorization process.
In-depth Document Analysis with Amazon Comprehend and Amazon Comprehend Medical
Once the text and structural data have been extracted, IDP utilizes Amazon Comprehend and Amazon Comprehend Medical to further analyze the content of medical and insurance documents. These AI-powered services identify critical information, such as entities, medical terms, and relationships, allowing the platform to gain a deep understanding of each patient’s medical history, insurance coverage, and requirements.
Intelligent Search with Amazon Kendra and Amazon CloudSearch
To enable quick and efficient retrieval of the extracted information, IDP uses Amazon Kendra and Amazon CloudSearch to create smart search indexes. Amazon Kendra is a machine learning-based enterprise search service, while Amazon CloudSearch is a fully managed search service. Both services allow users to search and retrieve relevant medical and insurance data effortlessly, further enhancing the platform’s efficiency.
The extracted information is then indexed and stored in an Amazon S3 bucket, with metadata stored in an Amazon DynamoDB database. This enables efficient retrieval and analysis of the data when generating prior authorization requests and processing insurance claims.
Scalable and Efficient Document Processing
To handle high volumes of documents and ensure that the platform remains responsive, IDP employs an asynchronous processing approach. This is managed through Amazon Simple Notification Service (Amazon SNS) and Amazon Simple Queue Service (Amazon SQS), which control the flow of documents awaiting processing. This architecture ensures that the platform can scale to meet the demands of healthcare and insurance providers, regardless of their size or the number of prior authorization requests and claims they submit.
Seamless Integration with Treatline Platform Components
IDP is fully integrated with the other components of the Treatline platform, such as the web app and the Generative AI Criteria Matching System. This seamless integration allows healthcare and insurance providers to benefit from the power of AI and advanced data processing techniques, streamlining the prior authorization process and improving efficiency across the board.
By combining the capabilities of IDP with the other components of the Treatline platform, healthcare and insurance providers can significantly reduce the time and effort required for prior authorization and claim processing, leading to better patient outcomes and more efficient healthcare and insurance delivery.
Generative AI Criteria Matching System: Enhanced Decision-Making
The Generative AI Criteria Matching System is a crucial part of the Treatline platform, designed to match medical summaries with both medical and insurance criteria. It ensures that prior authorization requests and insurance claims are tailored to meet the specific requirements of each payer, improving the likelihood of approval and reducing the need for time-consuming peer-to-peer reviews.
The system leverages the power of state-of-the-art generative AI models that excel in understanding and generating human-like text. By integrating these models into the platform, the Criteria Matching System can efficiently process medical summaries and match them with the most relevant criteria, ensuring that each prior authorization request and insurance claim is comprehensive and meets the necessary requirements.
The Generative AI Criteria Matching System is designed to match medical summaries with various types of criteria, such as allergies, diagnosis, measurement results, substance use, and treatment procedures. It can accurately identify the most important information in each medical summary and align it with the appropriate criteria.
Through the matching system, healthcare and insurance providers can enjoy a more streamlined and effective prior authorization and claim processing experience. Its ability to match medical summaries with the most relevant criteria ensures that requests are tailored to meet payer requirements, resulting in faster approvals, reduced administrative burden, and better patient care.
Results
The implementation of the Treatline platform has led to significant improvements in the prior authorization process for healthcare providers. These benefits are reflected in several key metrics, showcasing the platform’s effectiveness and efficiency:
Benefits for the Provider
Cost Savings and Additional Revenue.
The Treatline platform reduces the administrative burden and streamlines the prior authorization process, leading to cost savings and the potential for additional revenue.
Improved Employee Well-being.
By automating routine tasks and reducing administrative workload, the platform helps prevent burnout among medical staff, promoting a healthier work environment.
Faster A/R Turnover.
The efficient prior authorization process enabled by Treatline results in a quicker A/R payback period, improving the overall revenue cycle for healthcare providers.
Estimated Savings and Revenue Increase for the First Year
30% fewer peer-to-peer reviews.
The Treatline platform’s intelligent criteria matching system significantly reduces the need for time-consuming peer-to-peer reviews, contributing to a more efficient prior authorization process.
70% less time spent on administration.
The automation and AI capabilities of the Treatline platform minimize the time medical staff spend on administrative tasks, freeing them up to focus on patient care.
Significant savings and profit increase.
The implementation of the Treatline platform has the potential to result in substantial annual savings and a considerable increase in profit for healthcare providers.
By adopting the Treatline platform, healthcare providers can transform their prior authorization process, leading to improved efficiency, cost savings, and better patient outcomes. These results highlight the value of incorporating advanced automation and AI technologies into healthcare administration.
About Neurons Lab
Neurons Lab is an AI service provider that partners with fast-growing companies to co-create disruptive AI solutions, empowering them to gain a competitive edge in their industries.
Our goal is to help businesses to unlock the full potential of AI technologies with support from our diverse and highly skilled team, made up of applied scientists and PhDs, industry experts, data scientists, AI developers, cloud specialists, user design experts and business strategists with international expertise from across a variety of industries.
Let’s connect. Reach out to our expert team to discuss your next disruptive project.
Get in touch
Ready to partner with us?

