## Section 1

Client: Creative Practice Solutions
Industry: HealthTech
Region: USA
Developing an AI-Driven Medical Transcription & Billing System


## Section 2

jump to
Partner Overview
Challenge
Solution
Results
About Neurons Lab
Creative Practice Solutions (CPS) partnered with Neurons Lab to bring transformative digital solutions to the healthcare industry in order to improve operational efficiency.
Partner Overview
CPS is a healthcare business consulting group with projects across the USA. With 30 years of industry experience, the team is eager to help improve the functionality and efficiency of both private and public medical organizations through newly adopted digital solutions.
The company works with billing protocol development, billing services, personalized consulting, and audits.Their main mission is to reduce “screen time” in healthcare, lowering the administrative burden on industry professionals today.
Challenge
Creating a technology to automatically populate a medical note and select the relevant procedure and diagnosis code.
Many of the complex challenges the medical sector faces today can be resolved via the application of innovative technologies like artificial intelligence and machine learning. These challenges include:
Medical documentation and coding are performed across a wide variety of office settings making a universal solution difficult
Current EMR requirements contribute to physician burnout
Limited budget and time resources
Lack of historical data
Non-technical target audience
Concerns about maintaining HIPAA compliance
CPS were focused on developing a solution that would reduce administrative burden, since medical providers spend a lot of time on tasks such as preparing notes and assigning billing codes for invoices.
To tackle this, CPS partnered with Neurons Lab to work on
SmartNoteMD
. A technology to effectively organize medical note taking and speed up the reimbursement process.
“Despite the many features of current EMR systems, we consistently hear how medical providers are 30-60 notes behind schedule and have to spend their weekends catching up. As a solution, we decided to create an application using AI, machine learning and listening technology to automatically create medical documentation with suggested billing codes in real time. This allows providers to cut down on labor, while increasing revenue through fully supported medical notes”.
Matt Dallmann, President @Creative Practice Solutions
A reasonably large system had been developed before Neurons Lab started working on the project. The application and the infrastructure as a whole were ready and working but needed some substantial improvements. CPS counted on Neurons Lab to introduce a new approach to the work, which would simplify the system’s architecture throughout further development.
The initial client’s requirements were to continue the backlog of the previous progress as well as:
Performance optimization and improvement
Development of an OpenAPI for integration with clinics
Improvement of information processing and interaction with ML components
Creation of documentation; since the client is not “technical,” it was necessary to describe the documentation for the API, etc.
Solution
An AI/ML solution for the creation of medical notes and billing codes in real-time.
The SmartNoteMD project aims to simplify the process and save time on documentation for healthcare providers and practitioners. During the patient examination, the visit is recorded with subsequent automatic processing of the text and the extraction of information about the patient from the text, such as:
Meta information like height, weight, name, date of birth, etc.
Diagnoses, allergies, and individual symptoms
CPT codes (
Current Procedural Terminology
) and billing codesA key point detector AI/ML model was built, trained, and deployed using Amazon SageMaker to identify key points in the human body in photos and videos. An anomaly diagnostic AI/ML model was also built, trained, and deployed using Amazon SageMaker to detect anomalies in human posture.
With the help of AI/ML in line with
AWS assistance
, the program automatically creates a medical note with billing codes in real-time. In addition, the software uses iOS technology to convert handwriting to text and transfer other medical records directly into application fields via the device camera. Like the popular “Amazon Alexa,” certain keywords and phrases allow SmartNoteMD to automatically populate a medical note and select the relevant procedure and diagnosis codes.
Within the app settings, the professional has the option to select a specialty to improve app accuracy, and the software will use only the applicable codes per treatment type chosen. For further accuracy, the professional can also create a custom set of diagnosis and procedure codes from the library.
The app also has a hands-free function, where the software starts recording once the conversation begins and finishes after a few seconds of silence.
The solution was designed to meet the following requirements:
Create a HIPPA- compliant solution with an enabled audio recording that allows healthcare providers and practitioners to take notes with structured entities
Attain a high level of accuracy in recognizing specific medical terms in speech and text
Complete the solution’s launch on 4 operating systems (iOS, iPadOS, macOS) without assembling 4 distinct development teams
Allow secure medical note exportation in different formats to match with existing billing systems
Ability to collect medical data securely and store it for further data science operations
Guarantee that there will not be extra system administration expenditures due to automated serverless infrastructure
Help patients and doctors meet one-on-one and remotely as if they were in-person
Provide the modularity of the solution where any new features will not affect the existing ones
In addition, an
OpenAPI
is being created for integration with clinics to make it possible to automatically enter the information received into the clinic’s EHR system.
Click
here
to see the software in action.
Working with AWS technologies
As an
AWS partner
of Advanced Tier, Neurons Lab created a complete serverless solution with a microservices architecture based on AWS managed services.
The current system uses different components, namely:
The API layer is powered by
Amazon API Gateway
The backend system is based on
AWS Lambda
functions
Data layer is based on
Amazon Aurora Serverless
and
Amazon S3
Amazon API Gateway is integrated with
Amazon Cognito
, which handles user management and authentication
ML components for medical note creation are implemented with
Amazon Transcribe
,
Amazon Transcribe Medical
,
Amazon Translate
In addition, Amazon API Gateway is secured with
AWS WAF
, which is a web application firewall that helps protect your web applications or APIs against common web exploits and bots that may affect availability.
Results
At this moment, the SmartNoteMD system is being tested among its users in real-life cases, namely independent practitioners, nursing services, and other medical organizations. CPS is focused on increasing accuracy for medical notes in line with reducing the time spent to produce the results.
Once the solution is released, it will allow hospitals and practitioners to see more patients whilst reducing administrative burden. This has the potential to reduce burn amongst staff whilst also increasing revenue. The solution can also be positioned as a powerful recruitment initiative for specific organizations to attract top physicians and nurse practitioners.
Finally, timely and accurate medical note submission to the insurance providers will also increase medical billing efficiency. Accuracy in this matter is essential, especially in the USA. Incorrectly filled insurance claims may result in revenue loss, huge penalties, and lawsuits.
About Neurons Lab
Neurons Lab is an AI service provider that partners with fast-growing companies to co-create disruptive AI solutions, empowering them to gain a competitive edge in their industries.
Our goal is to help businesses to unlock the full potential of AI technologies with support from our diverse and highly skilled team, made up of applied scientists and PhDs, industry experts, data scientists, AI developers, cloud specialists, user design experts and business strategists with international expertise from across a variety of industries.
Get in touch
Ready to partner
with us?
Making empty promises is not our style, but sharing cases of in-depth feasibility analysis for businesses is. Here are some of them
All  stories
Creative Practice Solutions :
Developing an AI-Driven Medical Transcription & Billing System
HealthTech
Explore story

