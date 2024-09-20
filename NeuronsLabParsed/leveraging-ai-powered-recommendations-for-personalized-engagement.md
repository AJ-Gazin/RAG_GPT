## Section 1

CLIENT: NECTION
INDUSTRY: CUSTOMER RELATIONSHIP MANAGEMENT
Leveraging AI-powered recommendations for personalized engagement


## Section 2

jump to
Project Overview
Business Challenges
Solution Space
Results
Project Overview
Nection
, a company innovating in the cutting edge of professional relationship management, aimed to transform how individuals build and manage their networks. They developed NexGen, an innovative personal customer relationship management (CRM) application for smartphones.
In the era of information overload, individuals often struggle to maintain regular, meaningful interactions with their diverse professional contacts. Nection partnered with Neurons Lab to build NexGen, the generative AI module of their personal CRM,
leveraging artificial intelligence to provide users with intelligent recommendations for smarter engagement
and interactions with contacts. Nection aimed to
improve relationship management, drive user engagement, and ultimately increase the value proposition
of their CRM platform.
Business Challenges
Nection sought to enhance user engagement by sending personalized messages, including birthday wishes, congratulations, and event reminders. However, their main challenge was delivering personalized messages to users due to limited data availability and sparse contact information. The existing data sources provided insufficient information, making it difficult to create personalized and engaging content for users.
Solution Space
To address the challenge, Nection worked with Neurons Lab to develop a solution leveraging AWS services. The solution focused on a notification-centric approach, involving AWS Lambda, MongoDB, and the specialized large language model, Antropic Claude through Amazon Bedrock.
The recommendation system uses the following flow to bring smarter, personalized recommendations to users:
1.  Action Recommendations Generation:
Periodically, action recommendations, such as birthday wishes or event reminders, are generated for each user and stored in the database.
2. Batch Execution by NexGen:
NexGen runs periodically (e.g., every 4 hours) to:
Load new action recommendations for users with pending notifications, based on their notification cadence.
Load user types to match users with specific notification configurations.
For each user:
Convert action recommendations into a format suitable for the Antropic Claude language model input.
Create a prompt template.
Run Antropic Claude to generate personalized notifications.
Apply output validation.
Store the generated notifications in the database, including timestamps and configuration IDs.
3. Nection System Propagation:
The Nection system runs periodically to fetch pending notifications and propagate them to users. Nection defines the implementation details of this step.
Results
The collaboration between Neurons Lab and Nection resulted in a successful solution for delivering personalized messages through NexGen.
Key benefits and outcomes include:
1. Personalized Messaging Strategy:
The implemented solution allowed NexGen to generate personalized messages for users, significantly enhancing user engagement and satisfaction.
2. Flexible and Efficient Execution:
The system’s periodic execution ensures timely and efficient generation of notifications, considering each user’s notification cadence and preferences.
3. Seamless Integration with Nection System:
Neurons Lab worked closely with Nection’s technical lead to ensure a smooth integration with the Nection system. Documentation provided facilitates ongoing maintenance and development.
4. Comprehensive User Activity Handling:
Nection’s handling of user activities, including push and open timestamps, provides valuable insights into user engagement, enabling data-driven decision-making.
Ultimately, the collaboration between Neurons Lab and Nection resulted in an effective solution for delivering personalized messages through NexGen. The implemented system not only addressed the challenge of limited data but also provided a foundation for future enhancements, allowing NexGen to deliver a unique and engaging messaging experience to its users.
About Neurons Lab
Neurons Lab is an AI consultancy that provides end-to-end services – from identifying high-impact AI applications to integrating and scaling the technology. We empower companies to capitalize on AI’s capabilities.
As an AWS Advanced Partner, our global team comprises Data Scientists, Subject Matter Experts, Cloud Specialists supported by an extensive talent pool of 500 experts. This expertise allows us to solve the most complex AI challenges, mobilizing and delivering with outstanding speed and supporting both urgent priorities and strategic long-term needs.
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
Treatline:
Implementing Intelligent Document Processing to streamline the Prior Authorization Process in Healthcare
HealthTech, InsurTech
Explore story
iPlena:
Transforming user experience in physiotherapy
HealthTech
Explore story

