## Section 1

Predictive maintenance for smart warranty service of PV Inverters
Predictive maintenance
Jul 26, 2023
|
6 min read


## Section 2

SHARE THE ARTICLE
We’re living in a world with a clear demand for clean, zero-emissions, and sustainable energy with s
o
lar energy technologies as one of the most heavily funded renewables sources (Statista research). In order to keep this amazing growth, green energy company owners have to think not only about cultural movement but also about effectiveness and competition, which makes any market healthy and long-term oriented.
We already have seen in recent history moments when cleantech couldn’t prove itself, actually forming a bubble, where
startups couldn’t back up their traction with working technology
and actual business results (The Green Bubble), and our job is to
fix this using AI technology
as one of the means.
You can also read this article republished on our
Medium
.
The inverter is all you need
A solar inverter is one of the most essential elements of a solar electric power plant. Often considered as the brain of the system, apart from basic functions, solar inverters help in maximizing electricity production as well as ensuring compliance with regulations, optimizing power, managing temperature, and controlling and monitoring all parameters, yields, and operational data of the power plant (Imarc research). The solar inverter is the critical and most complex part of any solar system and unfortunately, it’s the part that is the
most prone to having issues
. The report from Raptormaps that analyzed PV systems inspected in 2020, encompassing 70,121,507 modules across 1,126 PV systems, with a total nominal capacity of 22,032,460 kW showed that:
String faults
affected 173,002 kW, which is 42.4%
Combiner faults
affected 58,247 kW, which is 14.3%
Tracker faults
affected 49,065 kW, which is 12.0%
This is not surprising taking into account that inverters are usually located outside in heavy weather conditions including rain, humidity, and extreme heat, all while generating thousands of watts of power for up to 10 hours a day. This is why it’s important to use a quality inverter and mount it in a protected location if possible.
The 10-year warranty curse
Today,
reactive and preventive maintenance
approaches are the main methods to identify and fix inverter faults. The assumption that the components will not under-performer fail until the scheduled visit, leads to a significant loss of production and revenue. To deal with the disadvantages of current maintenance methods, the solar industry is very keen on understanding the possibility of
early detection of inverter faults
by the implementation of
predictive maintenance
.
Most of the modern brands now offer 10-year warranties but some will only repair a faulty unit if it is shipped to them (at the owner’s costs) and returned, which can take weeks and sometimes months. This will have a much greater financial impact due to system downtime, especially in the central or off-grid inverter. Such events cause
significant inconvenience for customers and can lead to reputational losses on the market for the manufacturer
.
But technologies like machine learning can lead to
more proactive warranty management
. Predictions from the data can determine that one component in a product is prone to overheating and causing outages. Manufacturing and supplier data allow the companies to identify the problem and replace the defective component before the product is shipped to the client.
The inverter has all the data
Illustration from
https://ieeexplore.ieee.org/document/9272625
Inverter, as the key equipment to collect core data and control power of PV plants. Most solar inverters come with a built-in Wi-Fi or PLC interface and cloud-based monitoring platforms, but the capabilities of built-in monitoring are still poor or require too many manual steps to perform accurate analysis. Also,
not all of the data measured by the inverter is sent to the cloud
. An extended list of measured parameters in the monitoring system can help to develop an advanced ML model that can predict catastrophic failures before they occur.
Where does AI step in?
Communications, ground faults, heat management systems, and insulated gate bipolar transistors emerge as the most frequently discussed inverter subsystems. Further evaluation of these failure modes identified distinct variations in failure frequencies over time and across inverter types, with communication failures occurring more frequently in early years. Increased understanding of these failure patterns can inform ongoing PV system reliability activities, including simulation analyses, spare parts inventory management, cost estimates for operations and maintenance, and the development of standards for inverter testing. Advanced implementations of machine learning techniques coupled with standardization of asset labels and descriptions can extend these insights into actionable information that can support the development of algorithms for predictive maintenance, which will further
reduce failures and associated energy losses
at PV sites.
Screenshot from this webinar.
For example, let’s consider how an AI-based model can help
predict the IGBT fault before that happens
. IGBT is a crucial component of the inverter. On the other hand, the investigation and the study of the IGBT component show several changes within its behavior and lifetime, while this component is highly influenced by the operating conditions. Using the approach based on combining both principal component analysis (PCA) technique and the feedforward neural network (FFNN) technique can be achieved early detection of changes in the IGBT behavior. PCA is used to reduce features extracted from IGBTs and the FFNN is implemented to achieve online regression of the trend parameter obtained from the PCA technique.
Illustration from
https://www.mdpi.com/2079-9292/9/10/1571/pdf
Similar models can be applied to each inverter’s sub-systems and accurately predict the remaining useful life. Such an approach helps to generate smart alerts for maintenance and activate warranty without downtime.
Planning AI integration in the product
Before any feature development, there is a necessary step of discovery and planning. For you, as a renewables industry expert, there will be no problem analyzing the market, competitors, and what features they already have. Also, it is relatively straightforward to define inner inefficiencies and develop metrics that will represent the growth of the performance (i.e. production output, number of faults, etc). Now we need to design a technological side of the product and this is a place, where standard recommendations about AI from the digital world might be delusional, if not completely wrong.
Data sourcing done right
Typically only 2–3 parameters are monitored to the cloud, you should start to collect more as soon as possible. Usually, inverters send to the cloud several parameters like DC/AC Voltage, Current, and Power. But writing parameters for each inverter subsystem
like heat system, IGBT and etc. would help to develop more accurate predictive maintenance models
.
Pay attention to the post-mortem data after previous breakouts — it will help in predictions and root-cause analysis.
Inverters alarm logs are a good source of labeled data
. But unfortunately, this data is not stored in the right way, and most commonly can’t be used for supervised ML learning
ML problem formulation done right
The field
lacks a lot of labeled data
compared to the traditional ML fields as computer vision and NLP, hence, classical pipelines most probably work
Important to know, that additional information about the industry and the domain augments the process of making more precise and accurate models based on the available features in the dataset and as a result, the model would then generalize better into real-world situations. On the other hand, in case of lack of data,
domain knowledge can allow us to use digital twin approaches
and physical-awareness models that help to achieve accurate results with limited amounts of data
Next steps
Taken the above-written, we can see that AI feature embedding into the product in the PV industry is rather different from the standard processes in the “digital” AI industry with tasks such as computer vision or natural language processing embedded in the websites or mobile applications. We need to understand the physical processes, properties of the data very well to design a product that will actually work for the end customers. We hope that the list of issues above will be a helpful starting point to discuss with your team. From our side, we will be happy to support you on this stage as well as with the implementation of the AI algorithms and their integration into the final product. You can reach us out at
[email protected]
or contact Andrew, author of this article and Managing Director & Partner, Industrial IoT Practice directly at
[email protected]
.
Contact the team
Drop us a line for
a consultation
SHARE THE ARTICLE
Making empty promises is not our style, but sharing cases of in-depth feasibility analysis for businesses is. Here are some of them
All stories
Magellan X:
Creating a predictive maintenance solution in shipping with Magellan X
Cleantech
Explore story
Creative Practice Solutions :
Developing an AI-Driven Medical Transcription & Billing System
HealthTech
Explore story
iPlena:
Transforming user experience in physiotherapy
HealthTech
Explore story
featured articles
AI for portfolio management: from Markowitz to Reinforcement Learning
Jul 16, 2023
|
11 min read
FinTech

