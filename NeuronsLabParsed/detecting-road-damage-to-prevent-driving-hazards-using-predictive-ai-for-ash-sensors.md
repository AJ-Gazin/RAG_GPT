## Section 1

CLIENT: ASH SENSORS
INDUSTRY: CIVIL ENGINEERING
Detecting road damage to prevent driving hazards using predictive AI for ASH Sensors


## Section 2

Partner Overview
ASH Sensors
delivers
real-time remote structural health monitoring to reduce risk, minimize maintenance costs, and increase safety.
It provides state-of-the-art monitoring and maintenance solutions for infrastructure.
Based in California, the company installs sensors in structures to monitor for cracks, leaks, stress, and other types of damage. Its services include continuous intelligent structural monitoring, early detection of structural problems, and real-time preventive maintenance paired with
AI-enabled software
.
With this in mind, ASH Sensors aimed to deliver an
innovative proof of concept
(PoC) for an important
city project in San Jose
, Silicon Valley, to detect road damage before it becomes a hazard.
Challenges
In a
highly competitive bidding process
, ASH Sensors needed an innovative solution to stand out from other tenders.
The company needed a partner with the required level of AI expertise in-house to
deliver a working PoC quickly.
Among the required and desired features for project proposals were the
use of cameras and simple image recognition
, but these were not yet part of the existing ASH Sensors package choices.
Project Overview
Given city officials’ interest in exploring the potential for AI in this area, ASH Sensors was eager to collaborate on this initiative with Neurons Lab. The company was committed to exploring the possibility of
leveraging cameras, sensors, and AI to detect issues
for city services to address proactively.
ASH Sensors
accelerated development of its patented laser surface scanning technologies
including a vibration sensor, added a camera feature, and worked with Neurons Lab in a highly accelerated project.
In
under four weeks
– after an introduction from AWS – we discussed, defined, and collaboratively scoped out the project. We also identified
valuable developer benefits
that ASH Sensors was eligible for through AWS and secured all the required approvals.
The goal of the PoC was to ascertain the
condition of roads
, specifically to identify whether there is any damage that needs fixing.
Road condition classifications for the AI to detect included the following categories: Longitudinal cracks, transverse cracks, alligator cracks, white line blur, crosswalk blur, and manhole covers.
Neurons Lab
delivered the entire program in two sprints, each lasting only two weeks,
including the final presentation.
Solution
Neurons Lab developed a system that accurately predicts and identifies road damage. The solution offers precise timing and geographical mapping of issues, presented through an easy-to-use interface.
The diagram below provides a high-level overview of the
AWS architecture
that Neurons Lab used to deliver the PoC, including a real-time demo on
gradio
:
#1 Web interface
Gradio hosted on
Amazon Elastic Compute Cloud (EC2)
provided the platform.
We built the front-end demo application using this flexible interface that allows users to upload images and receive detection results via an API.
#2 NVIDIA Triton Interface Server
We used the versatile
NVIDIA Triton Interface Server
, which is deployable on
Amazon SageMaker
and EC2.
It supports multiple frameworks and can efficiently manage various GPUs and CPUs.
The server retrieves the latest model weights from the
Amazon S3
bucket for inference tasks.
#3 Training process
We trained the machine learning model using EC2 and
SageMaker Notebook Instances
.
Upon completion, we exported the trained model weights to the S3 bucket for storage.
#4 S3 bucket
This bucket stores the latest model weights, and the Triton Interface Server retrieves them, ensuring it uses the most recent one for inference.
Neurons Lab set up the server using a Docker container provided by NVIDIA and configured by
Ultralytics
. We tested the API endpoint for inference using a
YOLO v8
model for
image classification and object detection
.
The final
Road Damages detection demo
visualizes driving risks by uploading images and their metadata, processing them through an API, and
displaying the results on a map
.
The diagram above shows a high-level overview of the user interface components and their functionalities, with the user flow as follows:
#1 Upload ZIP file
The user uploads a ZIP file containing images and metadata.
The uploaded ZIP file contains road damage images and metadata with GPS data for each image.
#2 Road Damage images
The demo displays the processed images of the detected road damage.
Once the API processes the images, it visualizes and presents them in a gallery format.
#3 Damage Points on Map
It visualizes the GPS data on a map, showing the locations of detected road damage.
The metadata’s GPS data plots the damage points on a map, color-coded by the type of damage.
The user interface and mapping functions were modular, allowing ASH Sensors to map other data where it could be synced with GPS locations.
The company is currently leveraging this to add an
under vehicle scanning feature
with a
laser capable of finding damage or defects to a 1-2” resolution
. It offers
close to 100% detection
in the laser scanned surface, compared to the industry standard of 80-90%.
Results
Neurons Lab provided a solution delivering robust AI prediction and detection of road damage – timestamped, with its location on a map via an intuitive user interface.
Users can filter by road damage type using clickable checkboxes:
Alongside data processing, the innovative solution involved fine-tuning and deploying the model, as well as developing the dashboard.
Final deliverables included:
Deployed adapted object detection and classification model
for ASH Sensors’ data on AWS with the API
Streamlit demo application
showcasing road damage predictions and visualizations
Comprehensive documentation
for the API
Processed and stored sensory and image data
in the S3 bucket
In addition,
as a mid-project addition
, we also provided functionality to update the data and add new categories to the road damage model – or even create new models –
plus training
to use these features.
This gave the ASH Sensors team confidence to focus on their value-add infrastructure offering, while also assuring San Jose city officials they could deliver the line items that any other vendor could.
Feedback
“You’ve exceeded our expectations. The team is professional, very in sync, structured – it’s amazing. You are innovative and you helped us innovate. Truly remarkable… The final presentation was so thorough, we were able to invite key angel investors to watch the demo. This is truly next level service.”
Bob O’Donnell, Co-Founder & VP of AI & Development, ASH Sensors
About Neurons Lab
Neurons Lab is an AI consultancy that provides end-to-end services – from identifying high-impact AI applications to integrating and scaling the technology. We empower companies to capitalize on AI’s capabilities.
As an AWS Advanced Partner, our global team comprises data scientists, subject matter experts, and cloud specialists supported by an extensive talent pool of 500 experts. We solve the most complex AI challenges, mobilizing and delivering with outstanding speed to support urgent priorities and strategic long-term needs.
In another case study, find out how we
developed machine learning models to predict renewable energy output and consumption
for a CleanTech company.
Also, see how we
created a predictive maintenance solution for the shipping industry
with Magellan X.
Ready to leverage AI for your business? Get in touch with the team
here
.
Get in touch
Leverage the power of AI for your business.
Related Stories
Magellan X:
Creating a predictive maintenance solution in shipping with Magellan X
Cleantech
Explore story
CleanTech client:
Developing machine learning models to predict PV output and consumption
Cleantech
Explore story
Solar Manager:
Scaling support operations for a growing customer base with an AI assistant chatbot for Solar Manager
CleanTech
Explore story

