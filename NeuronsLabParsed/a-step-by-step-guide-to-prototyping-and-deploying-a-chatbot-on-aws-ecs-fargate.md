## Section 1

A Step-by-Step Guide to Prototyping and Deploying a Chatbot on AWS ECS Fargate
AWS
Nov 17, 2023
|
5 mins


## Section 2

SHARE THE ARTICLE
Written by Artem Korbin, Head of Cloud Practice & Partner at Neurons Lab
Have you ever wanted to add AI/ML features to your products, but felt intimidated by the technical barriers and high costs? Look no further, because
ChatGPT
has changed the game. With the help of OpenAI, ChatGPT allows you to easily prototype AI/ML solutions without deep knowledge or a big budget. It’s like the personal computer revolution, where everyone suddenly had access to their own machine instead of having to rely on expensive mainframes.
Now, anyone can have their own personal AI to enhance their products.
Fast Prototyping and Customer Feedback
Time is of the essence in the world of product development. The faster you can get feedback from customers, the faster you can improve and succeed. This is where fast prototyping comes in — by quickly creating a prototype, you can gather customer feedback and make the necessary adjustments before fully committing to a solution. ChatGPT makes it easy to prototype AI/ML solutions, allowing you to get customer feedback faster and improve your chances of success.
The Cloud Revolution
The rise of the cloud has been a game changer for startups and product development. The cloud provides an on-demand cost model, managed services, and serverless options, making it easy to scale and deploy solutions without the upfront infrastructure costs. AWS is a leader in the cloud market and by using their services, you can easily prototype and deploy your AI/ML solutions.
A Chatbot Prototyped with Streamlit and Deployed to AWS ECS Fargate
Streamlit
is a powerful tool for prototyping AI/ML solutions and when combined with the
managed services of AWS
, it becomes even more powerful.
By using
CloudFormation
, you can easily deploy your chatbot to
AWS ECS Fargate
, which is a fully managed container service.
Amazon Polly
and
Amazon ECS
can then be used to add voice and container capabilities to your chatbot. The end result is a fully functional chatbot that can be easily integrated into your products and services.
We’ve discussed the benefits of using ChatGPT, fast prototyping, and the cloud to extend your products with AI/ML features. We’ve also provided a detailed guide on how to deploy a chatbot prototyped with Streamlit and deployed to AWS ECS Fargate using CloudFormation, Docker, and AWS ECS.
With these tools, you can easily prototype and deploy AI/ML solutions to enhance your products and services. Don’t let technical barriers hold you back any longer — take advantage of the power of ChatGPT and the cloud to bring your products to the next level.
My code can be found in this
GitHub repo
.
Run Docker Container Locally
To build the Docker image for this project, use the Dockerfile in the ./app directory. You can do this using the docker build command.
For example, to build the image using the Dockerfile in the current directory, you can use the following command:
docker build -t chatgpt-streamlit .
This will build the image and tag it with the name
chatgpt-streamlit.
To run the Docker container locally, you will need to have Docker installed on your machine. You can then use the docker run command to start the container.
Here is an example of how to run the container locally:
docker run -p 8501:8501 -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
-e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
-e AWS_SESSION_TOKEN=$AWS_SESSION_TOKEN \
-e AWS_DEFAULT_REGION=”us-east-1″ \
chatgpt-streamlit
This will start the container and bind the container’s port 8501 to the host’s port 8501. You will also need to set the appropriate AWS access keys and session token as environment variables. The container will use the us-east-1 region as the default region.
Once the container is running, you can access the Streamlit application by visiting
http://localhost:8501
in your web browser.
CloudFormation Stack Deployment
This CloudFormation template creates a Virtual Private Cloud (VPC) with four subnets (two public and two private), an Internet Gateway, and a NAT Gateway. The template also creates four route tables (one for each subnet) and routes for each route table to direct traffic to the Internet Gateway or NAT Gateway as appropriate.
The VPC has a CIDR block of 10.0.0.0/16, and the four subnets are created with CIDR blocks of 10.0.1.0/24, 10.0.2.0/24, 10.0.3.0/24, and 10.0.4.0/24. The public subnets are mapped to have public IP addresses on launch, whereas the private subnets are not.
There are also a few parameters defined at the beginning of the template that allow the user to specify an allowed IP address range for accessing port 80 on the load balancer, the name of an Elastic Container Registry (ECR) repository, and the path to an API key in the SSM Parameter Store.
To deploy the CloudFormation stack for this project, use the ./cf.yaml file. You can do this using the AWS Management Console, the AWS CLI, or the AWS SDKs.
For example, to deploy the stack using the AWS CLI, you can use the following command with default values:
aws cloudformation create-stack \
–stack-name chatgpt-streamlit \
–template-body file://cf.yaml \
–capabilities CAPABILITY_IAM
To deploy the stack using the AWS CLI, you can use the following command with custom values:
aws cloudformation create-stack –stack-name chatgpt-streamlit \
–template-body file://cf.yaml \
–parameters ParameterKey=AllowedIP,ParameterValue=1.2.3.4/32 ParameterKey=Repository,ParameterValue=public.ecr.aws/a9t7y4w6/demo-chatgpt-streamlit:latest ParameterKey=ChatGptApiKeyPath,ParameterValue=/openai/api_key
This will create a new stack called chatgpt-streamlit using the cf.yaml template file and the CAPABILITY_IAM capability.
The Risks of Off-the-Shelf Chatbots for Enterprises
Despite the convenience and low cost of implementing an off-the-shelf chatbot solution like those powered by large language models such as GPT-3, there are several risks that enterprises should carefully consider before going this route:
1.Data Privacy and Security Risks
Services based on models like GPT-3 are trained on vast amounts of data scraped from the public internet. There is little transparency or control over what data these models have been exposed to. Any proprietary or sensitive customer data handled by an off-the-shelf chatbot solution becomes available to the service provider. Enterprises have little legal recourse if data is misused or leaked.
2.Lack of Customization
Generic chatbot solutions have limited ability to be tailored to an enterprise’s specific use cases, terminologies, workflows, and integrations. There is no easy way to customize the chatbot’s capabilities or improve its responses to better serve customers. The chatbot remains a black box.
3.Unreliable and Erratic Behavior
Large language models can demonstrate biased and even toxic behavior in certain contexts. They are prone to hallucinating false information and answering unambiguously incorrectly at times. For customer-facing enterprise applications, the lack of reliability is highly risky.
4.A Case for Building a Custom Chatbot
To avoid these pitfalls, investing in a custom conversational AI solution is advisable. By working with an experienced team of natural language processing experts, enterprises can build chatbots specifically for their needs and use cases.
Key advantages of custom-building include:
Full control over data security and privacy
Ability to tailor conversations to terminology, tone, and workflows
Integrations with internal systems and databases
Ongoing model refinement and human-in-the-loop training
Explainability and auditability of model behavior
Gradual capability ramp-up aligned to use cases
While more complex, developing a proprietary chatbot leveraging expert NLP skills enables greatly reduced business risk and the flexibility to expand the chatbot’s capabilities over time. The payoff in customer satisfaction and efficient operations is well worth the investment.
To Wrap Everything Up
ChatGPT, fast prototyping, and the cloud are game changers for those looking to extend their products with AI/ML features. By using ChatGPT, you can easily prototype solutions and gather customer feedback, although it is important to take into account the risks associated with doing vs the benefits of building a custom chatbot.
The cloud, specifically AWS, provides managed services and an on-demand cost model that make it easy to deploy and scale your solutions. Streamlit and AWS ECS Fargate allow you to easily create a chatbot that can be integrated into your products and services.
About Neurons Lab
Neurons Lab is an AI consultancy that provides end-to-end services – from identifying high-impact AI applications to integrating and scaling the technology. We empower companies to capitalize on AI’s capabilities.
As 1 of 18 certified AWS partners in Applied AI globally, our goal is to help businesses to unlock the full potential of AI technologies with support from our diverse and highly skilled team, made up of applied scientists and PhDs, industry experts, data scientists, AI developers, cloud specialists, user design experts and business strategists with international expertise from across a variety of industries.
Get in touch
with the team to discuss your next AI initiative.
Get in touch
Leverage the power of AI for your business
SHARE THE ARTICLE
Explore success stories from our clients already leveraging the power of AI
All Stories
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
Featured Articles
7 Tips for Effective Risk Management in AI Delivery
Oct 19, 2023
|
7 mins
AI Delivery
Discover the new AI Design Sprint framework for AI feature integration
Aug 17, 2023
|
5 mins
AI Delivery
Explore The Managed Capacity Model for successful AI Solution Development
Nov 6, 2023
AI Delivery

