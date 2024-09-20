## Section 1

CLIENT: INSURANCE CLIENT
INDUSTRY: INSURANCE
AI-driven automated medical insurance claim analysis to detect fraud


## Section 2

Neurons Lab
delivered a cutting-edge AI-powered tool to support treatment claim authorization processes for
one of the world’s largest insurance companies
.
Business Challenges
To authorize insurance claims,
medical officers manually review each treatment claim
submitted by hospitals to determine its legitimacy.
This process involves analyzing PDFs with prescribed treatment details – procedures, medications, and other expenses – and determining their legitimacy for the specific diagnosis.
Challenges include:
Time-consuming manual review
of extensive documentation.
Potential for
human error and inconsistency
in applying international guidelines.
Delays in claim approvals
due to the manual process.
Losing money
due to fraudulent claims and overtreatment.
Project Overview
Neurons Lab implemented an AI-powered tool
leveraging a
large language model (LLM)
and a
knowledge graph
of clinical guidelines.
This tool analyzes treatment claims, compares them with international guidelines, and provides assessments to facilitate
rapid decision-making
. In addition, it includes a
Human-in-the-Loop (HITL)
validation mechanism for doctor validation, ensuring continuous model improvement.
AI-based claim analysis
:
The AI tool analyzes treatment claims by extracting individual treatment elements – procedures, medications, etc. – and comparing them against
relevant international guidelines and patient-specific data
. It assesses treatment appropriateness in the context of the patient’s medical history, diagnosis, and other relevant clinical factors.
Nuanced scoring and explanation
: The solution provides a nuanced scoring system that reflects the complexity of clinical judgment. Each treatment element is assigned a clear color code and a
score based on adherence to guidelines, potential risks, and alternative options
. The solution also generates detailed explanations highlighting specific guideline deviations, potential risks, alternative treatment options, and the assessment’s rationale.
HITL validation
: Medical officers can review the AI-generated assessments and provide feedback through a
user-friendly interface
. Their feedback is used to refine the AI model’s performance and ensure accuracy continuously.
Solution
Our proprietary, highly customized SaaS AI solution uses generative AI from
Anthropic Claude 3
in
Amazon Bedrock
and a
comprehensive knowledge base
of treatment procedures and medicines to automate claim analysis. It
streamlines the medical necessity review process and provides expert recommendations.
1. Document input
The user, a medical officer, uploads the treatment claim document into the AI tool, or it can be automatically uploaded.
2. AI Analysis
The
analysis stage
involves:
Intelligent Document Processing (IDP)
:
The AI tool processes the claims, extracting individual treatment elements – procedures, medications, and other information – row by row.
Knowledge base enrichment
: The extracted information is linked to a comprehensive knowledge base.
Contextual matching
: The AI tool compares the extracted treatment element with the corresponding recommendations in the pre-defined international guidelines databases. It also retrieves the patient’s medical history, diagnoses, and other relevant information from the database.
Based on the level of concurrence, the AI tool gives each element a score corresponding to the relevance of the treatment in the submitted claim.
Relevant (100%)
:
A
full match with guidelines and procedures.
Relevant—with a warning (99-75%)
:
It matches the guidelines, but alternative options may exist, or further investigation is needed.
Not relevant (<75%)
:
Either
there is no match or significant deviation from guidelines.
The AI-predicted color-coded relevance score is displayed on the processed claim for each element.
For each treatment element suggested in the insurance claim, the LLM generates a
detailed natural language explanation
for the assigned relevance score based on its concurrence with international treatment guidelines and patient medical data.
The explanation
highlights key factors impacting the score and supporting evidence
– for example, links to relevant sections in the source guidelines used for comparison.
For more information about IDP… Take a look at our
IDP guide for the modern enterprise
and how to
maximize its ROI
.
3. Doctor validation
The medical officer:
Reviews the AI-generated assessments
for each treatment element.
Can access the specific international treatment recommendations
used for comparison through the link and other relevant details for the claim – such as the patient’s medical history and diagnoses.
Adds their professional judgment
to the input field next to the AI generation relevance score based on the same rules as before.
May add additional comments to the specific treatment element. Their input is captured through the HITL interface, facilitating continuous model improvement.
4. Decision and rationale:
Once the treatment assessment is finalized, the
medical officer decides on the claim
.
The medical officer makes a final decision—approve, deny, or request further information—on each claim based on AI-generated insights and their own clinical judgment.
The medical officer documents the rationale for approving or denying the treatment. The insurance provider’s system records the decision for audit and compliance purposes.
Results
Key benefits include:
Fraud detection and cost containment
: Insurance companies’ primary goal is to ensure that claims are legitimate and adhere to established guidelines.
This system, powered by AI, helps flag potential fraud
—for example, unnecessary procedures and overtreatment—
saving the provider significant costs
.
Efficient claims processing
:
The
automated analysis and clear explanations streamline the review process for medical officers
. This leads to faster claim decisions, improved customer satisfaction, and reduced administrative overhead.
Objective and consistent assessments
:
The AI consistently applies standardized guidelines,
reducing the risk of human error and bias
in claim decisions. This ensures fairness and can help defend against potential disputes.
Data-driven insights
:
The system can gather data on claim patterns, treatment trends, and guideline adherence. This information can be used to
identify areas for improvement, refine insurance policies, and negotiate better rates with healthcare providers
.
Continuous improvement
:
HITL validation allows the system to learn from medical officer feedback and become more accurate while
adapting to changing guidelines and medical practices
.
Combining AI with human expertise can empower insurance teams to make informed decisions faster, leading to significant cost savings and improved healthcare outcomes.
About Neurons Lab
Neurons Lab is an AI consultancy that provides end-to-end services – from identifying high-impact AI applications to integrating and scaling the technology. We empower companies to capitalize on AI’s capabilities.
As an AWS Advanced Partner, our global team comprises data scientists, subject matter experts, and cloud specialists supported by an extensive talent pool of 500 experts. We solve the most complex AI challenges, mobilizing and delivering with outstanding speed to support urgent priorities and strategic long-term needs.
Ready to leverage AI for your business? Get in touch with the team
here
.
Get in touch
Leverage the power of AI for your business.
Related Stories
HealthTech client:
Building a Remote Patient Monitoring system for health providers
HealthTech
Explore story
Creative Practice Solutions :
Developing an AI-Driven Medical Transcription & Billing System
HealthTech
Explore story
Treatline:
Implementing Intelligent Document Processing to streamline the Prior Authorization Process in Healthcare
HealthTech, InsurTech
Explore story

