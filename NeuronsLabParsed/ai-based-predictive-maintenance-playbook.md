## Section 1

AI-based Predictive Maintenance Playbook
Predictive maintenance
Jul 26, 2023
|
7 min read


## Section 2

SHARE THE ARTICLE
Preventive and predictive maintenance are not fantastic technologies within Industry 4.0, today they’re more like standard baseline solutions that are employed by every company that deals with heavy industry and has sensors installed on the machinery.
The question is if this predictive maintenance is done right? Are dashboards with statistical models really a breakthrough and a reliable solution that really changes how the company works? Or is it just a baseline for technologies that really revolutionize the processes and bring tangible changes?
This Playbook aims to show the difference between statistics visualized in the dashboard and AI-powered actionable insights and how to organize a transition from the first to the latter, which is (spoiler!) a bit more sophisticated than changing a couple of formulas.
Baseline and statistical approaches
Every predictive maintenance starts with the data from the sensors, which are heterogeneous, non-standardized, hard to collect and to maintain. This is a challenge we omit in this Playbook, since we’re sure you’ve got rockstar engineers in-house and great partners who installed sensors and had set up a cloud data platform and set of dashboards that show all these temperatures, pressures, vibrations in real time.
Now you want to perform data-driven decisions based on the information presented on the dashboards. How is it usually done? The two major tasks you want to solve (before decision making) are anomalies detection early enough and making accurate long-term predictions about digital health.
How to find anomalies from these numbers in the dashboards? The solution is known to us from the universities – we just record the “healthy” state of every sensor on every machine and then calculate statistical deviations from the normality! We can define soft or hard thresholds to signal about the problems, and, voila, anomaly detection is done!
What do we do with health prediction? Well, this is already much harder, we might want to use heuristics, build mathematical models and, if we already have some data of the “dying” machines, we can build prediction models that, hopefully, can catch the patterns that signal about that “dying” state early enough. Sounds good, right?
Why and where they fail
In reality, such graphs on the dashboards are, at best, not very reliable, and sometimes really misleading and even dangerous. Why so?
First of all, mechanical health depends not on the single graph, but on
dozens of sensors
installed on it. In order to really tell what’s going on, you need to follow all graphs simultaneously. It’s already a challenge, if you have several aggregates but more important, you need to define a group of thresholds for all sensors on every device! Doesn’t look like automated decision making, more like even more work to be done than before and more possibility to mistake in the process
Apart from general overhead with all those sensors, what we really care about is
root cause analysis
– understanding of what exactly led to the anomaly. Statistical threshold-based approaches treat difference sensors as independent entities which is not true! The real physical process is hierarchical, and the change in temperature could be influenced by pressure, which we can’t catch from these dashboards with graphs
Another problem is that anomalies and their reasons
change with time
. Yes, we can adapt thresholds every time we encounter new environmental change, but it can be too late, and again, what it has to do with automation?
The anomalies change because of the
non-linear nature
of underlying processes. It’s never “if something goes up – another thing goes up as well”, especially in complex systems with multiple degrees of freedom. Physicists were exploring all those differential equations for hundreds of years, and today we want to simplify everything with correlations and regressions, that are actually simple linear measures?
Real mathematicians who already use differential equations shouldn’t be very happy either, because most scientific findings are focused to explain some physical phenomena in-sample, i.e. within some relatively short experiment. However, in anomaly detection and in predictions we are interested in accuracy of our
models
performance in the future
, not in some experiment in the past.
Last but not least, most of the simple statistical approaches and mathematical models will
never be able to generalize
to other machineries, data sources and properties. New aggregate – and you have to do all the work literally from scratch. And again, it has nothing to do with automation and all bright promises of the Industry 4.0.
Why can AI help at all?
As you could guess from the title, we are going to discuss why Artificial Intelligence technologies (that are also statistics-based) can overcome the above-mentioned problems. AI methods grew out from applied mathematics, statistics and computer science as already an independent scientific branch with its own theories and best industrial practices.
Let’s review on the examples of celebrated AI technologies how they can fix standard predictive maintenance baselines and what added value they bring with:
Statistics problem
AI concept
Predictive maintenance case
Scalability with number of sensors
VC dimension bounds
Every new sensor brings thousands and millions of new data points, hence, it doesn’t harm, but the opposite, boosts the accuracy of the AI models
Root cause analysis
SHAP feature importance, digital twins
To perform root cause analysis of AI models, we can “reverse-engineer” the impact of each sensor measurement in any time towards the anomaly detected or the prediction
Environmental changes
Retraining schedule and regularization
Compared to the hard statistically inferred thresholds, we can make AI models “soft”, i.e. regularized and constantly automatically updated over time to adjust to the changes
Non-linear data nature
Gradient boosting trees and deep neural networks
Most of AI models by definition have non-linear kernels, deep neural networks and boosting trees are the most used in the industry families of such algorithms
Performance in the future
Cross-validation and scenario-based analysis
Instead of in-sample variance justification of mathematical models, AI models aim to maximize performance out-of-sample and specific scenarios that might happen
Generalization and Standardization
Autoencoders, GANs, digital twins
Looks very promising, right? Now we just need to integrate it into our dashboards! Seems like a couple of months of re-coding some formulas will do the job… or not?
How AI is integrated
Self-learning systems are different from classical digitalization and automation processes. Digitalization and automation are not changing current processes, they optimize them with respect to some metrics, meanwhile AI introduces new processes, activities, policies and even positions in the company in order to unlock underlying benefits. At least the good point is, that AI definitely creates more jobs than it wipes away.
Diagnostics and discovery to establish a baseline (Current processes are measured; Internal stakeholders assigned to track the progress; Data pipelines are created for major data sources)
Pilot projects execution to gain momentum (First data-driven processes are implemented and value is measured; Data warehouse created for future data collection and improvements; Key stakeholders are updated on the progress in the pilots)
In-house team development and broad AI training (Internal “Head of data” or “Head of AI” assigned, even if still working with external teams; Key stakeholders are educated about AI possibilities in the context of current automations; Data and AI routines are automated (feedback, re-training, troubleshooting, etc)
AI strategy development and communication (Define a strategy in the “Client-Data-Action” loop; Extend current warehouses with respect to the strategic directions; Communicate new policies and directions to the key stakeholders)
Data-driven and research process integration (Expand the strategy to other departments and divisions; Educate managers and stakeholders on AI capabilities there; Execute more pilot projects and measure the value
To get more details about the engineering process at Neurons Lab (the most related to the steps 1 and 2 in the integration pipeline),
contact us for a consultation
.
Success stories
Neurons Lab’s services are tailored into building custom AI solutions. The strategy to success lies in exceptional collaboration conditions with leading experts who have deep expertise not only in the AI algorithms, but in the specific business area. In heavy industry, Neurons Lab has completed several projects.
One of them was gas consumption optimization for steelmaking companies, where we could decrease analytics time from hours to minutes and improve accuracy by 15%, in another one we have developed a system for fully automated control over the kite for renewable energy generation, which could completely remove humans from the control routines.
The largest diamond in the collection is development of a predictive maintenance solution for an Eastern European power plant station. The objective of this project was to develop a system for predicting the failure of various equipment. Informing of employees at the right time makes it possible in time to identify problem areas of the system and take the necessary measures. At the end of the project, the following metrics were achieved:
Reduced equipment downtime due to breakdowns by 20%;
Service time decreased by 30%;
Reduced time to detect breakdowns hundreds of times
We also configured the system for collecting and storing data on the operation of the equipment.
To see this solution in action, you can schedule an interactive demo session or text to
[email protected]
.
Benefits and Super-Benefits
Apart from the technical excellence, what business benefits are unlocked with uplifting your current predictive maintenance baseline with the AI technologies?
Uninterrupted production and performance
Measured reducement of maintenances time and cost
Standardization and scaling towards regional and global objects
To learn more about how to speed up AI transformation for your predictive maintenance routine, ping us at
[email protected]
! Long life to your machinery!
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

