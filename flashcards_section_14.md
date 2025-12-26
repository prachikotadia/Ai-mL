# Section 14: MLOps & Deployment

## 651. MLOps
🟦 **What is MLOps?**

🟩 **Definition**
MLOps is the practice of building, deploying, and maintaining ML systems reliably. It combines machine learning with software engineering and operations. The goal is to ship models safely and keep them working over time.

🟨 **How It Works / Example**
A team trains a model, packages it, and deploys it as an API. They monitor accuracy and latency in production. If performance drops, they retrain and redeploy with a controlled process.

🟪 **Quick Tip**
DevOps for ML.

---

## 652. MLOps Importance
🟦 **Why is MLOps important for production ML?**

🟩 **Definition**
Models can fail in production due to data changes, bugs, or scaling issues. MLOps adds testing, monitoring, and automation to reduce these risks. It helps teams deliver stable ML features to users.

🟨 **How It Works / Example**
A fraud model may work well today but degrade as attackers change behavior. Monitoring detects the drop early. Automated retraining and safe rollout keep performance strong.

🟪 **Quick Tip**
Reliability & scale.

---

## 653. Model Deployment
🟦 **What is a model deployment?**

🟩 **Definition**
Model deployment is making a trained model available for real users or systems. It can be an API, a batch job, or an on-device model. Deployment also includes versioning and rollback plans.

🟨 **How It Works / Example**
You export a model and serve it behind a REST endpoint. Your app sends features to the endpoint and receives predictions. If something breaks, you roll back to the previous model version.

🟪 **Quick Tip**
Serving models.

---

## 654. Serving System
🟦 **What is a model serving system?**

🟩 **Definition**
A model serving system runs models in production and returns predictions. It handles scaling, routing, and latency requirements. It often includes logging and monitoring.

🟨 **How It Works / Example**
A service receives a request, loads the correct model version, and runs inference. It returns a score like "fraud probability." Metrics like QPS and latency are tracked continuously.

🟪 **Quick Tip**
Inference server.

---

## 655. Online vs Batch
🟦 **What is online inference vs batch inference?**

🟩 **Definition**
Online inference returns predictions in real time for user requests. Batch inference runs predictions on large datasets on a schedule. The choice depends on latency needs and use case.

🟨 **How It Works / Example**
A ride pricing model needs online inference within milliseconds. A churn model can run nightly batch jobs for all users. Both can use the same model but different serving methods.

🟪 **Quick Tip**
Real-time vs Scheduled.

---

## 656. Feature Store
🟦 **What is a feature store?**

🟩 **Definition**
A feature store is a system to store, manage, and serve features for ML. It helps keep features consistent between training and inference. It also supports reuse across models.

🟨 **How It Works / Example**
You compute "30-day purchase count" once and store it in a feature store. Training pipelines read it for model building. Online services read the same feature for real-time predictions.

🟪 **Quick Tip**
Consistent features.

---

## 657. Training-Serving Skew
🟦 **What is training-serving skew?**

🟩 **Definition**
Training-serving skew happens when features used in training differ from features used in production. This causes model performance to drop. It is a common production ML issue.

🟨 **How It Works / Example**
In training you compute a feature with full data, but in production the feature is missing or computed differently. Predictions become unreliable. Using a feature store and shared feature code reduces skew.

🟪 **Quick Tip**
Data mismatch.

---

## 658. Model Versioning
🟦 **What is model versioning?**

🟩 **Definition**
Model versioning tracks different model builds over time. It includes weights, code, data, and metadata. Versioning supports rollback and reproducibility.

🟨 **How It Works / Example**
You label models as v1, v2, v3 with training date and metrics. If v3 causes issues, you deploy v2 again quickly. Version logs also help you audit what changed.

🟪 **Quick Tip**
Tracking changes.

---

## 659. Model Registry
🟦 **What is a model registry?**

🟩 **Definition**
A model registry stores models and their versions in a managed system. It tracks metrics, approvals, and deployment stage. It helps teams control what goes to production.

🟨 **How It Works / Example**
After training, you register the model with validation results. A reviewer promotes it to "staging" then "production." Serving systems pull the production model automatically.

🟪 **Quick Tip**
Model library.

---

## 660. ML CI/CD
🟦 **What is CI/CD for machine learning?**

🟩 **Definition**
CI/CD automates testing, building, and deploying ML code and models. It reduces manual errors and speeds up releases. In ML, it also includes data checks and model validation.

🟨 **How It Works / Example**
A new model training change triggers tests and a training run. If metrics pass thresholds, the pipeline packages the model and deploys it to staging. After approval, it rolls out to production.

🟪 **Quick Tip**
Automated pipeline.

---

## 661. Training Pipeline
🟦 **What is a training pipeline in MLOps?**

🟩 **Definition**
A training pipeline is an automated workflow to prepare data, train a model, and evaluate it. It makes model building repeatable and reliable. It often runs on schedules or triggers.

🟨 **How It Works / Example**
A daily pipeline pulls new data, cleans it, and trains a model. It evaluates performance on a validation set. If results meet criteria, it registers the new model version.

🟪 **Quick Tip**
Automated training.

---

## 662. Data Pipeline
🟦 **What is a data pipeline in MLOps?**

🟩 **Definition**
A data pipeline collects, cleans, and transforms data for training or inference. It ensures data arrives on time and in the right format. Data pipeline reliability is critical for ML quality.

🟨 **How It Works / Example**
User events stream into a warehouse like BigQuery or Snowflake. A job builds aggregated features each night. The model training pipeline reads those features for learning.

🟪 **Quick Tip**
Data flow.

---

## 663. Data Validation
🟦 **What is data validation in MLOps?**

🟩 **Definition**
Data validation checks data quality before training or inference. It catches issues like missing values, wrong ranges, or schema changes. It prevents training on broken data.

🟨 **How It Works / Example**
If "age" suddenly becomes a string instead of a number, validation fails. The pipeline stops and alerts the team. This avoids deploying a model trained on corrupted inputs.

🟪 **Quick Tip**
Quality check.

---

## 664. Schema Drift
🟦 **What is schema drift in production data?**

🟩 **Definition**
Schema drift is when the structure of input data changes over time. Columns may be added, removed, or renamed. It can break feature extraction and model serving.

🟨 **How It Works / Example**
An upstream team renames "user_id" to "uid." Your model service can't find the expected field. Schema checks detect this quickly so you can update or roll back.

🟪 **Quick Tip**
Structure change.

---

## 665. Data Drift
🟦 **What is data drift in ML systems?**

🟩 **Definition**
Data drift is when the distribution of input features changes over time. This can reduce model accuracy. Drift can happen due to seasonality, user behavior changes, or product updates.

🟨 **How It Works / Example**
A fraud model trained on last year’s patterns sees new fraud strategies. Feature distributions shift and the model becomes less accurate. Drift monitoring alerts the team to retrain or adjust.

🟪 **Quick Tip**
Distribution shift.

---

## 666. Concept Drift
🟦 **What is concept drift?**

🟩 **Definition**
Concept drift is when the relationship between inputs and labels changes. Even if feature distributions are stable, the meaning of patterns can change. It often causes real performance drops.

🟨 **How It Works / Example**
Users start clicking different items due to a UI redesign. The model’s learned mapping from features to clicks becomes outdated. Retraining on new labeled data helps.

🟪 **Quick Tip**
Meaning shift.

---

## 667. Model Monitoring
🟦 **What is model monitoring in production?**

🟩 **Definition**
Model monitoring tracks system health and prediction quality after deployment. It includes latency, errors, drift, and business metrics. Monitoring helps detect issues early.

🟨 **How It Works / Example**
You track request rate, p95 latency, and prediction distribution. You also track downstream metrics like conversion rate. If metrics shift unexpectedly, alerts notify the team.

🟪 **Quick Tip**
Production health.

---

## 668. Delayed Labels
🟦 **What is model performance monitoring when labels arrive late?**

🟩 **Definition**
Sometimes true labels are not available immediately, like churn or fraud. You can still monitor proxies and distributions in the short term. When labels arrive, you compute true metrics and compare over time.

🟨 **How It Works / Example**
For fraud, chargebacks may arrive weeks later. You monitor input drift and score distribution daily. When chargebacks arrive, you compute AUROC or precision and update dashboards.

🟪 **Quick Tip**
Lagging feedback.

---

## 669. Drift Detection
🟦 **What is model drift detection?**

🟩 **Definition**
Model drift detection finds when a model’s behavior or accuracy changes over time. It uses drift metrics, distribution checks, and performance signals. It helps decide when to retrain.

🟨 **How It Works / Example**
You compare current feature distributions to training distributions using statistical tests. You also check if score histograms shift. If drift is large, you investigate and consider retraining.

🟪 **Quick Tip**
Sensing change.

---

## 670. Alerting System
🟦 **What is an alerting system in MLOps?**

🟩 **Definition**
Alerting notifies the team when metrics cross thresholds. It helps you respond quickly to failures. Alerts should be actionable and not too noisy.

🟨 **How It Works / Example**
If p95 latency goes above 200ms, an alert triggers. If prediction values become mostly zeros, another alert triggers. Engineers check logs, roll back if needed, and fix the issue.

🟪 **Quick Tip**
Incident warning.

---

## 671. Rollback
🟦 **What is a rollback in model deployment?**

🟩 **Definition**
Rollback means switching back to a previous stable model or system version. It reduces user impact when a new model causes problems. Rollbacks should be fast and well-tested.

🟨 **How It Works / Example**
A new model increases error rates in production. You flip traffic back to the previous model version. Then you debug offline without harming users further.

🟪 **Quick Tip**
Undo deploy.

---

## 672. Canary Deployment
🟦 **What is a canary deployment for ML models?**

🟩 **Definition**
A canary deployment sends a small portion of traffic to a new model first. It checks stability and guardrails before full rollout. It reduces risk compared to a full switch.

🟨 **How It Works / Example**
You route 1% of requests to the new model. You monitor latency and key metrics for a few hours. If all is good, you expand to 10%, then 50%, then 100%.

🟪 **Quick Tip**
Gradual rollout.

---

## 673. Shadow Deployment
🟦 **What is shadow deployment for ML models?**

🟩 **Definition**
Shadow deployment runs a new model in parallel without affecting user outputs. It collects predictions and metrics safely. It is useful before an A/B test or rollout.

🟨 **How It Works / Example**
The old model still serves real responses. The new model receives the same inputs and produces outputs in the background. You compare outputs and latency to evaluate readiness.

🟪 **Quick Tip**
Silent test.

---

## 674. A/B Testing Deployment
🟦 **What is A/B testing for model deployment?**

🟩 **Definition**
A/B testing compares a new model to the current model using real user traffic. It measures impact on business and user metrics. It is the best way to validate real-world improvement.

🟨 **How It Works / Example**
50% of users get the old model and 50% get the new one. You track conversion, retention, and latency. If the new model wins and guardrails are safe, you roll it out fully.

🟪 **Quick Tip**
Direct comparison.

---

## 675. Model Endpoint
🟦 **What is a model endpoint?**

🟩 **Definition**
A model endpoint is a network interface (often an API) that returns predictions. It accepts input features and returns outputs like scores or labels. It must be reliable and fast.

🟨 **How It Works / Example**
A client sends JSON features to `/predict`. The server runs the model and returns a probability. The calling application uses that probability to make decisions like approve or review.

🟪 **Quick Tip**
API access.

---

## 676. Inference Pipeline
🟦 **What is an inference pipeline?**

🟩 **Definition**
An inference pipeline includes preprocessing, model prediction, and postprocessing. It ensures inputs are transformed correctly and outputs are usable. Many bugs come from inconsistent preprocessing.

🟨 **How It Works / Example**
You scale numeric features, one-hot encode categories, then run the model. After prediction, you convert a probability into a decision threshold. The full pipeline is packaged and deployed together.

🟪 **Quick Tip**
Inputs to outputs.

---

## 677. Preprocessing Consistency
🟦 **What is preprocessing consistency in deployment?**

🟩 **Definition**
Preprocessing consistency means doing the same feature transformations in training and in production. Inconsistent scaling or encoding can break model behavior. It is a major MLOps concern.

🟨 **How It Works / Example**
If training uses mean-std scaling but production uses min-max scaling, predictions can be wrong. Teams solve this by exporting the preprocessing steps with the model. Feature stores and shared libraries help.

🟪 **Quick Tip**
Matching logic.

---

## 678. Model Packaging
🟦 **What is model packaging?**

🟩 **Definition**
Model packaging bundles the model and its dependencies for deployment. It may include preprocessing code, config files, and artifacts. Packaging makes deployments repeatable.

🟨 **How It Works / Example**
You build a Docker image containing the model weights, Python environment, and inference code. You tag it with a version. Then you deploy the same image across staging and production.

🟪 **Quick Tip**
Bundling code.

---

## 679. Docker in MLOps
🟦 **What is Docker and why is it used in MLOps?**

🟩 **Definition**
Docker is a tool for packaging software into containers. It ensures the same environment runs in development and production. This reduces "it works on my machine" problems.

🟨 **How It Works / Example**
You create a Dockerfile that installs dependencies and copies model code. The container runs the inference server. Kubernetes or a cloud service then scales these containers as needed.

🟪 **Quick Tip**
Containerization.

---

## 680. Kubernetes
🟦 **What is Kubernetes and how is it used for ML deployment?**

🟩 **Definition**
Kubernetes is a system for running and scaling containers. It helps manage replicas, load balancing, and updates. Many ML services run on Kubernetes for reliability.

🟨 **How It Works / Example**
You deploy your model container as a Kubernetes deployment. Kubernetes automatically restarts crashed pods. It can also scale up replicas when traffic increases.

🟪 **Quick Tip**
Container orchestration.

---

## 681. Autoscaling
🟦 **What is autoscaling for model serving?**

🟩 **Definition**
Autoscaling adjusts the number of serving instances based on load. It helps handle traffic spikes while controlling cost. It can scale by CPU, GPU, or request rate.

🟨 **How It Works / Example**
If requests per second increases, the system adds more pods. When traffic drops, it removes pods. This keeps latency stable without paying for unused capacity.

🟪 **Quick Tip**
Dynamic capacity.

---

## 682. Latency
🟦 **What is latency and why is it critical in ML serving?**

🟩 **Definition**
Latency is how long it takes to return a prediction. Many products need low latency to feel responsive. High latency can reduce user satisfaction and revenue.

🟨 **How It Works / Example**
A search ranking model must respond in tens of milliseconds. If inference takes 300ms, users may leave. Teams optimize with batching, caching, and smaller models.

🟪 **Quick Tip**
Response time.

---

## 683. Throughput
🟦 **What is throughput in model serving?**

🟩 **Definition**
Throughput is how many requests a system can handle per second. High throughput is needed for large-scale products. Throughput depends on model size, hardware, and batching.

🟨 **How It Works / Example**
A system might serve 5,000 predictions per second during peak time. If throughput is too low, requests queue up and latency increases. Adding instances or using batching can improve throughput.

🟪 **Quick Tip**
Requests per second.

---

## 684. Batching
🟦 **What is batching in inference?**

🟩 **Definition**
Batching groups multiple requests together to run inference more efficiently. It improves GPU utilization and throughput. It can increase latency if you wait too long to form batches.

🟨 **How It Works / Example**
An LLM server collects requests for 20ms and runs them as one batch. This speeds total processing. You tune batch size and waiting time to balance latency and efficiency.

🟪 **Quick Tip**
Grouping requests.

---

## 685. Caching
🟦 **What is caching in ML serving?**

🟩 **Definition**
Caching stores results so repeated requests are faster. It reduces compute cost and latency. Caching works best when many requests repeat or have common parts.

🟨 **How It Works / Example**
For embeddings, you cache the embedding of common queries. For LLMs, you might cache frequent FAQ answers. When a repeat request comes, you return cached results instantly.

🟪 **Quick Tip**
Saving results.

---

## 686. Model Compression
🟦 **What is model compression?**

🟩 **Definition**
Model compression reduces model size to speed inference and cut cost. Common methods include pruning, quantization, and distillation. Compression can reduce accuracy if done poorly.

🟨 **How It Works / Example**
You quantize weights from 32-bit to 8-bit to run faster. Or you distill a large model into a smaller one. Then you test accuracy and latency trade-offs before deployment.

🟪 **Quick Tip**
Shrinking models.

---

## 687. Quantization
🟦 **What is quantization in deployment?**

🟩 **Definition**
Quantization reduces numeric precision of model weights and activations. It speeds up inference and reduces memory use. It is common for LLM serving and edge devices.

🟨 **How It Works / Example**
You convert a model from FP32 to INT8. The model runs faster and uses less memory. You validate that accuracy stays acceptable after quantization.

🟪 **Quick Tip**
Lower precision.

---

## 688. Model Distillation
🟦 **What is model distillation?**

🟩 **Definition**
Distillation trains a smaller "student" model to mimic a larger "teacher" model. It can keep much of the performance while being faster. It is useful when latency or cost is tight.

🟨 **How It Works / Example**
A large LLM produces soft probabilities or outputs. The student is trained to match those outputs. The student model then serves in production with lower cost.

🟪 **Quick Tip**
Teacher-student.

---

## 689. Model Pruning
🟦 **What is model pruning?**

🟩 **Definition**
Pruning removes less important weights or neurons to reduce model size. It can speed inference and reduce memory. It often requires fine-tuning after pruning.

🟨 **How It Works / Example**
You remove weights with very small magnitude. Then you fine-tune the model to recover accuracy. If pruning is too aggressive, performance can drop sharply.

🟪 **Quick Tip**
Cutting weights.

---

## 690. SLA (Service Level Agreement)
🟦 **What is an SLA and why does it matter for ML services?**

🟩 **Definition**
An SLA (Service Level Agreement) defines reliability and latency targets, like 99.9% uptime. ML services must meet these targets to support products. SLAs guide capacity planning and monitoring.

🟨 **How It Works / Example**
An SLA may require p95 latency under 100ms. You monitor latency dashboards and set alerts. If latency rises, autoscaling or rollback helps maintain the SLA.

🟪 **Quick Tip**
Performance promises.

---

## 691. Fallback Strategy
🟦 **What is a fallback strategy in ML deployment?**

🟩 **Definition**
A fallback strategy is what the system does when the model fails or is slow. It keeps the product working even during incidents. Fallbacks improve reliability and user experience.

🟨 **How It Works / Example**
If the recommender service times out, you show popular items instead. If an LLM is unavailable, you route to a simpler FAQ system. Users still get something useful instead of an error.

🟪 **Quick Tip**
Plan B.

---

## 692. Observability
🟦 **What is observability for ML systems?**

🟩 **Definition**
Observability means being able to understand system behavior using logs, metrics, and traces. It helps debug latency spikes, errors, and quality issues. ML observability also includes drift and model outputs.

🟨 **How It Works / Example**
You trace a request through preprocessing, model inference, and postprocessing. Metrics show where time is spent. Logs store inputs and predictions so you can investigate abnormal outputs.

🟪 **Quick Tip**
System visibility.

---

## 693. Feature Importance Monitoring
🟦 **What is feature importance monitoring in production?**

🟩 **Definition**
Feature importance monitoring checks which inputs the model relies on over time. If importance shifts unexpectedly, it may signal drift or bugs. It helps detect silent failures.

🟨 **How It Works / Example**
If a model suddenly relies heavily on a "missing_value" indicator, something may be wrong upstream. You investigate feature computation. Fixing the pipeline can restore normal behavior.

🟪 **Quick Tip**
Input tracking.

---

## 694. Data Leakage Risk
🟦 **What is data leakage risk in deployment pipelines?**

🟩 **Definition**
Data leakage is when training uses information that would not be available at prediction time. It can make offline metrics look great but fail in production. MLOps checks help catch leakage early.

🟨 **How It Works / Example**
A churn model might accidentally use "cancel date" as a feature. That feature is only known after churn happens. The model looks perfect offline but is useless online, so pipelines must validate feature availability.

🟪 **Quick Tip**
Using future data.

---

## 695. Model Governance
🟦 **What is model governance in MLOps?**

🟩 **Definition**
Model governance is the process of managing risk, compliance, and approvals for models. It includes documentation, audits, and access controls. Governance is important in regulated industries.

🟨 **How It Works / Example**
A bank requires model documentation, bias checks, and approval before production. The registry stores approval status. Only approved versions can be deployed to production.

🟪 **Quick Tip**
Compliance & control.

---

## 696. Model Documentation
🟦 **What is model documentation in production ML?**

🟩 **Definition**
Model documentation describes what the model does, how it was trained, and its limitations. It helps onboarding, debugging, and compliance. Good documentation reduces operational risk.

🟨 **How It Works / Example**
You write a model card including training data, metrics, and known failure cases. You include the intended use and forbidden use. This helps teams deploy and monitor the model responsibly.

🟪 **Quick Tip**
Model manual.

---

## 697. Incident Response
🟦 **What is an incident response process for ML systems?**

🟩 **Definition**
Incident response is how a team reacts to production issues like outages or wrong predictions. It includes detection, mitigation, and postmortems. A clear process reduces downtime and confusion.

🟨 **How It Works / Example**
An alert shows prediction values are abnormal. The on-call engineer rolls back to a stable model and investigates logs. After recovery, the team writes a postmortem and fixes the root cause.

🟪 **Quick Tip**
Handling failures.

---

## 698. Retraining
🟦 **What is retraining and why is it needed?**

🟩 **Definition**
Retraining updates a model using newer data to handle drift and keep accuracy high. Many real-world systems need retraining regularly. Retraining can be scheduled or triggered by monitoring.

🟨 **How It Works / Example**
A weekly pipeline retrains a demand forecasting model with last week’s data. It evaluates and compares against the current production model. If it improves, it is deployed through a safe rollout.

🟪 **Quick Tip**
Updating knowledge.

---

## 699. Continuous Training
🟦 **What is continuous training (CT) in MLOps?**

🟩 **Definition**
Continuous training automates retraining and validation over time. It is like CI/CD but focused on model refresh. It helps maintain performance in changing environments.

🟨 **How It Works / Example**
New data arrives daily and triggers a training pipeline. The pipeline trains a candidate model and runs tests. If it passes, it registers and deploys the model with canary checks.

🟪 **Quick Tip**
Automated refresh.

---

## 700. ML Deployment Lifecycle
🟦 **What is a complete ML deployment lifecycle?**

🟩 **Definition**
The lifecycle includes data collection, training, validation, deployment, monitoring, and retraining. Each step needs checks and automation for reliability. The goal is stable model performance over time.

🟨 **How It Works / Example**
You train a model, deploy it with canary rollout, and monitor drift and business metrics. If drift appears, you retrain and compare against baseline. Then you promote a new version and keep repeating the cycle.

🟪 **Quick Tip**
Full ML loop.
