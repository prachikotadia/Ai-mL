# Section 13: Experimentation & A/B Testing

## 601. Experimentation
🟦 **What is experimentation in machine learning?**

🟩 **Definition**
Experimentation is the process of testing changes to models or systems to see what works better. It helps you make decisions using data instead of guesses. In ML, experiments can be offline, online, or both.

🟨 **How It Works / Example**
You try a new feature set or a new model and compare it to the old one. You track metrics like accuracy or user clicks. If the new version improves metrics, you consider deploying it.

🟪 **Quick Tip**
Testing changes.

---

## 602. A/B Testing
🟦 **What is an A/B test?**

🟩 **Definition**
An A/B test compares two versions of a system: A (control) and B (variant). Users are split into groups so you can measure which version performs better. It is widely used to evaluate product changes.

🟨 **How It Works / Example**
Half of users see the old ranking model and half see the new one. You compare outcomes like conversion rate or time on page. If the difference is significant, you choose the better version.

🟪 **Quick Tip**
Comparing versions.

---

## 603. Importance of A/B Testing
🟦 **Why is A/B testing important for ML products?**

🟩 **Definition**
Offline metrics do not always predict real user outcomes. A/B testing measures real-world impact with real users. It helps avoid launching changes that look good in lab tests but hurt the product.

🟨 **How It Works / Example**
A new recommender model may improve offline precision but reduce user engagement because it feels repetitive. A/B testing reveals this quickly. You then adjust the model or choose not to ship.

🟪 **Quick Tip**
Data-driven decisions.

---

## 604. Control Group
🟦 **What is a control group in an A/B test?**

🟩 **Definition**
The control group is the baseline version you compare against. It usually runs the current production system. It provides the reference point for measuring improvement or regression.

🟨 **How It Works / Example**
If you are testing a new LLM prompt, the control uses the old prompt. The treatment uses the new prompt. You compare metrics like user satisfaction or resolution rate between groups.

🟪 **Quick Tip**
Baseline reference.

---

## 605. Treatment Group
🟦 **What is a treatment group in an A/B test?**

🟩 **Definition**
The treatment group uses the new version being tested. It is compared against the control. The goal is to measure if the treatment produces better outcomes.

🟨 **How It Works / Example**
Treatment users see the new recommendation algorithm. You track their click-through rate and watch time. If treatment improves these, the change may be beneficial.

🟪 **Quick Tip**
New version.

---

## 606. Randomization
🟦 **What is randomization in A/B testing?**

🟩 **Definition**
Randomization assigns users to groups by chance. It helps ensure groups are similar so differences are caused by the change, not user bias. Good randomization is key for valid results.

🟨 **How It Works / Example**
You randomly assign each user ID to control or treatment. This prevents placing more power users in one group. Then you can trust the outcome comparison more.

🟪 **Quick Tip**
Reducing bias.

---

## 607. Experiment Metric
🟦 **What is a metric in an experiment?**

🟩 **Definition**
A metric is a number that measures performance, like click-through rate or revenue. Metrics guide decisions about whether a change is better. Choosing the right metrics is crucial.

🟨 **How It Works / Example**
For a chatbot, metrics could include resolution rate, user satisfaction score, and escalation rate. You compare these between control and treatment. You ship only if key metrics improve without harming others.

🟪 **Quick Tip**
Performance measure.

---

## 608. Primary Metric
🟦 **What is a primary metric in A/B testing?**

🟩 **Definition**
A primary metric is the main success measure for the experiment. It is chosen before running the test to avoid bias. Decisions are mainly based on this metric.

🟨 **How It Works / Example**
If your goal is revenue, the primary metric might be conversion rate. You still track other metrics, but conversion drives the final call. This prevents changing goals after seeing results.

🟪 **Quick Tip**
Main goal.

---

## 609. Secondary Metric
🟦 **What is a secondary metric in experimentation?**

🟩 **Definition**
Secondary metrics provide extra context beyond the primary metric. They help explain why results changed and detect side effects. They are not usually the main decision maker.

🟨 **How It Works / Example**
Primary metric might be engagement, while secondary metrics include latency and bounce rate. If engagement improves but latency worsens, you investigate. Secondary metrics help balance the decision.

🟪 **Quick Tip**
Context/Side-effects.

---

## 610. Guardrail Metric
🟦 **What is a guardrail metric?**

🟩 **Definition**
A guardrail metric is a "must not get worse" metric. It protects against harmful side effects. If a guardrail fails, you stop or roll back the experiment.

🟨 **How It Works / Example**
For an LLM support bot, a guardrail could be safety violation rate or complaint rate. Even if resolution improves, you do not ship if safety worsens. Guardrails keep experiments responsible.

🟪 **Quick Tip**
Safety check.

---

## 611. Statistical Significance
🟦 **What is statistical significance in A/B testing?**

🟩 **Definition**
Statistical significance measures whether a difference is likely real or just random noise. It is often checked using a p-value or confidence interval. Significance helps avoid false conclusions.

🟨 **How It Works / Example**
If conversion is 10.0% in control and 10.5% in treatment, you test if that gap is significant. If p-value is below a threshold like 0.05, you treat it as likely real. If not, you may need more data.

🟪 **Quick Tip**
Real vs Random.

---

## 612. P-Value
🟦 **What is a p-value in experiments?**

🟩 **Definition**
A p-value is the probability of seeing a result at least as extreme as yours if there is actually no real difference. Smaller p-values suggest stronger evidence of a real effect. It does not measure effect size or business value.

🟨 **How It Works / Example**
If p=0.03, it means the observed difference would be rare if control and treatment were truly equal. You might decide the change is real. You still check if the improvement is large enough to matter.

🟪 **Quick Tip**
Probability of evidence.

---

## 613. Confidence Interval
🟦 **What is a confidence interval in A/B testing?**

🟩 **Definition**
A confidence interval gives a range of likely values for the true effect. It shows uncertainty around the estimated lift. It is often easier to interpret than only a p-value.

🟨 **How It Works / Example**
If lift is +1% with a 95% interval of [0.2%, 1.8%], the change is likely positive. If the interval crosses 0, the effect might be negative or positive. Teams use this to judge risk.

🟪 **Quick Tip**
Range of effect.

---

## 614. Effect Size
🟦 **What is effect size in experiments?**

🟩 **Definition**
Effect size is how big the change is, like "+0.5% conversion." It matters because tiny significant changes may not be worth shipping. Always consider both significance and effect size.

🟨 **How It Works / Example**
A huge website might see a statistically significant +0.1% lift that is still valuable. A small product might not care about +0.1%. Effect size connects experiment results to real impact.

🟪 **Quick Tip**
Magnitude of change.

---

## 615. Statistical Power
🟦 **What is power in A/B testing?**

🟩 **Definition**
Power is the chance your test will detect a real effect if it exists. Low power means you might miss true improvements. Power depends on sample size, variance, and effect size.

🟨 **How It Works / Example**
If you test with too few users, results are noisy and often inconclusive. Increasing sample size raises power. Teams do power analysis to decide how long to run experiments.

🟪 **Quick Tip**
Detection chance.

---

## 616. Sample Size Calculation
🟦 **What is sample size calculation for A/B tests?**

🟩 **Definition**
Sample size calculation estimates how many users you need to detect a target effect reliably. It uses expected baseline rate, desired lift, confidence, and power. It prevents running tests that are too small to learn from.

🟨 **How It Works / Example**
If your baseline conversion is 5% and you care about +0.5%, you compute needed users for each group. If the required number is huge, you might choose a different metric or bigger change. This planning saves time and prevents weak conclusions.

🟪 **Quick Tip**
How many users.

---

## 617. Type I Error
🟦 **What is a Type I error in A/B testing?**

🟩 **Definition**
A Type I error is a false positive, meaning you think B is better when it is not. It often relates to the significance threshold like 0.05. It can cause you to ship harmful or useless changes.

🟨 **How It Works / Example**
Random noise makes treatment look better, and you declare success. After shipping, the metric returns to normal or gets worse. Proper significance control and avoiding repeated peeking reduce Type I errors.

🟪 **Quick Tip**
False positive.

---

## 618. Type II Error
🟦 **What is a Type II error in A/B testing?**

🟩 **Definition**
A Type II error is a false negative, meaning you miss a real improvement. It often happens when tests have low power. It can make you reject good ideas.

🟨 **How It Works / Example**
A real +1% lift exists, but you ran the test too short. The result looks "not significant" so you stop. With more users, you would have detected the improvement.

🟪 **Quick Tip**
False negative.

---

## 619. Peeking
🟦 **What is "peeking" in A/B testing and why is it risky?**

🟩 **Definition**
Peeking is checking results many times and stopping early when you see significance. This increases false positives. It breaks standard statistical assumptions.

🟨 **How It Works / Example**
If you check every hour, eventually random noise may cross p<0.05. You stop and declare success even if there is no true effect. Sequential testing methods or fixed-duration tests help avoid this.

🟪 **Quick Tip**
Checking too often.

---

## 620. Sequential Testing
🟦 **What is sequential testing in experimentation?**

🟩 **Definition**
Sequential testing allows you to look at results during the test while controlling false positives. It adjusts rules for stopping early. It is useful when you want faster decisions safely.

🟨 **How It Works / Example**
You use methods like spending functions or Bayesian approaches. The system decides if evidence is strong enough to stop early. This avoids peeking problems while still allowing early stopping.

🟪 **Quick Tip**
Safe monitoring.

---

## 621. Bayesian Testing
🟦 **What is Bayesian A/B testing?**

🟩 **Definition**
Bayesian A/B testing estimates a probability distribution over the effect size. It answers questions like "What is the probability B is better than A?" It can be more intuitive for decision making.

🟨 **How It Works / Example**
You start with a prior belief about conversion rates. As data arrives, you update the belief to a posterior. Then you compute probabilities like P(lift > 0) and decide.

🟪 **Quick Tip**
Probability based.

---

## 622. Metric Noise
🟦 **What is metric noise and why does it affect experiments?**

🟩 **Definition**
Metric noise is random variation that makes measurements jump around. High noise makes it hard to detect real effects. It increases required sample size.

🟨 **How It Works / Example**
Daily traffic changes can make conversion vary even without any changes. Seasonal events can also add noise. Running longer or using variance reduction can help.

🟪 **Quick Tip**
Random jumps.

---

## 623. Variance Reduction
🟦 **What is variance reduction in experimentation?**

🟩 **Definition**
Variance reduction methods reduce noise so you can detect effects faster. They improve statistical power without needing as many users. Common methods include CUPED and stratification.

🟨 **How It Works / Example**
If users have different baseline behavior, you adjust using pre-experiment data. This removes some variation unrelated to treatment. Then the remaining difference reflects treatment more clearly.

🟪 **Quick Tip**
Less noise, more power.

---

## 624. Stratified Randomization
🟦 **What is stratified randomization?**

🟩 **Definition**
Stratified randomization ensures important user groups are balanced between control and treatment. It reduces imbalance that can bias results. Common strata include country, device, or user tenure.

🟨 **How It Works / Example**
You split users by mobile vs desktop, then randomize within each group. This prevents treatment from accidentally getting more mobile users. Balanced groups make comparisons fairer.

🟪 **Quick Tip**
Balanced groups.

---

## 625. Novelty Effect
🟦 **What is the novelty effect in A/B tests?**

🟩 **Definition**
The novelty effect happens when users react differently just because something is new. Early metrics may look better or worse than long-term behavior. This can mislead decisions if tests are too short.

🟨 **How It Works / Example**
A new UI might get more clicks at first because users explore it. After a week, engagement may drop back. Running long enough to see stable behavior helps.

🟪 **Quick Tip**
Newness factor.

---

## 626. Primacy Effect
🟦 **What is the primacy effect in user experiments?**

🟩 **Definition**
The primacy effect means first impressions strongly influence user behavior. Early experiences can shape long-term usage patterns. This matters for onboarding and recommendation changes.

🟨 **How It Works / Example**
If new users see lower quality recommendations on day 1, they might leave and never return. An A/B test focusing only on active users might miss this. Segmenting by new vs existing users helps detect it.

🟪 **Quick Tip**
First impression.

---

## 627. Interaction Effect
🟦 **What is an interaction effect in experiments?**

🟩 **Definition**
An interaction effect happens when the impact of a change depends on another factor, like user type or device. A change may help one segment and hurt another. Segment analysis is important.

🟨 **How It Works / Example**
A new ranking model might improve results for English queries but worsen for Spanish queries. Overall lift may look small. Segment breakdown reveals where it works and where it fails.

🟪 **Quick Tip**
Segment differences.

---

## 628. A/A Testing
🟦 **What is A/A testing and why do it?**

🟩 **Definition**
An A/A test splits users into two groups but serves the same experience to both. It checks the experiment system for bugs and false positives. Results should be near zero difference.

🟨 **How It Works / Example**
You run A/A before a major testing platform change. If you see large differences, randomization or logging may be broken. Fixing this prevents invalid A/B conclusions.

🟪 **Quick Tip**
System check.

---

## 629. Holdout Group
🟦 **What is a holdout group in ML experimentation?**

🟩 **Definition**
A holdout group is a set of users kept on an older baseline for a long time. It helps measure long-term impact and drift. It is useful for recommendation and ads systems.

🟨 **How It Works / Example**
You keep 1% of users on the old recommender for months. You compare their retention and revenue to updated users. This shows if continuous changes are truly improving long-term outcomes.

🟪 **Quick Tip**
Long-term test.

---

## 630. Offline vs Online Evaluation
🟦 **What is offline evaluation vs online evaluation?**

🟩 **Definition**
Offline evaluation tests models using historical data and metrics like accuracy. Online evaluation tests with real users in production, often using A/B tests. Both are needed because they can disagree.

🟨 **How It Works / Example**
A ranking model improves offline NDCG but online user time drops. Offline data may not reflect user satisfaction or feedback loops. Online tests reveal real impact.

🟪 **Quick Tip**
Lab vs Reality.

---

## 631. Experiment Hypothesis
🟦 **What is an experiment hypothesis?**

🟩 **Definition**
A hypothesis is a clear statement of what change you expect and why. It defines the metric you think will improve. Writing it prevents random testing without learning.

🟨 **How It Works / Example**
Hypothesis: "Adding personalized embeddings will increase search click-through rate by 2%." You run an A/B test to check. After results, you confirm or reject the hypothesis and document what you learned.

🟪 **Quick Tip**
Expected outcome.

---

## 632. Design Document
🟦 **What is an experiment design document?**

🟩 **Definition**
An experiment design document describes what you will test, metrics, duration, risks, and rollout plan. It aligns teams and reduces mistakes. It is common in ML product teams.

🟨 **How It Works / Example**
You write: control vs treatment details, primary and guardrail metrics, and sample size plan. You also list expected failure modes and rollback triggers. This makes reviews and approvals easier.

🟪 **Quick Tip**
Experiment plan.

---

## 633. Rollout Strategy
🟦 **What is a rollout strategy after a successful A/B test?**

🟩 **Definition**
A rollout strategy gradually increases traffic to the winning variant. It reduces risk and allows monitoring for issues at scale. Rollouts often include canary and staged steps.

🟨 **How It Works / Example**
After success, you go from 10% -> 25% -> 50% -> 100% traffic. You watch guardrail metrics at each stage. If something breaks at 50%, you stop and investigate.

🟪 **Quick Tip**
Gradual release.

---

## 634. Canary Testing
🟦 **What is canary testing in ML deployment?**

🟩 **Definition**
Canary testing sends a small portion of traffic to a new model in production. It checks stability and guardrails before full rollout. It is a safety step in MLOps.

🟨 **How It Works / Example**
You deploy the new model to 1% of users. You monitor latency, error rate, and key product metrics. If all looks good, you expand traffic gradually.

🟪 **Quick Tip**
Safety canary.

---

## 635. Shadow Deployment
🟦 **What is a shadow deployment (shadow mode) experiment?**

🟩 **Definition**
Shadow deployment runs a new model in parallel without affecting user experience. It collects predictions and metrics safely. It is useful for comparing models before A/B testing.

🟨 **How It Works / Example**
Users still see control results, but the new model also generates outputs in the background. You compare outputs and latency. If it looks good, you move to a true online test.

🟪 **Quick Tip**
Silent test.

---

## 636. Logging
🟦 **What is logging and why is it critical for experiments?**

🟩 **Definition**
Logging records user assignments, inputs, outputs, and metrics. Without correct logs, you cannot trust experiment results. Good logging supports debugging and auditing.

🟨 **How It Works / Example**
You log user_id, experiment group, timestamp, and outcome like "clicked." Later you compute conversion by group. If assignments are missing, your analysis becomes biased.

🟪 **Quick Tip**
Recording data.

---

## 637. Experiment Unit
🟦 **What is an experiment unit and why does it matter?**

🟩 **Definition**
The experiment unit is what you randomize, like user, session, or device. Choosing the right unit prevents contamination. Wrong units can bias results.

🟨 **How It Works / Example**
If you randomize by session, one user may see both control and treatment, which can confuse behavior. Randomizing by user keeps experience consistent. For ads, you might randomize by account instead.

🟪 **Quick Tip**
What to split.

---

## 638. Contamination
🟦 **What is contamination in A/B testing?**

🟩 **Definition**
Contamination happens when users are exposed to both control and treatment effects. It reduces the ability to measure true differences. It can happen due to shared caches or social sharing.

🟨 **How It Works / Example**
If a user sees the treatment in one browser and control in another, behavior mixes. Or if results are cached across users, control users might get treatment content. Fixing caching and consistent user assignment helps.

🟪 **Quick Tip**
Mixed signals.

---

## 639. Simpson's Paradox
🟦 **What is the Simpson's paradox risk in experiment analysis?**

🟩 **Definition**
Simpson's paradox happens when a trend appears in aggregated data but reverses in segments. It can lead to wrong conclusions if you only look at overall averages. Segment analysis helps avoid it.

🟨 **How It Works / Example**
Overall conversion improves, but both mobile and desktop conversions drop when analyzed separately. The mix of traffic changed between groups. You must check segments to understand the real effect.

🟪 **Quick Tip**
Segment reversal.

---

## 640. Multiple Testing
🟦 **What is multiple testing and why is it a problem?**

🟩 **Definition**
Multiple testing means checking many metrics or running many experiments and looking for significant results. This increases false positives. You need corrections or careful planning.

🟨 **How It Works / Example**
If you test 20 metrics, one may look significant by chance at 0.05. You can pre-register primary metrics or use corrections like Bonferroni. This keeps conclusions more reliable.

🟪 **Quick Tip**
Too many checks.

---

## 641. Duration
🟦 **What is an experiment "duration" decision based on?**

🟩 **Definition**
Duration depends on needed sample size, traffic rate, and how quickly metrics stabilize. Some metrics require full weekly cycles to capture patterns. Ending too early can mislead results.

🟨 **How It Works / Example**
A shopping site might need at least 1–2 weeks to capture weekend behavior. A low-traffic product may need longer to reach required users. Teams plan duration using power analysis and seasonality.

🟪 **Quick Tip**
How long to run.

---

## 642. Seasonality
🟦 **What is seasonality in A/B testing?**

🟩 **Definition**
Seasonality means metrics change over time due to days, weeks, or holidays. It can hide or exaggerate experiment effects. You must account for it in design and analysis.

🟨 **How It Works / Example**
Conversion may rise on weekends and drop midweek. If your test runs only weekdays, results may not generalize. Running over full cycles helps reduce seasonality bias.

🟪 **Quick Tip**
Time cycles.

---

## 643. Funnel Metric
🟦 **What is a funnel metric in product experiments?**

🟩 **Definition**
A funnel metric tracks a step-by-step user journey, like visit -> add to cart -> purchase. It helps find where changes improve or hurt behavior. Funnels are common in growth experiments.

🟨 **How It Works / Example**
A new recommendation model might increase "add to cart" but not "purchase." Funnel metrics show where the drop happens. Then you investigate the checkout experience or product relevance.

🟪 **Quick Tip**
Step-by-step.

---

## 644. Uplift Model
🟦 **What is an uplift model and how is it related to experimentation?**

🟩 **Definition**
An uplift model predicts which users will benefit most from a treatment. It focuses on the causal effect of an action. It is often used in marketing and personalization.

🟨 **How It Works / Example**
Instead of recommending a coupon to everyone, you predict who will actually buy because of the coupon. You train using treatment/control outcomes. Then you target only users with high predicted uplift.

🟪 **Quick Tip**
Predicting impact.

---

## 645. Causal Inference
🟦 **What is a causal inference goal in A/B testing?**

🟩 **Definition**
Causal inference aims to measure what the change caused, not just correlation. A/B tests are strong for causal inference because randomization removes many confounders. The goal is "did treatment cause the lift?"

🟨 **How It Works / Example**
You randomize users, so groups are similar on average. Any consistent difference in outcomes is likely caused by the treatment. This is why A/B tests are trusted for product decisions.

🟪 **Quick Tip**
Finding cause.

---

## 646. Recommender Pitfall
🟦 **What is a common A/B testing pitfall for recommender systems?**

🟩 **Definition**
A common pitfall is feedback loops where the model changes what data it sees. This can make short-term results misleading. Recommenders can also shift content diversity in ways that matter long-term.

🟨 **How It Works / Example**
A model that shows only popular items can boost short-term clicks. Over time, users may get bored and retention drops. Holdout groups and long-term metrics help detect this.

🟪 **Quick Tip**
Feedback loops.

---

## 647. Chatbot Pitfall
🟦 **What is a common experimentation pitfall for LLM chatbots?**

🟩 **Definition**
A common pitfall is measuring only easy metrics like response length instead of real success. Another is ignoring safety and correctness. LLM behavior is multi-dimensional, so experiments need careful metrics.

🟨 **How It Works / Example**
A new prompt increases "messages per session" because the bot is more verbose, not more helpful. You should also track resolution rate and user ratings. Safety guardrails prevent shipping risky behavior.

🟪 **Quick Tip**
Ignoring quality.

---

## 648. Experiment Reset
🟦 **What is an "experiment reset" and when do you do it?**

🟩 **Definition**
An experiment reset restarts the test because the setup was wrong or contaminated. It prevents drawing conclusions from bad data. It is better to reset than to trust flawed results.

🟨 **How It Works / Example**
You discover a logging bug that misassigned users. You stop the experiment and fix the bug. Then you start a fresh test with clean randomization and correct logs.

🟪 **Quick Tip**
Starting over.

---

## 649. Post-Experiment Analysis
🟦 **What is post-experiment analysis?**

🟩 **Definition**
Post-experiment analysis explains results, checks segments, and documents learnings. It helps decide rollout and guides future experiments. Good analysis includes both wins and failures.

🟨 **How It Works / Example**
You review primary metric lift and guardrails. You break down results by region and device. Then you write a summary: what changed, what happened, and what you will do next.

🟪 **Quick Tip**
Learning results.

---

## 650. Experimentation Culture
🟦 **What is an experimentation culture in ML teams?**

🟩 **Definition**
Experimentation culture means teams regularly test ideas, measure outcomes, and learn quickly. It encourages data-driven decisions and reduces opinion battles. It requires good tooling and discipline.

🟨 **How It Works / Example**
Teams have standard metrics, dashboards, and review processes. They run A/B tests for model changes and document results. Over time, the product improves through repeated measured iterations.

🟪 **Quick Tip**
Continuous testing.
