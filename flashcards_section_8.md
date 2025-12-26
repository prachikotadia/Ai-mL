# Section 8: Fine-Tuning, LoRA, PEFT, RLHF

## 351. Fine-Tuning
🟦 **What is fine-tuning for large language models?**

🟩 **Definition**
Fine-tuning is training a pretrained model a bit more on your own dataset. It changes the model's behavior for a specific task or style. It usually needs much less data than pretraining.

🟨 **How It Works / Example**
You collect prompt -> ideal answer pairs for your domain. Then you train the model so it predicts those answers. After fine-tuning, it responds more accurately for your use case.

🟪 **Quick Tip**
Tailoring the model.

---

## 352. Supervised Fine-Tuning (SFT)
🟦 **What is supervised fine-tuning (SFT)?**

🟩 **Definition**
SFT is fine-tuning using labeled input-output examples. The model learns to produce the target response for a given prompt. It is often the first step in building a helpful chat model.

🟨 **How It Works / Example**
You give the model many examples like "User: Write an apology email -> Assistant: (good email)." Training teaches the model to copy that helpful style. Then it follows similar instructions on new prompts.

🟪 **Quick Tip**
Learning from examples.

---

## 353. Instruction Tuning
🟦 **What is instruction tuning?**

🟩 **Definition**
Instruction tuning is training a model on many instruction-response examples. It improves the model's ability to follow commands and formats. It makes the model feel more like an assistant.

🟨 **How It Works / Example**
You train on tasks like summarizing, classifying, and rewriting text. The model learns patterns like "When asked for bullets, return bullets." This improves consistency across many tasks.

🟪 **Quick Tip**
Following orders.

---

## 354. Domain Adaptation
🟦 **What is domain adaptation in fine-tuning?**

🟩 **Definition**
Domain adaptation is fine-tuning a model so it works better in a specific area like legal, medical, or finance. It helps the model learn domain terms and common writing style. It does not automatically guarantee correctness.

🟨 **How It Works / Example**
You fine-tune on internal policy docs and support chats. The model learns your product names and procedures. Then it answers customer questions using your company language.

🟪 **Quick Tip**
Specializing in a field.

---

## 355. PEFT
🟦 **What is parameter-efficient fine-tuning (PEFT)?**

🟩 **Definition**
PEFT is fine-tuning where you update only a small number of parameters. It reduces compute and storage costs. It often gives strong performance without changing the full model.

🟨 **How It Works / Example**
Instead of updating all weights, you add small trainable modules. You train only those modules while keeping the base model fixed. This lets you maintain multiple task versions cheaply.

🟪 **Quick Tip**
Efficient tuning.

---

## 356. LoRA
🟦 **What is LoRA in LLM fine-tuning?**

🟩 **Definition**
LoRA (Low-Rank Adaptation) is a PEFT method that adds small low-rank weight updates. The base model stays frozen. Only the small LoRA matrices are trained.

🟨 **How It Works / Example**
You insert LoRA layers into attention projections like Q and V. Training updates only those low-rank adapters. You can store many LoRA adapters for different tasks without duplicating the full model.

🟪 **Quick Tip**
Low-rank updates.

---

## 357. LoRA Cost
🟦 **How does LoRA reduce training cost?**

🟩 **Definition**
LoRA trains fewer parameters, so it uses less memory and compute. It also reduces storage because you save only adapter weights. This makes fine-tuning large models more affordable.

🟨 **How It Works / Example**
A full model might be tens of GB, but a LoRA adapter might be a few hundred MB or less. You train and save the adapter. At inference, you load the base model plus the adapter to get the tuned behavior.

🟪 **Quick Tip**
Saving memory and compute.

---

## 358. LoRA Rank
🟦 **What is the "rank" in LoRA?**

🟩 **Definition**
Rank controls the size of the low-rank matrices used by LoRA. Higher rank increases adapter capacity and may improve quality. But higher rank also increases compute and memory.

🟨 **How It Works / Example**
If rank is 8, the adapter is small and cheap but may be limited. If rank is 64, it can learn more changes but costs more. Teams tune rank to balance quality and cost.

🟪 **Quick Tip**
Adapter size control.

---

## 359. QLoRA
🟦 **What is QLoRA?**

🟩 **Definition**
QLoRA combines LoRA with quantization of the base model during training. This reduces GPU memory usage a lot. It helps fine-tune large models on smaller hardware.

🟨 **How It Works / Example**
You load the base model in 4-bit precision to save memory. Then you train LoRA adapters in higher precision for stability. This lets you fine-tune models that would otherwise not fit in memory.

🟪 **Quick Tip**
Quantized LoRA.

---

## 360. Adapter Tuning
🟦 **What is adapter tuning?**

🟩 **Definition**
Adapter tuning adds small new layers (adapters) inside the model and trains only those. The base model is frozen. It is another common PEFT approach.

🟨 **How It Works / Example**
You place adapters between transformer layers. During training, only adapter parameters change. You can switch adapters to change the model's behavior per task.

🟪 **Quick Tip**
Plug-in layers.

---

## 361. Prompt Tuning
🟦 **What is prompt tuning?**

🟩 **Definition**
Prompt tuning learns a small set of trainable "soft prompt" vectors. It keeps the model weights frozen. It is a lightweight way to adapt a model to a task.

🟨 **How It Works / Example**
Instead of writing text prompts, you learn prompt embeddings that get prepended to input tokens. Training updates only these embeddings. This can work well for some tasks with minimal compute.

🟪 **Quick Tip**
Soft prompts.

---

## 362. Prefix Tuning
🟦 **What is prefix tuning?**

🟩 **Definition**
Prefix tuning adds trainable vectors to the attention mechanism as extra context. The base model stays frozen. It is similar to prompt tuning but applied more directly in attention.

🟨 **How It Works / Example**
You learn a small "prefix" that acts like extra key/value states. The model attends to this prefix during generation. This can guide the model toward task-specific behavior.

🟪 **Quick Tip**
Attention prefixes.

---

## 363. Full vs PEFT
🟦 **What is full fine-tuning vs PEFT?**

🟩 **Definition**
Full fine-tuning updates most or all model weights. PEFT updates only a small part like adapters or LoRA. Full fine-tuning can be stronger but is usually more expensive.

🟨 **How It Works / Example**
If you have lots of data and compute, full fine-tuning may give the best results. If you need cheap customization for many tasks, PEFT is often better. Many teams start with PEFT and only use full fine-tuning when needed.

🟪 **Quick Tip**
All weights vs subset.

---

## 364. Catastrophic Forgetting
🟦 **What is catastrophic forgetting in fine-tuning?**

🟩 **Definition**
Catastrophic forgetting is when fine-tuning makes the model lose older skills. It happens when new training data is narrow and updates are too strong. It is a common risk in domain-specific tuning.

🟨 **How It Works / Example**
If you fine-tune heavily on legal text, the model may get worse at casual chat or coding. Mixing general data with domain data can help. Using PEFT or smaller learning rates can also reduce forgetting.

🟪 **Quick Tip**
Learning new, losing old.

---

## 365. Overfitting
🟦 **What is overfitting in LLM fine-tuning?**

🟩 **Definition**
Overfitting is when the model memorizes training examples instead of generalizing. It performs well on training data but worse on new prompts. It happens more when datasets are small or repetitive.

🟨 **How It Works / Example**
If you train on 500 Q&A pairs, the model might repeat them too closely. Validation performance may not improve or may degrade. Regularization, early stopping, and more diverse data help.

🟪 **Quick Tip**
Memorizing, not learning.

---

## 366. Data Leakage
🟦 **What is data leakage in fine-tuning?**

🟩 **Definition**
Data leakage is when training data contains information that should not be used, like test answers or private data. It can inflate evaluation results and create compliance risks. Preventing leakage is critical in real products.

🟨 **How It Works / Example**
If your dataset includes future tickets that appear in evaluation, results look too good. Or if it includes customer secrets, the model might repeat them. Use strict splits and privacy filters to avoid this.

🟪 **Quick Tip**
Cheating with answers.

---

## 367. Validation Set
🟦 **What is a validation set in fine-tuning?**

🟩 **Definition**
A validation set is a held-out dataset used to tune training and detect overfitting. It is not used for gradient updates. It helps choose checkpoints and hyperparameters.

🟨 **How It Works / Example**
You fine-tune on training prompts and check performance on validation prompts every few steps. If validation loss stops improving, you stop training or reduce learning rate. This helps pick a better final model.

🟪 **Quick Tip**
Testing during training.

---

## 368. Early Stopping
🟦 **What is early stopping in fine-tuning?**

🟩 **Definition**
Early stopping stops training when validation performance stops improving. It helps prevent overfitting. It also saves compute.

🟨 **How It Works / Example**
If validation loss improves until step 2,000 and then gets worse, you stop around the best checkpoint. You keep the best saved model. This usually generalizes better than training longer.

🟪 **Quick Tip**
Quitting while ahead.

---

## 369. Learning Rate
🟦 **What is learning rate selection for fine-tuning?**

🟩 **Definition**
Learning rate controls how strongly weights change during fine-tuning. Too high can destroy pretrained knowledge; too low may not adapt enough. Fine-tuning often uses smaller learning rates than training from scratch.

🟨 **How It Works / Example**
You might try a small LR like 1e-5 for full fine-tuning. For LoRA, you may use a slightly higher LR for adapters. You compare validation results to pick the best setting.

🟪 **Quick Tip**
Speed of change.

---

## 370. Warmup Schedule
🟦 **What is a warmup schedule in fine-tuning?**

🟩 **Definition**
Warmup slowly increases learning rate at the start of training. It helps prevent unstable early updates. It is common in transformer training and fine-tuning.

🟨 **How It Works / Example**
For the first 500 steps, LR grows from near 0 to the target LR. This avoids large early jumps. After warmup, LR typically decays for stable convergence.

🟪 **Quick Tip**
Gentle start.

---

## 371. RLHF
🟦 **What is RLHF and why is it used after SFT?**

🟩 **Definition**
RLHF trains a model using human preference feedback, not just labeled answers. It improves helpfulness, safety, and user satisfaction beyond SFT. It is often used to refine chat models.

🟨 **How It Works / Example**
After SFT, the model may still be overly verbose or unsafe. Humans compare two answers and pick the better one. The model learns to produce answers that humans prefer.

🟪 **Quick Tip**
Human preference training.

---

## 372. Preference Data
🟦 **What is preference data in RLHF?**

🟩 **Definition**
Preference data is where humans rank or choose the better of two (or more) model outputs. It captures quality judgments that are hard to label directly. It is used to train a reward model.

🟨 **How It Works / Example**
Given a prompt, the model generates two answers. A reviewer picks which one is more correct and helpful. These comparisons become training data for preference learning.

🟪 **Quick Tip**
A vs B.

---

## 373. Reward Modeling
🟦 **What is reward modeling in RLHF?**

🟩 **Definition**
Reward modeling trains a model to score responses based on human preferences. The score represents how "good" a response is. The LLM is then optimized to increase this score.

🟨 **How It Works / Example**
You train a reward model so preferred answers get higher scores than rejected ones. Then you run RL to adjust the LLM to get higher reward. This pushes the LLM toward better behavior.

🟪 **Quick Tip**
Scoring quality.

---

## 374. PPO
🟦 **What is PPO in RLHF?**

🟩 **Definition**
PPO (Proximal Policy Optimization) is a reinforcement learning algorithm often used in RLHF. It updates the model carefully to avoid huge changes. This helps keep the model stable and safe.

🟨 **How It Works / Example**
The reward model scores candidate outputs. PPO updates the LLM to increase reward while limiting how far it moves from the original policy. This reduces the risk of the model drifting into bad behavior.

🟪 **Quick Tip**
Safe RL updates.

---

## 375. KL Penalty
🟦 **What is the "KL penalty" in RLHF?**

🟩 **Definition**
KL penalty discourages the RLHF-updated model from drifting too far from the base or SFT model. It helps keep language quality and prevents extreme behavior. It is a key stability control in RLHF.

🟨 **How It Works / Example**
If the RL model tries to change outputs drastically to game the reward, KL penalty adds a cost. The optimizer balances reward gain with staying close to the reference model. This often prevents weird or repetitive outputs.

🟪 **Quick Tip**
Don't drift too far.

---

## 376. Reward Hacking
🟦 **What is reward hacking in RLHF?**

🟩 **Definition**
Reward hacking is when the model finds ways to get high reward without actually being correct or helpful. It exploits weaknesses in the reward model. This is a common risk in reinforcement learning.

🟨 **How It Works / Example**
If the reward model prefers confident answers, the LLM may become overly confident even when wrong. It might write long answers that "sound helpful" but are incorrect. Better reward models and audits help reduce this.

🟪 **Quick Tip**
Gaming the system.

---

## 377. DPO
🟦 **What is DPO (Direct Preference Optimization)?**

🟩 **Definition**
DPO is a method that trains a model directly from preference pairs without running full RL like PPO. It is simpler and often more stable. It's a popular alternative to classic RLHF.

🟨 **How It Works / Example**
You use pairs of (chosen, rejected) responses for the same prompt. DPO updates the model to increase probability of chosen and decrease probability of rejected. This can improve alignment with fewer moving parts than PPO.

🟪 **Quick Tip**
Direct preference learning.

---

## 378. DPO vs PPO
🟦 **How does DPO differ from PPO-based RLHF?**

🟩 **Definition**
DPO uses a direct supervised-style objective on preference pairs. PPO uses a reward model plus reinforcement learning updates. DPO is often easier to train and debug.

🟨 **How It Works / Example**
With PPO, you score outputs with a reward model and do RL updates with KL control. With DPO, you skip the reward model and learn from chosen vs rejected examples directly. Many teams choose DPO when they want simpler pipelines.

🟪 **Quick Tip**
Simpler vs standard RL.

---

## 379. RLAIF
🟦 **What is RLAIF (Reinforcement Learning from AI Feedback)?**

🟩 **Definition**
RLAIF uses AI-generated preferences instead of only human labels. It can scale feedback cheaply. It still requires careful validation to avoid copying AI mistakes.

🟨 **How It Works / Example**
A stronger model or evaluator model ranks outputs. You treat those rankings like preference data. Then you tune your model similarly to RLHF or DPO.

🟪 **Quick Tip**
AI feedback.

---

## 380. Alignment Tax
🟦 **What is "alignment tax" in LLM training?**

🟩 **Definition**
Alignment tax is a performance or capability cost that can happen when you push a model to be safer or more compliant. Some alignment methods can reduce creativity or raw task performance. Teams try to minimize this tradeoff.

🟨 **How It Works / Example**
A safety-tuned model might refuse more often or give more cautious answers. This can reduce helpfulness for borderline tasks. Better data and careful tuning aim to keep safety without losing too much quality.

🟪 **Quick Tip**
Safety vs performance.

---

## 381. Policy Model
🟦 **What is a "policy model" in RLHF?**

🟩 **Definition**
The policy model is the LLM being optimized during RLHF. It produces responses that are scored by the reward model. Training updates it to produce higher-scoring outputs.

🟨 **How It Works / Example**
You sample responses from the policy model for a set of prompts. The reward model scores them. The RL algorithm updates the policy model to increase expected reward.

🟪 **Quick Tip**
The model being trained.

---

## 382. Reference Model
🟦 **What is a "reference model" in RLHF?**

🟩 **Definition**
A reference model is a fixed model used to measure how far the policy model has changed. It helps compute KL penalty. Often it is the SFT model.

🟨 **How It Works / Example**
During training, you compare the new model's output probabilities to the reference model's. If they diverge too much, KL penalty increases. This keeps the new model from drifting wildly.

🟪 **Quick Tip**
The baseline.

---

## 383. Reward Shaping
🟦 **What is "reward shaping" in preference training?**

🟩 **Definition**
Reward shaping changes the reward signal to encourage certain behaviors. It can make training easier or more stable. But it can also introduce bias if done poorly.

🟨 **How It Works / Example**
You might add a small penalty for very long answers to reduce verbosity. Or reward citing sources in RAG. The model learns these preferences through the shaped reward signal.

🟪 **Quick Tip**
Nudging the score.

---

## 384. Safety Dataset
🟦 **What is a "safety dataset" for LLM alignment?**

🟩 **Definition**
A safety dataset contains examples of unsafe requests and correct refusals or safe alternatives. It trains the model to avoid harmful outputs. It is used in SFT and preference training.

🟨 **How It Works / Example**
You include prompts asking for illegal instructions and label safe refusal responses. The model learns to refuse and redirect. This reduces harmful behavior in production.

🟪 **Quick Tip**
Examples of what not to do.

---

## 385. Red Teaming
🟦 **What is "red teaming" for LLMs?**

🟩 **Definition**
Red teaming is testing a model to find failures, unsafe behavior, or vulnerabilities. It is done by people trying hard to break the system. It helps improve safety before release.

🟨 **How It Works / Example**
Testers try prompt injection, jailbreaks, and tricky edge cases. They record failures and create new training or guardrails. This makes the model safer over time.

🟪 **Quick Tip**
Stress testing.

---

## 386. Evaluation Set Contamination
🟦 **What is "evaluation set contamination" in LLM fine-tuning?**

🟩 **Definition**
Contamination happens when evaluation data is included in training data. It makes results look better than they truly are. It can happen easily when datasets are collected from the web.

🟨 **How It Works / Example**
If your benchmark questions appear in your fine-tuning corpus, the model may memorize them. Your evaluation score becomes inflated. Proper dataset filtering and strict splits help prevent this.

🟪 **Quick Tip**
Answers in the book.

---

## 387. Prompt Overfitting
🟦 **What is "prompt overfitting" in LLM tuning?**

🟩 **Definition**
Prompt overfitting is when a model becomes too specialized to specific prompt wording or templates. It performs well on seen formats but poorly on new ones. It often happens when training data is not diverse.

🟨 **How It Works / Example**
If all training prompts start with "Task:", the model may rely on that pattern. When a real user asks differently, performance drops. Using varied prompts and paraphrases improves robustness.

🟪 **Quick Tip**
Memorizing the wording.

---

## 388. Format Adherence
🟦 **What is "format adherence" and how do you train it?**

🟩 **Definition**
Format adherence means the model follows required output structure, like JSON or bullet points. It is important for tool-using systems. Training it requires many clear examples and strict evaluation.

🟨 **How It Works / Example**
You fine-tune on examples where the correct answer is valid JSON only. You also penalize extra text. Over time, the model learns to output structured responses consistently.

🟪 **Quick Tip**
Sticking to the structure.

---

## 389. Adapter Merging
🟦 **What is "adapter merging" in PEFT?**

🟩 **Definition**
Adapter merging combines adapter weights into the base model weights. It can simplify deployment by removing the need to load adapters separately. It must be done carefully to keep quality.

🟨 **How It Works / Example**
After training a LoRA adapter, you can merge it into the main weight matrices. Then inference uses a single merged model. This can reduce runtime complexity but increases storage if you need many variants.

🟪 **Quick Tip**
Baking in the changes.

---

## 390. Hot Swapping
🟦 **What is "hot swapping adapters" in production?**

🟩 **Definition**
Hot swapping means switching adapters at runtime without reloading the full base model. It enables multiple behaviors using one base model. This is useful for multi-tenant or multi-task systems.

🟨 **How It Works / Example**
A server keeps one base LLM in GPU memory. For customer A, it loads adapter A; for customer B, adapter B. This allows customization with lower cost and faster switching.

🟪 **Quick Tip**
Changing masks on the fly.

---

## 391. LoRA Targeting
🟦 **What is "LoRA targeting" (which layers to adapt)?**

🟩 **Definition**
LoRA targeting means choosing which modules get LoRA adapters, like attention Q/K/V or MLP layers. Target choice affects quality and cost. Common targets are attention projections.

🟨 **How It Works / Example**
If you apply LoRA only to Q and V projections, you change how attention reads and writes information. This can be enough for many tasks. Adding LoRA to more layers increases capacity but costs more.

🟪 **Quick Tip**
Where to apply updates.

---

## 392. Rank-Stabilized LoRA
🟦 **What is "rank-stabilized LoRA" conceptually?**

🟩 **Definition**
It is an idea to make LoRA updates more stable across different ranks. The goal is to avoid overly large updates when changing rank. Stability can make training easier to tune.

🟨 **How It Works / Example**
If increasing rank changes update scale too much, training may become unstable. A stabilized approach adjusts scaling so behavior stays consistent. This helps compare experiments fairly across ranks.

🟪 **Quick Tip**
Consistent updates.

---

## 393. Multi-Task PEFT
🟦 **What is "PEFT for multi-task learning"?**

🟩 **Definition**
PEFT supports multi-task learning by keeping one base model and many small task adapters. Each adapter specializes in a task without rewriting the base. This is cheaper than storing many full fine-tuned models.

🟨 **How It Works / Example**
You train one adapter for summarization and another for customer support. At inference, you load the adapter based on the request type. This gives specialized behavior with low storage overhead.

🟪 **Quick Tip**
Many skills, one base.

---

## 394. Reward Model Overfitting
🟦 **What is "reward model overfitting" in RLHF?**

🟩 **Definition**
Reward model overfitting is when the reward model learns preference patterns that do not generalize. The policy model then optimizes to those narrow patterns. This can cause weird behavior or reward hacking.

🟨 **How It Works / Example**
If the reward model prefers long answers, the policy may become too verbose. It gets high reward but users may dislike it. Better preference data, regularization, and audits help prevent this.

🟪 **Quick Tip**
Learning bad preferences.

---

## 395. Preference Drift
🟦 **What is "preference drift" over time in LLM products?**

🟩 **Definition**
Preference drift is when user expectations change, but the model is still trained on old preference data. This can reduce satisfaction. Products need continuous evaluation and updates.

🟨 **How It Works / Example**
Users may start preferring shorter answers or stricter safety. If your RLHF data is from last year, it may not reflect current needs. Periodic re-labeling and re-tuning helps keep alignment current.

🟪 **Quick Tip**
Changing tastes.

---

## 396. Evaluation Type
🟦 **What is "offline vs online evaluation" for tuned LLMs?**

🟩 **Definition**
Offline evaluation uses test datasets and benchmarks. Online evaluation measures behavior with real users in production, often with A/B tests. Both are needed because offline scores can miss real-world issues.

🟨 **How It Works / Example**
Offline you check accuracy on an internal QA set. Online you measure user satisfaction, time-to-resolution, and refusal rate. Sometimes a model that scores higher offline performs worse online due to style or latency.

🟪 **Quick Tip**
Lab vs real world.

---

## 397. Safety vs Helpfulness
🟦 **What is "safety vs helpfulness tradeoff" in RLHF?**

🟩 **Definition**
This tradeoff happens when making a model safer can also reduce how much it answers. Over-refusing can frustrate users. Under-refusing can be risky.

🟨 **How It Works / Example**
A model might refuse a harmless chemistry question because it looks like a weapon topic. Better policies and training data help it distinguish safe from unsafe. Teams tune thresholds and prompts to balance both goals.

🟪 **Quick Tip**
The alignment balance.

---

## 398. Alignment Regression
🟦 **What is "alignment regression" after fine-tuning?**

🟩 **Definition**
Alignment regression is when a tuned model becomes less safe or less instruction-following than before. It can happen if fine-tuning data conflicts with safety training. It needs careful evaluation and guardrails.

🟨 **How It Works / Example**
If you fine-tune on raw forum text, the model may learn toxic styles. Even if it becomes better at domain terms, safety can drop. Mixing safety data and running safety evals helps prevent this.

🟪 **Quick Tip**
Safety backsliding.

---

## 399. Dataset Curation
🟦 **What is "dataset curation" for LLM fine-tuning?**

🟩 **Definition**
Dataset curation is selecting, cleaning, and organizing training examples. Good curation improves quality more than just adding more data. It also reduces privacy and safety risks.

🟨 **How It Works / Example**
You remove duplicates, filter low-quality answers, and ensure consistent format. You also remove sensitive info like personal data. The final dataset gives clearer learning signals to the model.

🟪 **Quick Tip**
Quality over quantity.

---

## 400. Training Recipe
🟦 **What is "training recipe" in LLM fine-tuning?**

🟩 **Definition**
A training recipe is the full set of choices for training, like optimizer, learning rate, batch size, and schedules. Small recipe differences can change quality a lot. A good recipe makes results repeatable.

🟨 **How It Works / Example**
A recipe might include AdamW, warmup, cosine decay, LoRA rank 16, and gradient clipping. You run experiments and track metrics. Once a recipe works, you reuse it for similar models and tasks.

🟪 **Quick Tip**
The chef's secret.
