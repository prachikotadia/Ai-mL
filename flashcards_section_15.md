# Section 15: Scaling, Performance & System Design

## 701. ML System Design
🟦 **What is ML system design?**

🟩 **Definition**
ML system design is planning how to build a full ML product, not just a model. It includes data, training, serving, monitoring, and user experience. The goal is a system that works reliably at scale.

🟨 **How It Works / Example**
For a recommendation system, you design how to collect clicks, train the model, and serve results quickly. You also design fallback behavior if the model is down. Monitoring ensures quality stays good after launch.

🟪 **Quick Tip**
Designing the full loop.

---

## 702. Scaling Importance
🟦 **Why is scaling important in ML system design?**

🟩 **Definition**
Scaling matters because real products handle many users and requests. As load grows, latency, cost, and reliability can break. Good design keeps performance stable as usage increases.

🟨 **How It Works / Example**
A chatbot that works for 100 users may fail for 100,000 users due to GPU limits. You add caching, batching, and autoscaling. This keeps response time acceptable during peak traffic.

🟪 **Quick Tip**
Handling growth.

---

## 703. Serving Latency
🟦 **What is latency in ML serving?**

🟩 **Definition**
Latency is how long it takes to return a prediction or response. Users notice high latency quickly. Many ML systems have strict latency targets.

🟨 **How It Works / Example**
A search ranking model may need p95 latency under 50ms. If inference takes too long, the page loads slowly. Teams optimize models and infrastructure to reduce that time.

🟪 **Quick Tip**
Wait time.

---

## 704. Serving Throughput
🟦 **What is throughput in ML serving?**

🟩 **Definition**
Throughput is how many requests a system can handle per second. High throughput is needed for large traffic systems. It depends on model speed and hardware capacity.

🟨 **How It Works / Example**
If your service must handle 10,000 requests per second, one server may not be enough. You run multiple replicas behind a load balancer. Batching and optimized runtimes also increase throughput.

🟪 **Quick Tip**
Volume of requests.

---

## 705. p95 Latency
🟦 **What is p95 latency and why do teams track it?**

🟩 **Definition**
p95 latency is the time under which 95% of requests complete. It shows how slow the "slow requests" are, not just the average. It matters because users feel tail latency.

🟨 **How It Works / Example**
Average latency might be 20ms but p95 could be 200ms due to spikes. Those spikes hurt user experience. Teams optimize tail latency using caching, better load balancing, and removing bottlenecks.

🟪 **Quick Tip**
Tail latency.

---

## 706. SLO (Service Level Objective)
🟦 **What is a service-level objective (SLO) for ML systems?**

🟩 **Definition**
An SLO is a target for system reliability and performance, like uptime and latency. It helps teams measure whether the service meets expectations. SLOs guide engineering priorities.

🟨 **How It Works / Example**
An ML endpoint might have an SLO of 99.9% uptime and p95 latency under 100ms. Dashboards track these metrics continuously. If the service breaks the SLO, it triggers incident response.

🟪 **Quick Tip**
Reliability target.

---

## 707. Cost-Performance Tradeoff
🟦 **What is cost-performance tradeoff in ML system design?**

🟩 **Definition**
Cost-performance tradeoff is balancing quality and speed against infrastructure cost. Better models often cost more to run. You must choose what is worth it for the product.

🟨 **How It Works / Example**
A larger LLM gives better answers but needs more GPUs. A smaller model is cheaper but may be less accurate. Teams sometimes use a small model first and call the large model only when needed.

🟪 **Quick Tip**
Quality vs Price.

---

## 708. Inference Batching
🟦 **What is batching in inference and why does it help?**

🟩 **Definition**
Batching groups multiple requests into one model run. It improves GPU efficiency and increases throughput. It can increase latency if batching waits too long.

🟨 **How It Works / Example**
A server collects requests for 10–20ms and runs them as a batch. This reduces overhead per request. You tune batch size and waiting time to keep latency acceptable.

🟪 **Quick Tip**
Group processing.

---

## 709. Dynamic Batching
🟦 **What is dynamic batching in LLM serving?**

🟩 **Definition**
Dynamic batching batches requests automatically as they arrive. It adapts to traffic levels instead of using a fixed schedule. It is common for high-traffic LLM APIs.

🟨 **How It Works / Example**
When traffic is high, the server forms larger batches quickly. When traffic is low, it forms smaller batches to avoid waiting. This keeps both throughput and latency in a good range.

🟪 **Quick Tip**
Adaptive grouping.

---

## 710. Caching
🟦 **What is caching in ML systems?**

🟩 **Definition**
Caching stores previously computed results to avoid recomputation. It reduces latency and cost for repeated requests. Caching is especially helpful for common queries.

🟨 **How It Works / Example**
If many users ask "reset password," you cache the retrieved docs or final answer. Next time, you return the cached result instantly. You set a TTL so caches refresh when content changes.

🟪 **Quick Tip**
Stored answers.

---

## 711. Embedding Cache
🟦 **What is embedding cache in RAG systems?**

🟩 **Definition**
Embedding cache stores embeddings for repeated queries or documents. It reduces compute and speeds retrieval. It is useful when the same inputs appear often.

🟨 **How It Works / Example**
You cache query embeddings for popular questions. When the question repeats, you skip embedding computation. Then you run vector search immediately.

🟪 **Quick Tip**
Stored vectors.

---

## 712. Load Balancer
🟦 **What is a load balancer and why is it needed for ML services?**

🟩 **Definition**
A load balancer distributes requests across multiple servers. It prevents one server from getting overloaded. It improves reliability and performance.

🟨 **How It Works / Example**
If you have 10 model replicas, the load balancer routes each request to a healthy one. If one replica crashes, traffic is sent to others. This keeps the service available.

🟪 **Quick Tip**
Traffic distributer.

---

## 713. Horizontal Scaling
🟦 **What is horizontal scaling for ML serving?**

🟩 **Definition**
Horizontal scaling adds more machines or service replicas to handle more traffic. It is common for stateless inference services. It helps increase throughput and reduce queueing.

🟨 **How It Works / Example**
During peak hours, you increase replicas from 5 to 20. Each replica serves part of the traffic. Autoscaling can do this automatically based on CPU/GPU usage or QPS.

🟪 **Quick Tip**
Adding servers.

---

## 714. Vertical Scaling
🟦 **What is vertical scaling for ML serving?**

🟩 **Definition**
Vertical scaling means using a larger machine with more CPU, RAM, or GPU power. It can reduce latency but has limits and can be expensive. It is often used when a single model needs more memory.

🟨 **How It Works / Example**
If an LLM does not fit on a small GPU, you move to a larger GPU. This may speed inference. But at very large scale, teams still usually combine vertical and horizontal scaling.

🟪 **Quick Tip**
Bigger servers.

---

## 715. Autoscaling
🟦 **What is autoscaling in ML infrastructure?**

🟩 **Definition**
Autoscaling automatically adds or removes serving instances based on load. It helps meet performance goals while controlling cost. It is essential for traffic spikes.

🟨 **How It Works / Example**
If QPS increases, autoscaling creates more pods. When QPS drops, it reduces pods. This keeps latency stable without paying for idle resources.

🟪 **Quick Tip**
Automatic sizing.

---

## 716. Model Parallelism
🟦 **What is model parallelism?**

🟩 **Definition**
Model parallelism splits a single model across multiple GPUs. It is used when the model is too large for one GPU. It enables serving or training very large models.

🟨 **How It Works / Example**
Layers 1–20 run on GPU1 and layers 21–40 run on GPU2. Data is passed between GPUs during inference. This allows bigger models but adds communication overhead.

🟪 **Quick Tip**
Split model.

---

## 717. Data Parallelism
🟦 **What is data parallelism?**

🟩 **Definition**
Data parallelism runs the same model on multiple GPUs with different data batches. It speeds training and sometimes batch inference. It works well when the model fits on one GPU.

🟨 **How It Works / Example**
GPU1 trains on batch A and GPU2 trains on batch B. Gradients are synchronized after each step. This reduces training time as you add more GPUs.

🟪 **Quick Tip**
Split data.

---

## 718. Pipeline Parallelism
🟦 **What is pipeline parallelism?**

🟩 **Definition**
Pipeline parallelism splits model layers across GPUs and processes micro-batches in a pipeline. It improves utilization when doing model parallelism. It is common in large model training.

🟨 **How It Works / Example**
GPU1 runs early layers on micro-batch 1 while GPU2 runs later layers on micro-batch 0. This keeps GPUs busy instead of waiting. It increases throughput but adds scheduling complexity.

🟪 **Quick Tip**
Pipelined split.

---

## 719. GPU Utilization
🟦 **What is GPU utilization and why does it matter?**

🟩 **Definition**
GPU utilization measures how much the GPU is doing useful work. Low utilization means wasted cost. High utilization usually improves throughput and reduces cost per request.

🟨 **How It Works / Example**
If a GPU is only 10% busy, requests may be too small or too few. Batching can increase utilization. Profiling helps find whether the bottleneck is CPU preprocessing or GPU compute.

🟪 **Quick Tip**
Hardware efficiency.

---

## 720. Bottleneck
🟦 **What is a bottleneck in an ML system?**

🟩 **Definition**
A bottleneck is the slowest part that limits system performance. It could be model compute, network, database, or preprocessing. Fixing bottlenecks improves overall latency and throughput.

🟨 **How It Works / Example**
If inference is fast but feature lookup is slow, the database is the bottleneck. You can add caching or move features to a faster store. Profiling and tracing help locate the bottleneck.

🟪 **Quick Tip**
Limiting factor.

---

## 721. Profiling
🟦 **What is profiling in ML performance tuning?**

🟩 **Definition**
Profiling measures where time and memory are spent in the system. It helps identify slow operations and inefficiencies. Profiling is required for meaningful optimization.

🟨 **How It Works / Example**
You profile an inference request and see 60% time is tokenization, not model compute. You optimize tokenization or parallelize it. This reduces total latency more than changing the model.

🟪 **Quick Tip**
Performance analysis.

---

## 722. Quantization
🟦 **What is model quantization for performance?**

🟩 **Definition**
Quantization reduces numeric precision of weights and activations, like FP32 to INT8. It usually speeds up inference and reduces memory use. It may slightly reduce accuracy.

🟨 **How It Works / Example**
You quantize an LLM to 8-bit so it fits on cheaper GPUs. Inference becomes faster and memory use drops. You test to ensure answer quality stays acceptable.

🟪 **Quick Tip**
Smaller numbers.

---

## 723. Distillation
🟦 **What is model distillation for scaling?**

🟩 **Definition**
Distillation trains a smaller model to mimic a larger model. It reduces serving cost while keeping much of the quality. It is useful for high-traffic products.

🟨 **How It Works / Example**
A large teacher model generates outputs for many inputs. A smaller student model is trained to match those outputs. The student is then deployed because it is faster and cheaper.

🟪 **Quick Tip**
Model copying.

---

## 724. Pruning
🟦 **What is pruning and how does it improve performance?**

🟩 **Definition**
Pruning removes less important parameters to reduce model size. Smaller models can run faster and use less memory. Pruning often needs fine-tuning to recover accuracy.

🟨 **How It Works / Example**
You remove weights with near-zero impact. Then you retrain briefly to adjust. If done well, latency improves with little accuracy loss.

🟪 **Quick Tip**
Trimming weights.

---

## 725. ONNX Runtime
🟦 **What is an inference runtime like TensorRT or ONNX Runtime used for?**

🟩 **Definition**
Inference runtimes optimize model execution for speed. They can fuse operations and use hardware-specific kernels. They are common for production inference.

🟨 **How It Works / Example**
You export a model to ONNX and run it with ONNX Runtime. The runtime applies graph optimizations and faster kernels. This can reduce latency compared to raw framework execution.

🟪 **Quick Tip**
Optimized execution.

---

## 726. Model Warm-up
🟦 **What is model warm-up in serving?**

🟩 **Definition**
Model warm-up runs a few fake requests to load weights and initialize caches. It reduces cold-start latency. Warm-up is useful after deployments or autoscaling events.

🟨 **How It Works / Example**
Right after a new pod starts, the first request might be slow due to model loading. Warm-up calls run inference once or twice early. Then real user requests are faster.

🟪 **Quick Tip**
Priming the model.

---

## 727. Cold Start Latency
🟦 **What is cold start latency and why is it common?**

🟩 **Definition**
Cold start latency happens when a service instance starts fresh and must load models and dependencies. It can cause slow first responses. It is common in autoscaling and serverless setups.

🟨 **How It Works / Example**
When traffic spikes, new pods start. They download the model and initialize GPU memory. During this time, requests may be slower unless you use warm pools or preloading.

🟪 **Quick Tip**
Startup delay.

---

## 728. Serving Queue
🟦 **What is a queue in ML serving architecture?**

🟩 **Definition**
A queue stores requests waiting to be processed. It helps smooth traffic spikes and supports batching. But long queues increase latency.

🟨 **How It Works / Example**
Requests arrive faster than the model can process. They wait in a queue until a worker is free. Autoscaling or increased batching reduces queue length and improves response time.

🟪 **Quick Tip**
Waiting line.

---

## 729. Backpressure
🟦 **What is backpressure in high-load ML systems?**

🟩 **Definition**
Backpressure is a way to slow down incoming requests when the system is overloaded. It prevents collapse and protects latency. It often uses rate limits or queue limits.

🟨 **How It Works / Example**
If the queue is full, the service returns "try again" or uses a fallback. This avoids infinite waiting times. It keeps the system stable during spikes.

🟪 **Quick Tip**
Load protection.

---

## 730. Rate Limiting
🟦 **What is rate limiting in ML APIs?**

🟩 **Definition**
Rate limiting restricts how many requests a client can send in a time period. It prevents abuse and keeps services stable. It is important for expensive ML endpoints like LLMs.

🟨 **How It Works / Example**
A client may be limited to 60 requests per minute. If it exceeds, requests are rejected or delayed. This protects GPUs and keeps latency acceptable for all users.

🟪 **Quick Tip**
Traffic control.

---

## 731. Multi-Tenancy
🟦 **What is multi-tenancy in ML system design?**

🟩 **Definition**
Multi-tenancy means one system serves multiple customers or groups. It requires isolation, access control, and fair resource usage. It is common in SaaS ML products.

🟨 **How It Works / Example**
A vector DB stores separate namespaces per customer. Retrieval is filtered by customer ID. Rate limits and quotas prevent one customer from consuming all resources.

🟪 **Quick Tip**
Shared system.

---

## 732. Data Isolation
🟦 **What is data isolation in multi-tenant ML systems?**

🟩 **Definition**
Data isolation ensures one customer cannot access another customer’s data. It is critical for security and privacy. It can be done with separate storage, encryption, or strict filtering.

🟨 **How It Works / Example**
Each request includes a tenant ID. The system filters retrieval and logs by that ID. Tests verify that cross-tenant access is impossible.

🟪 **Quick Tip**
Private data.

---

## 733. Feature Computation
🟦 **What is an ML feature computation strategy for real-time systems?**

🟩 **Definition**
It is the plan for creating features quickly at inference time. Some features are computed on the fly, others are precomputed and stored. The goal is low latency and correctness.

🟨 **How It Works / Example**
A fraud system may compute "transaction amount" instantly but precompute "user’s 7-day spend" in a feature store. The endpoint fetches precomputed features and combines them with real-time inputs. This keeps predictions fast.

🟪 **Quick Tip**
Real-time features.

---

## 734. Online/Offline Feature Store
🟦 **What is an online/offline feature store split?**

🟩 **Definition**
Offline stores support training with historical data. Online stores support low-latency retrieval for inference. Keeping both consistent avoids training-serving skew.

🟨 **How It Works / Example**
Offline features live in a warehouse for batch training. Online features live in Redis or a low-latency DB for serving. Both are generated using the same feature definitions.

🟪 **Quick Tip**
Dual storage.

---

## 735. RAG Retrieval Scaling
🟦 **What is a retrieval system design for RAG at scale?**

🟩 **Definition**
It is how you store, index, and search documents fast and safely. It includes chunking, embeddings, indexing, and filters. At scale, you also need caching and sharding.

🟨 **How It Works / Example**
You embed document chunks and store them in a vector DB with metadata. Queries retrieve top-k candidates and rerank them. You scale by sharding indexes and caching frequent queries.

🟪 **Quick Tip**
Scaling search.

---

## 736. LLM Serving Design
🟦 **What is LLM serving system design?**

🟩 **Definition**
LLM serving design focuses on managing expensive inference with good latency and cost. It includes batching, caching, and GPU scheduling. It also includes safety checks and fallbacks.

🟨 **How It Works / Example**
A gateway receives requests and applies rate limits. A scheduler batches requests and routes to available GPU workers. Outputs pass through moderation and then return to the user.

🟪 **Quick Tip**
LLM operations.

---

## 737. Context Window Management
🟦 **What is context window management in LLM system design?**

🟩 **Definition**
Context window management is choosing what text to include in the prompt within token limits. Too much context increases cost and can confuse the model. Good management keeps only the most useful information.

🟨 **How It Works / Example**
You keep recent chat turns, plus retrieved chunks, plus a short system instruction. Older history is summarized or dropped. This keeps the prompt under the maximum tokens while staying helpful.

🟪 **Quick Tip**
Token budget.

---

## 738. Prompt Caching
🟦 **What is prompt caching for LLMs?**

🟩 **Definition**
Prompt caching stores results for repeated prompts or shared prompt parts. It reduces compute and speeds responses. It is helpful for repeated system prompts and common user questions.

🟨 **How It Works / Example**
If your system prompt and policy text are the same for many users, you cache that prefix. Only the user-specific part changes. The server reuses cached computation to reduce latency and cost.

🟪 **Quick Tip**
Reusing prefills.

---

## 739. Structured Output
🟦 **What is structured output and why does it matter in system design?**

🟩 **Definition**
Structured output means the model returns a predictable format like JSON. It makes it easier for systems to parse and act on outputs. It reduces errors in downstream pipelines.

🟨 **How It Works / Example**
You ask the model to return `{ "intent": "...", "entities": [...] }`. The app reads the JSON and routes the request. If parsing fails, you retry or fall back to a rule-based method.

🟪 **Quick Tip**
Predictable format.

---

## 740. Idempotency
🟦 **What is idempotency and why is it useful for ML APIs?**

🟩 **Definition**
Idempotency means repeating the same request has the same effect. It prevents duplicate actions when clients retry due to timeouts. It is important for billing and workflow systems.

🟨 **How It Works / Example**
A client sends an inference request with an idempotency key. If the request is retried, the server returns the same stored response. This prevents double-counting events or charging twice.

🟪 **Quick Tip**
Safe retries.

---

## 741. Data Logging
🟦 **What is data logging for ML feedback loops?**

🟩 **Definition**
Data logging records inputs, outputs, and outcomes to improve the model later. It supports retraining, debugging, and monitoring. Good logging enables continuous improvement.

🟨 **How It Works / Example**
A recommendation system logs which items were shown and which were clicked. These logs become training data for the next model. You also log failures and user complaints to improve quality.

🟪 **Quick Tip**
Feedback loop.

---

## 742. Human-in-the-Loop
🟦 **What is human-in-the-loop design in ML systems?**

🟩 **Definition**
Human-in-the-loop means humans review or correct model outputs in some cases. It improves safety and quality when models are uncertain. It is common in high-stakes domains.

🟨 **How It Works / Example**
A fraud model flags borderline cases for manual review. Review decisions are logged as labels. The model is retrained later using these new labels to improve performance.

🟪 **Quick Tip**
Manual oversight.

---

## 743. Fallback Model
🟦 **What is an ML fallback model strategy?**

🟩 **Definition**
A fallback strategy uses a simpler or older model when the main model fails. It keeps the product working during outages or overload. It improves reliability.

🟨 **How It Works / Example**
If an LLM is too slow, you fall back to a smaller model or template-based replies. If a new model causes errors, you fall back to the previous stable version. This reduces user impact.

🟪 **Quick Tip**
Backup plan.

---

## 744. Recommender Scaling
🟦 **What is a common scaling challenge for recommendation systems?**

🟩 **Definition**
Recommendation systems must score many items quickly for each user. This can be expensive at large scale. Systems often use candidate retrieval plus reranking to manage cost.

🟨 **How It Works / Example**
First, a retrieval model selects 1,000 candidate items fast. Then a stronger model reranks the top candidates to pick the best 20. This reduces compute while keeping quality high.

🟪 **Quick Tip**
Retrieve then rank.

---

## 745. Candidate Generation
🟦 **What is candidate generation in recommender system design?**

🟩 **Definition**
Candidate generation is the first stage that selects a smaller set of items to consider. It focuses on recall and speed. It makes the full ranking step feasible.

🟨 **How It Works / Example**
You use embeddings to retrieve similar items to a user’s interests. This returns a few thousand items out of millions. Then the ranker scores them carefully to produce final recommendations.

🟪 **Quick Tip**
Fast selection.

---

## 746. Two-Tower Model
🟦 **What is a two-tower model used for scaling retrieval?**

🟩 **Definition**
A two-tower model embeds users and items into the same vector space. It supports fast nearest-neighbor search. It is commonly used for retrieval in recommendations and search.

🟨 **How It Works / Example**
One tower encodes user features into a user embedding. The other tower encodes item features into item embeddings. You retrieve nearest items to the user embedding using a vector index.

🟪 **Quick Tip**
Dense retrieval.

---

## 747. Sharding
🟦 **What is sharding and why is it used in ML system design?**

🟩 **Definition**
Sharding splits data or indexes across multiple machines. It allows scaling storage and throughput beyond one server. It adds complexity for routing and merging results.

🟨 **How It Works / Example**
A vector database may shard embeddings across 20 nodes. Each node searches its shard and returns top results. A coordinator merges results into final top-k.

🟪 **Quick Tip**
Distributed storage.

---

## 748. Reliability Engineering
🟦 **What is reliability engineering for ML services?**

🟩 **Definition**
Reliability engineering ensures ML services stay up and perform well under failures. It includes redundancy, monitoring, and safe rollouts. Reliable ML services protect user experience and revenue.

🟨 **How It Works / Example**
You run multiple replicas in different zones. Health checks remove unhealthy instances automatically. You use canary deployments and quick rollbacks to reduce risk.

🟪 **Quick Tip**
Staying online.

---

## 749. Capacity Planning
🟦 **What is capacity planning for ML workloads?**

🟩 **Definition**
Capacity planning estimates how much compute and storage you need to meet demand. It considers traffic, latency targets, and model cost. Good planning prevents outages and reduces waste.

🟨 **How It Works / Example**
You estimate peak QPS and required GPU time per request. Then you calculate how many GPUs are needed with headroom. You also plan for growth and traffic spikes.

🟪 **Quick Tip**
Resource estimation.

---

## 750. End-to-End Design
🟦 **What is a strong end-to-end system design answer in an ML interview?**

🟩 **Definition**
It clearly covers data, modeling, serving, scaling, monitoring, and failure handling. It explains trade-offs like accuracy vs latency and cost. It also includes how you evaluate and iterate safely.

🟨 **How It Works / Example**
For an LLM support bot, you describe RAG ingestion, retrieval, prompt design, serving with batching, and guardrails. You add monitoring for latency and answer quality. You explain rollout with A/B tests and rollback plans.

🟪 **Quick Tip**
Full system view.
