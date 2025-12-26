# Section 3: Data Processing & Feature Engineering

## 101. Data Preprocessing
🟦 **What is data preprocessing in machine learning?**

🟩 **Definition**
Data preprocessing is preparing raw data so a model can learn from it. It includes cleaning, formatting, and transforming data. Good preprocessing often improves model accuracy and stability.

🟨 **How It Works / Example**
For example, you may remove broken rows, fix data types, and scale numeric columns. Then the model receives consistent inputs. This reduces training errors and improves learning.

🟪 **Quick Tip**
"Garbage In, Good Data Out."

---

## 102. Data Cleaning
🟦 **What is data cleaning?**

🟩 **Definition**
Data cleaning means fixing problems in the data. This includes removing duplicates, correcting wrong values, and handling missing fields. Clean data reduces noise and improves model quality.

🟨 **How It Works / Example**
If a customer dataset has duplicate rows, you remove them to avoid double-counting. If ages contain impossible values like -5, you correct or remove them. Then training becomes more reliable.

🟪 **Quick Tip**
80% of a Data Scientist's job.

---

## 103. Missing Values
🟦 **What are missing values in a dataset?**

🟩 **Definition**
Missing values happen when some features have no recorded value. They can break training or bias results. You usually handle them with imputation or removal.

🟨 **How It Works / Example**
If "income" is missing for some users, you might fill it with the median income. Or you may add a "missing_income" flag feature. This helps the model handle missingness consistently.

🟪 **Quick Tip**
NaN, Null, None.

---

## 104. Imputation
🟦 **What is imputation for missing data?**

🟩 **Definition**
Imputation is filling missing values with reasonable replacements. Common choices are mean, median, most frequent value, or a special token. The goal is to keep data usable without adding too much bias.

🟨 **How It Works / Example**
For a numeric column like "temperature," you might replace missing values with the median temperature from training data. For text, you might use "unknown." This allows the model to train without errors.

🟪 **Quick Tip**
Educated guessing for empty cells.

---

## 105. Outlier Handling
🟦 **What is outlier handling in feature engineering?**

🟩 **Definition**
Outlier handling means dealing with extreme values that can distort learning. Outliers can be real events or data errors. You may clip, transform, or remove them depending on the case.

🟨 **How It Works / Example**
In salary data, a value of 10 million might be a mistake or a rare real case. You might cap salaries at the 99th percentile to reduce impact. This prevents the model from focusing too much on extreme points.

🟪 **Quick Tip**
Don't let the weirdos ruin the average.

---

## 106. Feature Engineering
🟦 **What is feature engineering?**

🟩 **Definition**
Feature engineering is creating or transforming input features to help a model learn better. It often uses domain knowledge to represent signals more clearly. Strong features can improve performance even with simple models.

🟨 **How It Works / Example**
For ride prediction, instead of raw timestamps, you create "hour_of_day" and "day_of_week." These capture patterns like rush hour. The model learns faster from these clearer signals.

🟪 **Quick Tip**
The art of making data learnable.

---

## 107. Categorical Feature
🟦 **What is a categorical feature?**

🟩 **Definition**
A categorical feature represents a group or type, like "city" or "color." It is not naturally numeric. Models often require encoding to convert categories into numbers.

🟨 **How It Works / Example**
"Payment_method" might be {card, cash, wallet}. You encode these so the model can use them. The encoding method depends on the model and number of categories.

🟪 **Quick Tip**
Labels, not quantities.

---

## 108. One-Hot Encoding
🟦 **What is one-hot encoding?**

🟩 **Definition**
One-hot encoding turns a category into a set of binary columns. Each column means "is this category present?" It works well for small to medium category counts.

🟨 **How It Works / Example**
If "color" is {red, blue, green}, you create three columns. A red item becomes [1,0,0], blue becomes [0,1,0]. This avoids giving categories a fake numeric order.

🟪 **Quick Tip**
Explodes dimensions (Curse of Dimensionality risk).

---

## 109. Label Encoding
🟦 **What is label encoding?**

🟩 **Definition**
Label encoding maps categories to integers like 0, 1, 2. It is simple but can accidentally imply order. It works better for tree-based models than linear models.

🟨 **How It Works / Example**
If "city" has 3 categories, you map them to 0, 1, 2. A tree model can split based on these values without assuming order. But a linear model may treat "2" as larger than "0," which can be wrong.

🟪 **Quick Tip**
Simple digits for complex trees.

---

## 110. Target Encoding
🟦 **What is target encoding?**

🟩 **Definition**
Target encoding replaces a category with the average target value for that category. It can work well for high-cardinality features. It must be done carefully to avoid leakage.

🟨 **How It Works / Example**
If predicting purchase, each "city" can be encoded as the city's historical purchase rate. But you compute these rates using only training folds. This avoids giving the model future information.

🟪 **Quick Tip**
Powerful but prone to leakage.

---

## 111. Feature Scaling
🟦 **What is feature scaling?**

🟩 **Definition**
Feature scaling adjusts numeric features to similar ranges. It helps many models train faster and behave better. Scaling is important for distance-based and gradient-based models.

🟨 **How It Works / Example**
If "income" ranges 0–200k and "age" ranges 0–100, income can dominate. Scaling makes them comparable. Then models like SVM, k-NN, and neural nets train more smoothly.

🟪 **Quick Tip**
Leveling the playing field.

---

## 112. Standardization (Z-Score)
🟦 **What is standardization (z-score) used for?**

🟩 **Definition**
Standardization rescales values to have mean 0 and standard deviation 1. It keeps relative differences but changes scale. It is common for linear models and neural networks.

🟨 **How It Works / Example**
You compute mean and std on training data. Then each value becomes (x − mean) / std. You reuse the same mean and std for validation and test data.

🟪 **Quick Tip**
Mean=0, Std=1.

---

## 113. Min-Max Normalization
🟦 **What is min-max normalization used for?**

🟩 **Definition**
Min-max normalization rescales values to a fixed range like 0 to 1. It is useful when features have different ranges. It can be sensitive to outliers.

🟨 **How It Works / Example**
You compute min and max from training data. Then each value becomes (x − min) / (max − min). This is common for image pixels or bounded feature inputs.

🟪 **Quick Tip**
Scaling to [0, 1].

---

## 114. Log Scaling
🟦 **What is data transformation with log scaling?**

🟩 **Definition**
Log scaling reduces the effect of very large values. It is useful for skewed distributions like income or counts. It can make relationships more linear for some models.

🟨 **How It Works / Example**
If "number_of_views" ranges from 1 to 1,000,000, the raw values are very skewed. Applying log(1+x) compresses large values. This helps the model learn without being dominated by huge numbers.

🟪 **Quick Tip**
Squashing the long tail.

---

## 115. Feature Selection
🟦 **What is feature selection?**

🟩 **Definition**
Feature selection is choosing a smaller set of useful features. It can improve accuracy, reduce overfitting, and speed up training. It also improves interpretability.

🟨 **How It Works / Example**
If you have 1,000 features, many may be noise. You can keep features that correlate with the target or that improve validation score. Then the model trains faster and may generalize better.

🟪 **Quick Tip**
More isn't always better.

---

## 116. Multicollinearity
🟦 **What is multicollinearity and why does it matter?**

🟩 **Definition**
Multicollinearity happens when two or more features are highly correlated. It can make linear model weights unstable and hard to interpret. It may also hurt generalization in some cases.

🟨 **How It Works / Example**
"Total_spend" and "average_spend_per_day" may strongly overlap. A linear model may assign weird positive and negative weights to both. You may drop one feature or use regularization.

🟪 **Quick Tip**
Redundant signals confuse linear models.

---

## 117. Derived Feature
🟦 **What is a derived feature?**

🟩 **Definition**
A derived feature is created by combining or transforming existing features. It can highlight patterns that are not obvious in raw data. Derived features often improve model performance.

🟨 **How It Works / Example**
In e-commerce, "price_per_unit" can be derived from price and quantity. This feature can better capture value than raw price alone. The model can then predict purchases more accurately.

🟪 **Quick Tip**
Synthetic helper variables.

---

## 118. Feature Interaction
🟦 **What is feature interaction?**

🟩 **Definition**
A feature interaction is when the effect of one feature depends on another. Some models learn interactions automatically, others need manual interaction features. Interactions can be very important in real data.

🟨 **How It Works / Example**
The effect of "discount" may depend on "customer_segment." You can create an interaction feature like discount × segment_flag. This helps simple models capture combined effects.

🟪 **Quick Tip**
X * Y > X + Y.

---

## 119. Binning
🟦 **What is binning in feature engineering?**

🟩 **Definition**
Binning converts continuous values into discrete buckets. It can reduce noise and handle non-linear patterns. It is often used in credit scoring and risk modeling.

🟨 **How It Works / Example**
You can bin "age" into groups like 0–18, 19–30, 31–50, 51+. Then the model learns patterns per age group. This can be more stable than using raw age.

🟪 **Quick Tip**
Grouping numbers into buckets.

---

## 120. Discretization
🟦 **What is discretization and how is it used?**

🟩 **Definition**
Discretization is turning continuous features into discrete values, similar to binning. It simplifies patterns and can improve interpretability. It may lose some information if bins are too coarse.

🟨 **How It Works / Example**
For "time_on_site," you can discretize into {short, medium, long}. This helps business teams understand the model. It can also reduce the impact of noisy time values.

🟪 **Quick Tip**
Turning sliders into checkboxes.

---

## 121. Text Preprocessing
🟦 **What is text preprocessing in NLP?**

🟩 **Definition**
Text preprocessing prepares text for modeling. It may include lowercasing, removing extra spaces, and handling punctuation. The exact steps depend on the model and task.

🟨 **How It Works / Example**
For a simple keyword model, you might remove punctuation and lowercase words. For a transformer model, you usually keep more raw text and rely on tokenization. The goal is consistent, clean input.

🟪 **Quick Tip**
Cleaning up the messiness of language.

---

## 122. Tokenization
🟦 **What is tokenization in data processing?**

🟩 **Definition**
Tokenization splits text into smaller units called tokens. Tokens can be words, subwords, or characters. Models use tokens as their basic input units.

🟨 **How It Works / Example**
The sentence "I love ML" may become tokens like ["I", "love", "ML"]. Subword tokenizers may split "unbelievable" into smaller parts. This helps handle rare words.

🟪 **Quick Tip**
Chopping text into bites.

---

## 123. Stopword Removal
🟦 **What is stopword removal?**

🟩 **Definition**
Stopword removal deletes very common words like "the" and "is." It can help simple text models focus on meaningful terms. For modern transformers, it is often not needed.

🟨 **How It Works / Example**
In a bag-of-words model, stopwords can add noise and inflate feature size. Removing them can improve speed and sometimes accuracy. But for sentiment tasks, some stopwords like "not" are important and should be kept.

🟪 **Quick Tip**
Removing "fluff" words.

---

## 124. Stemming
🟦 **What is stemming?**

🟩 **Definition**
Stemming reduces words to a shorter root form. It is a rough rule-based method and can produce non-real words. It helps group similar word forms together.

🟨 **How It Works / Example**
"Playing," "played," and "plays" may become "play." This reduces vocabulary size for simple models. It can improve matching in search or basic classifiers.

🟪 **Quick Tip**
Crude word chopping.

---

## 125. Lemmatization
🟦 **What is lemmatization?**

🟩 **Definition**
Lemmatization converts words to their dictionary base form. It is more accurate than stemming but often slower. It uses language rules to keep real words.

🟨 **How It Works / Example**
"Better" may become "good," and "running" becomes "run." This helps models treat similar meanings as the same feature. It is common in classic NLP pipelines.

🟪 **Quick Tip**
Smart word reduction.

---

## 126. Data Augmentation
🟦 **What is data augmentation?**

🟩 **Definition**
Data augmentation creates new training examples from existing ones. It helps models generalize and reduces overfitting. It is common in vision, audio, and sometimes text.

🟨 **How It Works / Example**
For images, you can flip, crop, or rotate pictures. The label stays the same, like "cat." This teaches the model that the object is still a cat even if the view changes.

🟪 **Quick Tip**
Free data from what you have.

---

## 127. Image Normalization
🟦 **What is normalization for images?**

🟩 **Definition**
Image normalization scales pixel values to a consistent range or distribution. It helps neural networks train more stably. Common methods include dividing by 255 and standardizing by mean and std.

🟨 **How It Works / Example**
Raw pixels might be 0–255. You can convert them to 0–1 by dividing by 255. Many pipelines also subtract a dataset mean and divide by std for better training.

🟪 **Quick Tip**
Pixels like to be small numbers.

---

## 128. Training vs Inference Preprocessing
🟦 **What is train-time vs inference-time preprocessing?**

🟩 **Definition**
Train-time preprocessing is applied when training the model. Inference-time preprocessing is applied when making predictions in production. They must match to avoid performance drops.

🟨 **How It Works / Example**
If you standardize features during training, you must use the same mean and std in production. If production skips scaling, the model receives different input ranges. This can cause wrong predictions.

🟪 **Quick Tip**
Consistency is key.

---

## 129. Data Pipeline
🟦 **What is a data pipeline in ML?**

🟩 **Definition**
A data pipeline is the set of steps that moves and transforms data for training or inference. It makes data processing repeatable and consistent. Pipelines reduce manual mistakes.

🟨 **How It Works / Example**
A pipeline may pull data from a database, clean it, encode categories, and save features. The same pipeline can run daily for new data. This makes production predictions consistent with training.

🟪 **Quick Tip**
Automated assembly line for data.

---

## 130. Feature Drift
🟦 **What is feature drift?**

🟩 **Definition**
Feature drift is when feature distributions change over time. It can reduce model performance even if the target definition stays the same. Monitoring feature drift helps catch issues early.

🟨 **How It Works / Example**
If average transaction amount rises due to inflation, a fraud model may behave differently. The model was trained on older distributions. Drift detection can trigger retraining or threshold updates.

🟪 **Quick Tip**
When input data changes shape.

---

## 131. Data Schema
🟦 **What is a data schema and why is it important?**

🟩 **Definition**
A data schema defines expected columns, types, and allowed values. It prevents silent data bugs. Schema checks protect ML pipelines from broken inputs.

🟨 **How It Works / Example**
If "age" suddenly becomes a string like "twenty," schema checks catch it. The pipeline can stop or fix the issue before training. This prevents corrupted models and wrong predictions.

🟪 **Quick Tip**
Contract for your data.

---

## 132. Data Deduplication
🟦 **What is data deduplication?**

🟩 **Definition**
Data deduplication removes repeated records. Duplicates can bias training and inflate metrics. It is especially important when data comes from logs or merges.

🟨 **How It Works / Example**
If the same user event is logged twice, the dataset may overcount that behavior. Deduplication removes exact or near-exact repeats. This leads to more accurate training statistics.

🟪 **Quick Tip**
One record, one vote.

---

## 133. Tabular Data Normalization
🟦 **What is data normalization for tabular data quality?**

🟩 **Definition**
Here, normalization means making formats consistent, not just scaling numbers. It includes standardizing text fields, units, and categories. This reduces messy variability in inputs.

🟨 **How It Works / Example**
You might convert "USA," "U.S.," and "United States" into one standard value. You may also convert heights into the same unit (cm). This prevents the model from treating the same meaning as different categories.

🟪 **Quick Tip**
Standardizing formats.

---

## 134. Time-Based Splitting
🟦 **What is time-based splitting for datasets?**

🟩 **Definition**
Time-based splitting separates train and test using time order. It avoids training on future data when predicting the future. It is important for forecasting and many real systems.

🟨 **How It Works / Example**
If predicting next month's churn, you train on older months and test on the most recent month. This matches how the model will be used. Random splitting could leak future trends into training.

🟪 **Quick Tip**
Don't peek into the future.

---

## 135. Rolling Window
🟦 **What is a rolling window in time series feature engineering?**

🟩 **Definition**
A rolling window computes features over a recent time period, like the last 7 days. It captures short-term trends and changes. Rolling features are common in forecasting and user behavior modeling.

🟨 **How It Works / Example**
For each day, you compute "purchases_last_7_days." This updates as time moves forward. The model uses it to predict future purchases or churn.

🟪 **Quick Tip**
Moving averages over time.

---

## 136. Lag Features
🟦 **What is lag feature creation in time series?**

🟩 **Definition**
Lag features use past values as inputs, like yesterday's sales. They help predict future values by using history. They are key for many time series models.

🟨 **How It Works / Example**
To predict today's demand, you add features like demand_1_day_ago and demand_7_days_ago. These capture daily and weekly patterns. The model learns how past demand relates to future demand.

🟪 **Quick Tip**
Using yesterday to predict today.

---

## 137. Data Balancing
🟦 **What is data balancing in classification?**

🟩 **Definition**
Data balancing addresses class imbalance by changing the training data mix. It can improve learning for rare classes. Common methods are oversampling and undersampling.

🟨 **How It Works / Example**
If fraud is rare, you can oversample fraud examples or undersample non-fraud. This makes the model see more positive cases. You still evaluate on the real distribution to measure true performance.

🟪 **Quick Tip**
Fixing the class ratio.

---

## 138. Oversampling
🟦 **What is oversampling?**

🟩 **Definition**
Oversampling increases the number of rare-class examples in training. It helps the model learn rare patterns. It can increase overfitting if done naively.

🟨 **How It Works / Example**
You can duplicate rare examples or use SMOTE to create synthetic ones. Then the model sees more positive samples per batch. This often improves recall on the minority class.

🟪 **Quick Tip**
Printing more money (data).

---

## 139. Undersampling
🟦 **What is undersampling?**

🟩 **Definition**
Undersampling reduces the number of common-class examples. It makes training more balanced and faster. It can lose useful information if you remove too much data.

🟨 **How It Works / Example**
If you have 1 million non-fraud and 10k fraud, you might keep only 100k non-fraud for training. This balances the dataset better. But you must be careful not to remove important variations.

🟪 **Quick Tip**
Dropping the crowd to hear the soloist.

---

## 140. SMOTE
🟦 **What is SMOTE?**

🟩 **Definition**
SMOTE is a method that creates synthetic minority-class examples. It interpolates between existing minority samples. It helps balance data without just duplicating points.

🟨 **How It Works / Example**
For rare fraud examples, SMOTE picks a fraud point and a nearby fraud neighbor. It creates a new point between them. This can help the model learn a smoother decision boundary.

🟪 **Quick Tip**
Synthetic Minority Over-sampling Technique.

---

## 141. Feature Hashing
🟦 **What is feature hashing?**

🟩 **Definition**
Feature hashing maps categories into a fixed-size numeric space using a hash function. It handles very large numbers of categories efficiently. It can cause collisions where different categories share a bucket.

🟨 **How It Works / Example**
If you have millions of unique user IDs, one-hot encoding becomes huge. Feature hashing maps each ID into one of, say, 1 million buckets. The model uses the bucket index as the feature.

🟪 **Quick Tip**
The "Hashing Trick".

---

## 142. High-Cardinality Data
🟦 **What is high-cardinality categorical data?**

🟩 **Definition**
High-cardinality means a categorical feature has many unique values. Examples include user IDs or URLs. It can be hard to encode without huge feature spaces.

🟨 **How It Works / Example**
If "product_id" has 500,000 unique values, one-hot encoding is too large. You might use target encoding, hashing, or learned embeddings. The best choice depends on model and leakage risk.

🟪 **Quick Tip**
Too many categories to count.

---

## 143. Feature Store
🟦 **What is a feature store?**

🟩 **Definition**
A feature store is a system to manage and serve features for training and inference. It helps keep features consistent across environments. It also supports reuse and versioning of features.

🟨 **How It Works / Example**
A team defines "7-day spend" once in the feature store. Training pipelines and production services both pull the same feature definition. This reduces mismatches and deployment bugs.

🟪 **Quick Tip**
Central bank for ML features.

---

## 144. Training-Serving Skew
🟦 **What is "training-serving skew"?**

🟩 **Definition**
Training-serving skew is when features differ between training and production. It causes performance drops after deployment. It often comes from inconsistent preprocessing or data sources.

🟨 **How It Works / Example**
If training uses "last_7_days_clicks" computed with full logs, but production computes it differently, predictions shift. The model behaves unpredictably. Using shared pipelines or a feature store reduces this risk.

🟪 **Quick Tip**
When dev != prod.

---

## 145. Feature Importance
🟦 **What is feature importance?**

🟩 **Definition**
Feature importance measures how much each feature affects model predictions. It helps interpret models and debug issues. Different models compute importance in different ways.

🟨 **How It Works / Example**
A tree model may show "income" is highly important for loan decisions. You can check if that makes sense and isn't leaking target info. If an unexpected feature is most important, you investigate data quality.

🟪 **Quick Tip**
Who's doing the heavy lifting?

---

## 146. Permutation Importance
🟦 **What is permutation importance?**

🟩 **Definition**
Permutation importance measures importance by shuffling one feature and seeing how performance changes. If performance drops a lot, the feature was important. It works for any model type.

🟨 **How It Works / Example**
You evaluate the model on validation data, then shuffle "age" across rows and evaluate again. If accuracy drops, "age" matters. If accuracy stays similar, "age" may not be helpful.

🟪 **Quick Tip**
Shake it up and see what breaks.

---

## 147. Leakage Risk in Feature Engineering
🟦 **What is leakage risk in feature engineering for time series?**

🟩 **Definition**
Leakage risk is accidentally using future information when building features. This makes offline metrics look great but fails in production. Time-based features must be computed using only past data.

🟨 **How It Works / Example**
If you compute "average sales this week" while predicting earlier days in that week, you leak future days. The model learns information it won't have at prediction time. Use strict time windows and backtesting to prevent this.

🟪 **Quick Tip**
Don't use tomorrow's news today.

---

## 148. Data Versioning
🟦 **What is data versioning and why is it important?**

🟩 **Definition**
Data versioning means tracking exactly which dataset snapshot was used. It helps reproduce experiments and debug changes. Without it, results can shift and become hard to explain.

🟨 **How It Works / Example**
You store dataset IDs or timestamps along with model artifacts. If performance drops later, you can compare data versions. This helps identify whether data changes caused the issue.

🟪 **Quick Tip**
Git for data.

---

## 149. Data Quality Monitoring
🟦 **What is "data quality monitoring" for ML pipelines?**

🟩 **Definition**
Data quality monitoring checks that incoming data stays valid and consistent. It looks for missing columns, weird values, or distribution shifts. It protects models from silent failures.

🟨 **How It Works / Example**
You set alerts if missing values jump from 1% to 30% in a key feature. Or if a category suddenly has many new unseen values. Monitoring catches issues before they harm predictions.

🟪 **Quick Tip**
Health checks for data.

---

## 150. Feature Normalization (Units)
🟦 **What is "feature normalization" in the sense of unit and scale consistency?**

🟩 **Definition**
This means ensuring features use the same units and consistent meaning. It prevents mixing incompatible values like miles vs kilometers. It improves model reliability and reduces confusing errors.

🟨 **How It Works / Example**
If distance is sometimes in miles and sometimes in km, the model sees inconsistent inputs. You convert everything to one unit during preprocessing. Then training and inference behave predictably.

🟪 **Quick Tip**
Apples to apples, not oranges.
