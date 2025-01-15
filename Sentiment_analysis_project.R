library(tidytext)
library(text2vec)
library(devtools)
library(e1071)
library(dplyr)
library(caret)
library(glmnet)
library(textstem) # for stemming and lemmatizing
library(stringr)  
library(gmodels)
library(magrittr)
library(FSelector)
library(randomForest)
library(FactoMineR)
library(tidyr)
library(scales)
library(syuzhet)
library(sentimentr)
library(RColorBrewer)
library(wordcloud)
library(ggplot2)
library(lubridate)  # For working with dates

################################################################################################################### getting sentiments of the words 

df <- read.csv('MN-DS-news-classification.csv', stringsAsFactors = FALSE) # read csv
str(df)
df <- df %>% mutate(id = row_number())

tidy_content <- df %>% # transforms the text data into a clean and simplified format
  select(id, content) %>%
  unnest_tokens(word, content)
tidy_content <- tidy_content %>% 
  filter(!grepl('[0-9]',word)) %>% # remove numbers
  anti_join(stop_words) # remove stop words
tidy_content

afinn <- get_sentiments("afinn") # load AFINN sentiment lexicon
sentiments <- tidy_content %>%
  inner_join(afinn, by = "word") # associate sentiment scores with each word

sentiment_scores <- sentiments %>%
  group_by(id) %>%
  summarize(sentiment_score = sum(value)) # group by id and calculate total sentiment score for each id
content_with_sentiment <- inner_join(df, sentiment_scores, by = "id")

# histogram of afinn sentiment score distribution across the content
ggplot(content_with_sentiment, aes(x = sentiment_score)) +
  geom_histogram(binwidth = 1, fill = "purple", color = "black") +
  scale_x_continuous(breaks = seq(-20, 20, 5), limits = c(-20, 20)) +
  scale_y_continuous(label = comma) +
  labs(x = "Sentiment Score", y = "Count", title = "Distribution of Sentiment Scores using AFINN lexicon") 

bing_word_counts <- tidy_content %>%
  inner_join(get_sentiments("bing")) %>% # load bing sentiment lexicon
  count(word, sentiment, sort = TRUE) %>% # count occurrence of each word, grouped by sentiment
  ungroup() 

# bar plot showing top words for each bing sentiment
bing_word_counts %>%
  group_by(sentiment) %>%
  slice_max(n, n = 20) %>% # select top 20 words for each sentiment
  ungroup() %>%
  mutate(word = reorder(word, n)) %>% # reorder based on their frequency
  ggplot(aes(n, word, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y") +
  labs(x = "Frequency of word",
       y = NULL) 

sentiment_scores <- tidy_content %>%
  inner_join(get_sentiments('nrc'), by = 'word') # associate each word with nrc sentiment lexicon
sentiment_summary <- sentiment_scores %>%
  group_by(word) %>%
  summarize(
    anger = sum(sentiment == 'anger'),
    anticipation = sum(sentiment == 'anticipation'),
    disgust = sum(sentiment == 'disgust'),
    fear = sum(sentiment == 'fear'),
    joy = sum(sentiment == 'joy'),
    negative = sum(sentiment == 'negative'),
    positive = sum(sentiment == 'positive'),
    sadness = sum(sentiment == 'sadness'),
    surprise = sum(sentiment == 'surprise'),
    trust = sum(sentiment == 'trust')
  ) 

# bar plot of each sentiment category in nrc
barplot(
  sort(colSums(prop.table(sentiment_summary[, 2:11]))), 
  horiz = TRUE, 
  cex.names = 0.7, 
  las = 1, 
  main = "Emotions in Text", xlab="Percentage"
) 

################################################################################################################### getting sentiments of the sentences 

content_senti <- df %>% get_sentences() %>% sentiment_by(by=c('id')) # sentiment score at sentence level with grouping by id
contents <- cbind(df, content_senti) # merging the sentiment scores with the content

content_sentiment <- aggregate(ave_sentiment ~ category_level_1, data = contents, FUN = mean) # aggregate sentiment scores by category_level_1

# bar plot of the sentiment scores for each category_level_1
ggplot(data = content_sentiment, aes(x = category_level_1, y = ave_sentiment, fill = category_level_1)) + 
  geom_bar(stat = "identity") +                               
  labs(title = "Sentiment Analysis of News Content",          
       x = "Category Level 1", y = "Sentiment Score") +      
  theme(axis.text.x = element_text(angle = 90, hjust = 1))   

temp <- df
corpus <- VCorpus(VectorSource(temp$content)) 
corpus_clean <- tm_map(corpus, content_transformer(tolower))
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords('english'))
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stripWhitespace)
dtm <- DocumentTermMatrix(corpus_clean)
dtm <- removeSparseTerms(dtm, 0.98) # removing sparse terms
tdm <- as.matrix(dtm)

w <- sort(colSums(tdm), decreasing = TRUE) # sort term frequencies in decreasing order
set.seed(42) # random seed to ensure reproducibility of the word cloud layout
# word cloud of 150 most frequent words, that appear at least 5 times
wordcloud(words = names(w),
          freq = w,
          max.words = 150,
          random.order = F,
          min.freq = 5,
          colors = brewer.pal(8, 'Dark2'),
          scale = c(5, 0.3),
          rot.per = 0.7)

neg_words_count <- temp %>%
  unnest_tokens(word, content) %>% # tokenize words from content
  inner_join(get_sentiments("bing") %>% # join with bing sentiment lexicon
               filter(sentiment == "negative"),
             by = "word") %>%
  group_by(category_level_1) %>%                      
  summarize(
    negative_words = n() # count negative words per group
  )
# ratio of negative words for each category_level_1
temp %>%
  group_by(category_level_1) %>%
  summarize(
    content_count = n(), # total number of content entries
    total_words = sum(str_count(content, "\\w+")) # total number of words in all contents
  ) %>%
  left_join(neg_words_count, by = "category_level_1") %>%
  mutate(
    negative_words = replace_na(negative_words, 0), # replace NA with 0
    ratio = negative_words / total_words 
  ) %>%
  arrange(desc(ratio))

pos_words_count <- temp %>%
  unnest_tokens(word, content) %>%                    
  inner_join(get_sentiments("bing") %>%               
               filter(sentiment == "positive"),
             by = "word") %>%
  group_by(category_level_1) %>%                      
  summarize(
    positive_words = n()                              
  )
# ratio of positive words for each category_level_1
temp %>%
  group_by(category_level_1) %>%
  summarize(
    content_count = n(),                              
    total_words = sum(str_count(content, "\\w+"))     
  ) %>%
  left_join(pos_words_count, by = "category_level_1") %>%
  mutate(
    positive_words = replace_na(positive_words, 0),   
    ratio = positive_words / total_words              
  ) %>%
  arrange(desc(ratio))

################################################################################################################### adding afinn, bing and nrc sentiment values of each entry into the dataframe

afinn_scores <- temp %>%
  mutate(row_id = row_number()) %>% # add a unique identifier for each row              
  unnest_tokens(word, content) %>% # break content column into tokens
  inner_join(get_sentiments("afinn"), by = "word") %>%
  group_by(row_id) %>% # group by row identifier
  summarize(afinn_score = sum(value), .groups = "drop") # summaries sentiment score for each row by summing afinn values 

bing_scores <- temp %>%
  mutate(row_id = row_number()) %>%               
  unnest_tokens(word, content) %>%
  inner_join(get_sentiments("bing"), by = "word", relationship = "many-to-many") %>% # handle many-to-many for bing lexicon
  group_by(row_id, sentiment) %>% # group by row_id and sentiment type (pos/neg)
  summarize(count = n(), .groups = "drop") %>% # count occurence for each sentiment
  pivot_wider(names_from = sentiment, values_from = count, values_fill = 0) %>%
  mutate(bing_sentiment = case_when( # overall sentiment based on counts
    positive > negative ~ "positive",
    negative > positive ~ "negative",
    TRUE ~ "neutral"
  )) %>%
  select(row_id, bing_sentiment)

nrc_scores <- temp %>%
  mutate(row_id = row_number()) %>%                
  unnest_tokens(word, content) %>%
  inner_join(get_sentiments("nrc"), by = "word", relationship = "many-to-many") %>% 
  group_by(row_id, sentiment) %>%
  summarize(count = n(), .groups = "drop") %>%
  group_by(row_id) %>%
  filter(count == max(count)) %>% # keep the dominant sentiment for each row
  slice(1) %>% # breaks ties by taking the first occurrence
  select(row_id, nrc_sentiment = sentiment)

final_output <- temp %>%
  mutate(row_id = row_number()) %>% # merge based on row_id
  left_join(afinn_scores, by = "row_id") %>%
  left_join(bing_scores, by = "row_id") %>%
  left_join(nrc_scores, by = "row_id") %>%
  select(-row_id) %>% # remove row_id after joining
  mutate(
    afinn_score = ifelse(is.na(afinn_score), 0, afinn_score),        # replace NA in afinn_score with 0
    bing_sentiment = ifelse(is.na(bing_sentiment), "neutral", bing_sentiment),  # replace NA in bing_sentiment with "neutral"
    nrc_sentiment = ifelse(is.na(nrc_sentiment), "neutral", nrc_sentiment)   # replace NA in nrc_sentiment with "neutral"
  )

head(final_output,1)

###################################################################################################################

# bar plot of bing sentiment (pos, neg, neutral) frequency
ggplot(final_output, aes(x = bing_sentiment)) +
  geom_bar(fill = "orange") +
  labs(title = "Distribution of Bing Sentiments", x = "Sentiment", y = "Count") +
  theme_minimal()

# box plot of category_level_1 distribution in afinn sentiment score
ggplot(final_output, aes(x = category_level_1, y = afinn_score, fill = category_level_1)) +
  geom_boxplot() +
  labs(title = "AFINN Sentiment Score by News Category", x = "Category Level 1", y = "AFINN Score") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# line graph of afinn sentiment fluctuations across Jan 2019 to Dec 2019
final_output %>%
  mutate(date = ymd(date)) %>%  # Convert date to proper date type
  group_by(date) %>%
  summarize(avg_afinn = mean(afinn_score, na.rm = TRUE)) %>%
  ggplot(aes(x = date, y = avg_afinn)) +
  geom_line(color = "darkred") +
  labs(title = "Average Sentiment (AFINN) Over Time",
       x = "Date",
       y = "Average AFINN Score") +
  theme_minimal()


# word cloud of top 10 most frequent positive words
final_output %>%
  filter(bing_sentiment == "positive") %>%
  unnest_tokens(word, content) %>%
  anti_join(stop_words, by = "word") %>% # remove stop words
  count(word, sort = TRUE) %>%
  with(wordcloud(word, n, max.words = 10, colors = "green"))

# word cloud of top 10 most frequent negative words
final_output %>%
  filter(bing_sentiment == "negative") %>%
  unnest_tokens(word, content) %>%
  anti_join(stop_words, by = "word") %>%
  count(word, sort = TRUE) %>%
  with(wordcloud(word, n, max.words = 10, colors = "red"))

# bar plot displaying extremely positive, extremely negative and neutral counts for each category_level_1
final_output %>%
  mutate(sentiment_type = case_when(
    afinn_score >= 10 ~ "extremely positive",
    afinn_score <= -10 ~ "extremely negative",
    TRUE ~ "neutral"
  )) %>%
  group_by(category_level_1, sentiment_type) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = category_level_1, y = count, fill = sentiment_type)) +
  geom_col(position = "dodge") +
  labs(title = "Extreme Sentiments by Category", x = "Category Level 1", y = "Count") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# scatter plot showing relationship between the word count of a sentence and afinn score
final_output %>%
  mutate(word_count = str_count(content, "\\w+")) %>%
  ggplot(aes(x = word_count, y = afinn_score)) +
  geom_point(alpha = 0.5, color = "purple") +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(title = "Word Count vs Sentiment Score", x = "Word Count", y = "AFINN Score") +
  theme_minimal()

# bar plot showing the proportions of the different nrc sentiments for each category_level_1
final_output %>%
  count(category_level_1, nrc_sentiment) %>%
  group_by(category_level_1) %>%
  mutate(prop = n / sum(n)) %>%
  ggplot(aes(x = category_level_1, y = prop, fill = nrc_sentiment)) +
  geom_bar(stat = "identity", position = "fill") +
  labs(title = "Sentiment Proportions by Category (NRC)",
       x = "Category Level 1",
       y = "Proportion",
       fill = "Sentiment") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# heatmap of sentiments by category level and counts
final_output %>%
  count(category_level_1, nrc_sentiment) %>%
  ggplot(aes(x = category_level_1, y = nrc_sentiment, fill = n)) +
  geom_tile() +
  scale_fill_gradient(low = "orange", high = "blue") +
  labs(title = "Heatmap of Sentiments by Category",
       x = "Category Level 1",
       y = "Sentiment",
       fill = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# box plot showing distribution of nrc sentiments according to afinn scores
final_output %>%
  ggplot(aes(x = nrc_sentiment, y = afinn_score, fill = nrc_sentiment)) +
  geom_boxplot() +
  labs(title = "AFINN Scores by NRC Sentiments",
       x = "NRC Sentiment",
       y = "AFINN Score") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# bar plot showing the frequency of each category_level_1 
final_output %>%
  count(category_level_1, sort = TRUE) %>%
  ggplot(aes(x = reorder(category_level_1, n), y = n)) +
  geom_bar(stat = "identity", fill = "violet") +
  coord_flip() +
  labs(title = "Number of Articles by Subcategory", x = "Subcategory", y = "Count") +
  theme_minimal()

################################################################################ filtering out impactful category_level_2 categories

# filter top n category_level_2 by afinn score
top_categories <- final_output %>%
  group_by(category_level_2) %>%
  summarize(avg_afinn = mean(afinn_score, na.rm = TRUE)) %>%
  arrange(desc(avg_afinn)) %>%
  slice_head(n = 27)  # Top 27 category_level_2
average_categories <- final_output %>%
  filter(category_level_2 %in% top_categories$category_level_2)
unique(average_categories$category_level_2)

# bar plot of average afinn score for selected category_level_2
average_categories %>%
  group_by(category_level_2) %>%
  summarize(avg_afinn = mean(afinn_score, na.rm = TRUE), count = n()) %>%
  ggplot(aes(x = reorder(category_level_2, avg_afinn), y = avg_afinn, fill = count)) +
  geom_col() +
  coord_flip() +
  labs(title = "Top N Categories by Average Sentiment (AFINN Score)", x = "Subcategory", y = "Average Sentiment") +
  theme_minimal()

# filter category_level_2 with both high frequency and extreme sentiments
important_combined <- final_output %>%
  group_by(category_level_2) %>%
  summarize(
    avg_afinn = mean(afinn_score, na.rm = TRUE),
    count = n()
  ) %>%
  filter(count > 50 & abs(avg_afinn) > 20)  # frequent with strong sentiment
extreme_categories <- final_output %>%
  filter(category_level_2 %in% important_combined$category_level_2)
unique(extreme_categories$category_level_2)

# bar plot of average afinn score for selected category_level_2
extreme_categories %>%
  group_by(category_level_2) %>%
  summarize(avg_afinn = mean(afinn_score, na.rm = TRUE), count = n()) %>%
  ggplot(aes(x = reorder(category_level_2, avg_afinn), y = avg_afinn, fill = count)) +
  geom_col() +
  coord_flip() +
  labs(title = "Categories with Both High Frequency and Extreme Sentiments", x = "Subcategory", y = "Average Sentiment") +
  theme_minimal()

# bar plot showing count of each bing sentiment for selected category_level_2
extreme_categories%>%
  group_by(category_level_2, bing_sentiment) %>%
  summarize(count = n(), .groups = "drop") %>%
  ggplot(aes(x = reorder(category_level_2, -count), y = count, fill = bing_sentiment)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Sentiment Distribution by Subcategory", x = "Subcategory", y = "Count", fill = "Sentiment") +
  theme_minimal()

# bar plot showing count of each nrc sentiment for selected category_level_2
extreme_categories %>%
  group_by(category_level_2, nrc_sentiment) %>%
  summarize(count = n(), .groups = "drop") %>%
  mutate(proportion = count / sum(count)) %>%
  ggplot(aes(x = category_level_2, y = proportion, fill = nrc_sentiment)) +
  geom_bar(stat = "identity", position = "fill") +
  coord_flip() +
  labs(title = "NRC Sentiment Proportions by Subcategory", x = "Subcategory", y = "Proportion", fill = "Sentiment") +
  theme_minimal()

# box plot showing variability of each selected categroy_level_2 using afinn score
extreme_categories %>%
  ggplot(aes(x = category_level_2, y = afinn_score)) +
  geom_boxplot() +
  coord_flip() +
  labs(title = "Sentiment Variability Across Subcategories", x = "Subcategory", y = "AFINN Score") +
  theme_minimal()

# box plot showing word count of each selected categroy_level_2 
extreme_categories %>%
  mutate(content_length = str_count(content, "\\w+")) %>%
  ggplot(aes(x = category_level_2, y = content_length)) +
  geom_boxplot() +
  coord_flip() +
  labs(title = "Article Length by Subcategory", x = "Subcategory", y = "Word Count") +
  theme_minimal()

################################################################################ selecting which category_level_1 to do predictive modelling for

# filter category_level_1 categories with both high frequency and extreme sentiments
important_combined <- final_output %>%
  group_by(category_level_1) %>%
  summarize(
    avg_afinn = mean(afinn_score, na.rm = TRUE),
    count = n()
  ) %>%
  filter(count > 50 & abs(avg_afinn) > 20)  # Frequent with strong sentiment
categories <- final_output %>%
  filter(category_level_1 %in% important_combined$category_level_1)
unique(categories$category_level_1) # "crime, law and justice", "human interest", "lifestyle and leisure", "conflict, war and peace"

categories %>%
  group_by(category_level_1) %>%
  summarize(
    num_subcategories = n_distinct(category_level_2),
    total_entries = n()
  ) %>%
  arrange(desc(num_subcategories)) # understanding how many sub-categories and total entries are there

categories %>%
  group_by(category_level_1, category_level_2) %>%
  summarize(count = n(), .groups = "drop") %>%
  group_by(category_level_1) %>%
  summarize(
    max_subcategory_ratio = max(count) / sum(count),
    total_entries = sum(count)
  ) %>%
  arrange(max_subcategory_ratio)
# lifestyle and leisure not ideal because high dominance of one subcategory and insufficient data

# bar plot to visualize the proportion of the different subcategories in each category
categories %>%
  group_by(category_level_1, category_level_2) %>%
  summarize(count = n(), .groups = "drop") %>%
  ggplot(aes(x = category_level_1, y = count, fill = category_level_2)) +
  geom_col(position = "fill") +
  labs(
    title = "Subcategory Proportion Across Main Categories",
    x = "Category Level 1",
    y = "Proportion"
  ) +
  coord_flip() +
  theme_minimal() 

# bar plot to visualize the different sentiments of each subcategory for each category 
categories %>%
  group_by(category_level_1, category_level_2) %>%
  summarize(avg_afinn = mean(afinn_score, na.rm = TRUE), .groups = "drop") %>%
  ggplot(aes(x = category_level_2, y = avg_afinn, fill = category_level_1)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Average Sentiment by Subcategory and Main Category",
    x = "Subcategory",
    y = "Average AFINN Score"
  ) +
  theme_minimal() 

# ideal category for predictive modelling: conflict, war and peace 

# word cloud of 20 most frequent words of conflict, war and peace 
final_output %>%
  filter(category_level_1 == "conflict, war and peace") %>%  # Change to specific subcategory
  unnest_tokens(word, content) %>%
  anti_join(stop_words, by = "word") %>%
  count(word, sort = TRUE) %>%
  with(wordcloud(word, n, max.words = 20, colors = "blue"))

write.csv(final_output, "final_output.csv", row.names = FALSE)

final_output <- read.csv("~/Desktop/Intro to Data Science/final_output.csv")

