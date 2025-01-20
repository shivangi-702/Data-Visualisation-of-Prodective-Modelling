library(ggplot2)
library(dplyr)
library(lubridate)
library(tidyverse)
library(tm)
library(wordcloud)
library(RColorBrewer)

df <- read.csv('final_output.csv') # read the csv file

###################################################################################################################
# preprocess the data to get average sentiment by category and month
heatmap_data <- df %>%
  mutate(month = floor_date(as.Date(date), "month")) %>%
  group_by(category_level_1, month) %>%
  summarise(avg_sentiment = mean(afinn_score, na.rm = TRUE)) %>% # get the average AFINN sentiment score
  ungroup()

# create the heatmap
ggplot(heatmap_data, aes(x = month, y = category_level_1, fill = avg_sentiment)) +
  geom_tile() +
  scale_fill_gradient2(low = "red", high = "darkgreen", mid = "white", midpoint = 0, 
                       name = "Avg Sentiment") +
  scale_x_date(date_breaks = "1 month", date_labels = "%b %Y") +  # ensures all months are displayed
  labs(title = "Heatmap of Sentiment by Category and Month",
       x = "Date",
       y = "Category") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

###################################################################################################################
# preprocess data to count sentiment occurrences by category
bar_data <- df %>%
  group_by(category_level_1, bing_sentiment) %>%
  summarise(count = n()) %>%
  ungroup()

# create the stacked bar chart
ggplot(bar_data, aes(x = category_level_1, y = count, fill = bing_sentiment)) +
  geom_bar(stat = "identity", position = "fill") +
  scale_fill_manual(values = c("negative" = "darkred", "positive" = "darkgreen", "neutral" = "lightblue"), 
                    name = "Sentiment") +
  labs(title = "Stacked Bar Chart of Sentiment Composition by Category",
       x = "Category",
       y = "Proportion") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

###################################################################################################################
# line graph of Jan 2019 to Dec 2019
df %>%
  mutate(date = ymd(date)) %>%  # Convert date to proper date type using lubridate: year, month, date (ymd)
  group_by(date) %>%
  summarize(avg_afinn = mean(afinn_score, na.rm = TRUE)) %>%
  ggplot(aes(x = date, y = avg_afinn)) +
  geom_line(color = "blue") +
  scale_x_date(
    date_breaks = "1 month", # displays all the months
    date_labels = "%b %Y" # displays the date by month and year   
  )+
  labs(title = "Average Sentiment (AFINN) Over Time",
       x = "Date",
       y = "Average AFINN Score") +
  theme_minimal()+theme(axis.text.x = element_text(angle = 45, hjust = 1))

###################################################################################################################
# getting summary statistics of the data
monthly_stats <- df %>%
  mutate(month = floor_date(ymd(date), "month")) %>%
  group_by(month) %>%
  summarize(
    mean = mean(afinn_score, na.rm = TRUE),
    median = median(afinn_score, na.rm = TRUE),
    sd = sd(afinn_score, na.rm = TRUE)
  )
print(monthly_stats) # view the summary statistics 

iqr_values <- df %>% # getting inter quartile range to filter out outliers 
  summarize(
    Q1 = quantile(afinn_score, 0.25, na.rm = TRUE),
    Q3 = quantile(afinn_score, 0.75, na.rm = TRUE),
    IQR = Q3 - Q1
  )
lower_bound <- iqr_values$Q1 - 1.5 * iqr_values$IQR
upper_bound <- iqr_values$Q3 + 1.5 * iqr_values$IQR

outliers <- df %>% # identify outliers
  filter(afinn_score < lower_bound | afinn_score > upper_bound)
length(outliers) # 16 outliers

###################################################################################################################
# preprocess for word clouds
jan_2019 <- df[df$date >= "2019-01-01" & df$date <= "2019-01-31", ] # filter to have only Jan 2019 data
dec_2019 <- df[df$date >= "2019-12-01" & df$date <= "2019-12-31", ] # filter to have only Dec 2019 data

par(mfrow = c(1, 2))  # plotting side-by-side word clouds; layout is 1 row, 2 columns

wordcloud <- function(df){ # function to create the word clouds
  data <- df %>%                                 # using either jan 2019 or dec 2019 data
          select(id, content, bing_sentiment) %>%      
          unnest_tokens(output = word, input = content) %>%  # tokenize the content into words
          anti_join(stop_words) %>%                    # remove stop words
          count(bing_sentiment, word, sort = TRUE) %>% # count words by the different sentiments (positive/negative)
          spread(key = bing_sentiment, value = n, fill = 0) %>%  # reshaping to have separate columns for positive/negative
          data.frame()
    
  # setting the row names to the words
  rownames(data) <- data$word
  data <- data[, c('positive', 'negative')]
  
  # generating the word cloud
  set.seed(42)
  comparison.cloud(
    term.matrix = data, # term frequency matrix
    scale = c(2, 0.5), # adjusting word size 
    max.words = 50, # maximum number of words
    rot.per = 0 # setting rotation percentage to 0 for horizontal words
  )
}

wordcloud(jan_2019) # wordcloud for Jan 2019
wordcloud(dec_2019) # wordcloud for Dec 2019
