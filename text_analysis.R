library(tidyverse)
library(tokenizers)
library(tidytext)
library(lexicon)


# Data import -------------------------------------------------------------

# Transcript extraction ---------------------------------------------------
all_files <- unzip("youtube-personality.zip", list = T)$Name

filenames <- all_files[grepl("^youtube-personality/transcripts/VLOG", all_files)]

readTextFile <- function(nm) {
  unzip("youtube-personality.zip", nm);
  text <- readChar(nm, nchars = 1000000);
  unlink(nm)
  text
}

transcripts <- tibble()

for (i in 1:404) {
  id <- str_extract(filenames[i], "VLOG\\d+")
  trans <- tibble("Id" = id, "vlog" = readTextFile(filenames[i]))
  transcripts <- bind_rows(transcripts, trans)
}


# Other files -------------------------------------------------------------
gender <- unzip("youtube-personality.zip", "youtube-personality/YouTube-Personality-gender.csv") %>% read_delim(delim = " ")
audiovisual <- unzip("youtube-personality.zip", "youtube-personality/YouTube-Personality-audiovisual_features.csv") %>% read_delim(delim = " ")
train <- unzip("youtube-personality.zip", "youtube-personality/YouTube-Personality-Personality_impression_scores_train.csv") %>% read_delim(delim = " ")

# join all variables for training set
train_set <- train %>% 
  inner_join(audiovisual) %>% 
  inner_join(gender)

# join all variables for test set
test_set <- audiovisual %>% 
  anti_join(train) %>% 
  inner_join(gender)


# Dictionary creation -----------------------------------------------------
# We found literature that related certain text features to the big five personality traits. 
# We had to create our own dictionary. We extracted the tokens from online resources and put it all
# together in Excel and saved it as a csv file. All dictionaries used are available from our github 
# repository: 
# https://github.com/Joris-H/BDA/tree/Text_analysis
# Furthermore, we used the R-package "lexicon" which had some useful dictionaries too. 
# We binded them all together so that we could score all features in one go. 


# our updated nrc dictionary, it additionally contains sentiments: article, I, you, sexual, tentative,
# negate
new_nrc <- read_csv("newnrc.csv")
new_nrc <- new_nrc %>% 
  filter(sentiment != "prep")

# LIWC: we created excel csv files with additional features found in the LIWC dictionary
assent <- tibble(word = read_csv("LIWC - assent.csv", 
                                 col_names = FALSE)$X1, 
                 sentiment = "assent")

bio <- tibble(word = read_delim("LIWC - bio.csv", 
                                ";", escape_double = FALSE, col_names = FALSE, 
                                trim_ws = TRUE)$X1, 
                 sentiment = "bio")

filler <- tibble(word = read_csv("LIWC - filler.csv", 
                                 col_names = FALSE)$X1, 
                 sentiment = "filler")

health <- tibble(word = read_csv("LIWC - health.csv", 
                                 col_names = FALSE)$X1, 
                 sentiment = "health")

nonflu <- tibble(word = read_csv("LIWC - nonflu.csv", 
                                 col_names = FALSE)$X1, 
                 sentiment = "nonflu")

# lexicon package
profanity <- tibble(word = profanity_zac_anger, sentiment = "profanity")

constrain <- tibble(word = constraining_loughran_mcdonald, sentiment = "constrain")

financial <- tibble(word = hash_sentiment_loughran_mcdonald$x, sentiment = "financial")

verb <- tibble(word = pos_action_verb, sentiment = "verb")

prep <- tibble(word = pos_preposition, sentiment = "prep")

pronouns <- tibble(word = pos_df_pronouns$pronoun, sentiment = pos_df_pronouns$point_of_view)

# Anonymised names
XXXX <- tibble(word = "xxxx", sentiment = "XXXX")

# bind
new_nrc <- new_nrc %>% 
  bind_rows(assent, bio, filler, health, nonflu, profanity, constrain, financial, verb, prep, pronouns, XXXX)


# Text analysis -----------------------------------------------------------

# extract nrc features of all transcripts ---------------------------------
transcript_features_nrc <- 
  transcripts %>%
  unnest_tokens(token, vlog, token = 'words') %>%
  inner_join(new_nrc, by = c(token = 'word')) %>%
  count(Id, `sentiment`) %>%
  spread(sentiment, n, fill = 0)


# extract proportions of nrc features used and add total count ------------
proportions_nrc <- transcript_features_nrc %>% 
  mutate(sumVar = rowSums(.[,2:31])) %>% 
  mutate_at(vars(XXXX:you), funs(./ sumVar))


# join new computed variables with train and test set ---------------------
train_joined <- train_set %>% 
  inner_join(proportions_nrc, by = c('vlogId' = 'Id'))

test_joined <- test_set %>% 
  inner_join(proportions_nrc, by = c('vlogId' = 'Id')) 

# recode gender to factor
train_joined <- train_joined %>% 
  mutate(gender = as.factor(gender))

test_joined <- test_joined %>% 
  mutate(gender = as.factor(gender))

# Analysis ----------------------------------------------------------------
library(MASS)

# Extroversion ------------------------------------------------------------
# extract data
data_ext <- train_joined %>% dplyr::select(-c(vlogId, Agr:Open))

# fit complete model
model_ext <- lm(Extr ~(.), data=data_ext)

# make stepwise selection
step_ext <- stepAIC(model_ext, direction = "both", trace = F)
summary(step_ext)

# remove outliers
cooksd <- cooks.distance(step_ext)
influential <- as.numeric(names(cooksd)[(cooksd > 4*mean(cooksd, na.rm=T))])
head(data_ext[influential, ])

data_ext_outrm <- data_ext %>% slice(-influential)

### fit again
# fit best model on data without outliers
model_ext_final <- lm(Extr ~ mean.pitch + mean.conf.pitch + sd.spec.entropy + 
                      sd.val.apeak + mean.loc.apeak + sd.loc.apeak + mean.energy + 
                      mean.d.energy + sd.d.energy + time.speaking + hogv.entropy + 
                      hogv.median + hogv.cogR + gender + anger + article + assent + 
                      disgust + filler + first + negate + positive + profanity + 
                      second + tentat + sumVar, data=data_ext_outrm)

summary(model_ext_final)

## predictions
predictions_extr <- tibble(Id = test_joined$vlogId, type = "Extr", Expected = predict(model_ext_final, newdata = test_joined))


# Agreeableness -----------------------------------------------------------

# extract data
data_agr <- train_joined %>% dplyr::select(-c(vlogId, Extr, Cons:Open))

# fit complete model
model_agr <- lm(Agr~(.), data=data_agr)

# make stepwise selection
step_agr <- stepAIC(model_agr, direction = "both", trace = F)
summary(step_agr)

# remove outliers
cooksd <- cooks.distance(step_agr)
influential <- as.numeric(names(cooksd)[(cooksd > 4*mean(cooksd, na.rm=T))])
head(data_agr[influential, ])

data_agr_outrm <- data_agr %>% slice(-influential)

### fit again
# fit best model on data without outliers
model_agr_final <- lm(Agr ~ hogv.cogC + gender + anticipation + article + 
                        assent + bio + fear + first + first_person + health + joy + 
                        nonflu + positive + prep + profanity + second + surprise + 
                        tentat + third + verb + sumVar, data=data_agr_outrm)

summary(model_agr_final)

## predictions
predictions_agr <- tibble(Id = test_joined$vlogId, type = "Agr", Expected = predict(model_agr_final, newdata = test_joined))


# Cons --------------------------------------------------------------------

# extract data
data_cons <- train_joined %>% dplyr::select(-c(vlogId, Extr:Agr, Emot:Open))

# fit complete model
model_cons <- lm(Cons~(.), data=data_cons)

# make stepwise selection
step_cons <- stepAIC(model_cons, direction = "both", trace = F)
summary(step_cons)

# remove outliers
cooksd <- cooks.distance(step_cons)
influential <- as.numeric(names(cooksd)[(cooksd > 4*mean(cooksd, na.rm=T))])
head(data_cons[influential, ])

data_cons_outrm <- data_cons %>% slice(-influential)

### fit again
# fit best model on data without outliers
model_cons_final <- lm(Cons ~ sd.pitch + mean.spec.entropy + sd.val.apeak + 
                         avg.voiced.seg + time.speaking + hogv.entropy + hogv.cogC + 
                         anger + assent + bio + fear + first_person + health + joy + 
                         negate + positive + trust + profanity, data=data_cons_outrm)

summary(model_cons_final)

## predictions
predictions_cons <- tibble(Id = test_joined$vlogId, type = "Cons", Expected = predict(model_cons_final, newdata = test_joined))

# Emot --------------------------------------------------------------------

# extract data
data_emot <- train_joined %>% dplyr::select(-c(vlogId:Cons, Open))

# fit complete model
model_emot <- lm(Emot~(.), data=data_emot)

# make stepwise selection
step_emot <- stepAIC(model_emot, direction = "both", trace = F)
summary(step_emot)

# remove outliers
cooksd <- cooks.distance(step_emot)
influential <- as.numeric(names(cooksd)[(cooksd > 4*mean(cooksd, na.rm=T))])
head(data_emot[influential, ])

data_emot_outrm <- data_emot %>% slice(-influential)

### fit again
# fit best model on data without outliers
model_emot_final <- lm(Emot ~ mean.spec.entropy + mean.val.apeak + sd.val.apeak + 
                         mean.energy + sd.d.energy + time.speaking + hogv.median + 
                         hogv.cogR + hogv.cogC + article + constrain + first + positive + 
                         second + sexual + surprise + tentat + third + profanity + 
                         negative, data=data_emot_outrm)

summary(model_emot_final)

## predictions
predictions_emot <- tibble(Id = test_joined$vlogId, type = "Emot", Expected = predict(model_emot_final, newdata = test_joined))


# Open --------------------------------------------------------------------

# extract data
data_open <- train_joined %>% dplyr::select(-c(vlogId:Emot))

# fit complete model
model_open <- lm(Open~(.), data=data_open)

# make stepwise selection
step_open <- stepAIC(model_open, direction = "both", trace = F)
summary(step_open)

# remove outliers
cooksd <- cooks.distance(step_open)
influential <- as.numeric(names(cooksd)[(cooksd > 4*mean(cooksd, na.rm=T))])
head(data_open[influential, ])

data_open_outrm <- data_open %>% slice(-influential)

### fit again
# fit best model on data without outliers
model_open_final <- lm(Open ~ sd.val.apeak + mean.loc.apeak + sd.loc.apeak + 
                         mean.d.energy + avg.voiced.seg + time.speaking + hogv.median + 
                         hogv.cogC + gender + anger + first + negative + positive + 
                         trust, data=data_open_outrm)

summary(model_open_final)

## predictions
predictions_open <- tibble(Id = test_joined$vlogId, type = "Open", Expected = predict(model_open_final, newdata = test_joined))


# Combine predictions -----------------------------------------------------
predictions_final <- bind_rows(predictions_extr, predictions_agr, predictions_cons, predictions_emot, predictions_open)

predictions_unite <- predictions_final %>% 
  unite(Id, Id, type, sep = "_")

# Write to CSV
write_csv(predictions_unite, "master_predictions_final2")

