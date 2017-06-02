library(ggplot2)
library(dplyr)

rer <- function(lo, hi) { ((1-lo)-(1-hi))/(1-lo) }

data <- read.table(header=TRUE, file="ambigu-layerwise.txt") %>%
  mutate(words = paste(as.character(word1), as.character(word2), sep="/")) %>%
  #filter(words != "played/plaid") %>% # spurious
  mutate(rer=rer(majority, acc))

lev <- (data %>% group_by(words) %>% summarize(rer=mean(rer)) %>% arrange(rer))$words

data <- data %>% mutate(words=factor(words, lev), minority=1-majority, mincount=pmin(count1, count2)) %>%
  filter(majority < 0.95) %>%
  arrange(rer)

data.io <- read.table(header=TRUE, file="ambigu-io.txt") %>% 
  mutate(words = paste(as.character(word1), as.character(word2), sep="/")) %>%
  mutate(rer=rer(majority, acc))
lev <- (data.io %>% group_by(words) %>% summarize(rer=mean(rer)) %>% arrange(rer))$words
data.io <- data.io %>% mutate(words=factor(words, lev), minority=1-majority, mincount=pmin(count1, count2)) %>%
  filter(majority < 0.95) %>%
  #filter(words != "played/plaid") %>%
  arrange(rer)

conv <- function(x){ sapply(x, function(io) {
   if (io=='input'){
      return(-1)
    } else if (io=='output'){
      return(5)
    }
  })
}

data.full <- rbind_list(data.io %>% mutate(layer = conv(io)), data)

data.prep <- data.full %>% 
  filter(words != "played/plaid") %>%
  group_by(words,io,layer) %>% 
  summarize(rer=mean(rer), minority=mean(minority), 
                mincount=mean(mincount))
ggplot(data.prep %>% filter(layer < 5 & layer > -1), 
       aes(x=layer+1, y=rer, color=words, alpha=log(mincount))) + 
  geom_line(size=2, linetype="solid") + 
  geom_line(data=data.prep %>% filter(layer <= 0 ), size=2, linetype="dotted") +
  geom_line(data=data.prep %>% filter(layer >= 4 ), size=2, linetype="dotted") +
  geom_line(data=data.prep %>% filter(layer <= 0 ) %>% group_by(layer) %>% summarize(rer=mean(rer), mincount=mean(mincount)), linetype="dotted", size=2, color="black") +
  geom_line(data=data.prep %>% filter(layer >= 0 & layer <= 4) %>% group_by(layer) %>% summarize(rer=mean(rer), mincount=mean(mincount)), size=2, color="black") +
  geom_line(data=data.prep %>% filter(layer >= 4 ) %>% group_by(layer) %>% summarize(rer=mean(rer), mincount=mean(mincount)), linetype="dotted", size=2, color="black") +
  xlab("layer") + ylab("RER") + 
  theme(text=element_text(size=22), aspect.ratio=1, legend.position="bottom")
ggsave(filename = "ambigu-layerwise.pdf", width=12, height=10)


