library(boot)
library(ggplot2)
library(reshape2)
library(gridExtra)

r <- function(d, i){
  b <- d[i,]
  corr = cor(b[,1], b[,2], method = "pearson")
  return(corr)
}

dist_to_sim <- function(dists) {
  return(1-dists)
}

bootstraps_5l <- function(data, n, correlate_with){
  for_r <- data[c("emb_cossim", correlate_with)]
  emb <- boot(for_r, r, n)
  for_r <- data[c("mfccs_cossim", correlate_with)]
  mfcc <- boot(for_r, r, n)
  for_r <- data[c("X0_cossim", correlate_with)]
  layer0 <- boot(for_r, r, n)
  for_r <- data[c("X1_cossim", correlate_with)]
  layer1 <- boot(for_r, r, n)
  for_r <- data[c("X2_cossim", correlate_with)]
  layer2 <- boot(for_r, r, n)
  for_r <- data[c("X3_cossim", correlate_with)]
  layer3 <- boot(for_r, r, n)
  for_r <- data[c("X4_cossim", correlate_with)]
  layer4 <- boot(for_r, r, n)
  
  ts <- data.frame(do.call(cbind, list(mfcc$t, layer0$t, layer1$t, 
                                          layer2$t, layer3$t, layer4$t, emb$t)))
  colnames(ts) <- c("MFCC", "Layer 1", "Layer 2", "Layer 3", "Layer 4", "Layer 5", "Embedding")
  return(ts)
}

plot_coco <- function(data, n) {
  ts_hr <- bootstraps_5l(data, n, "hr")
  ts_wm <- bootstraps_5l(data, n, "wordmodel_cossim")
  ts_ed <- bootstraps_5l(data, n, "edit_similarity")
  ts_hr = melt(ts_hr)
  ts_ed = melt(ts_ed)
  ts_wm = melt(ts_wm)
  all = cbind(ts_ed, ts_wm, ts_hr)
  colnames(all) <- c("layer", "Edit similarity", "layer", "Text RHN embedding", "layer", "Semantic similarity ratings")
  newall <- melt(all)
  plot <- ggplot(data = newall, aes(x=layer, y=value, colour = variable))
  return(plot)
}

# define nr of resamplings
n <- 10000

simdata <- read.delim("z_score_coco_sick.csv")
# convert edit distance to edit similarity
simdata$edit_similarity <- dist_to_sim(simdata$norm_edit_distance)
coco_sick_plot_basic <- plot_coco(simdata, n)
coco_sick_plot_2 <- coco_sick_plot_basic + geom_boxplot(lwd=1, position="identity") + coord_flip()
coco_sick_plot <- coco_sick_plot_2 + theme(legend.position="none")
#coco_plot <- grid.arrange(coco_sick_plot, coco_sts_plot, ncol=2)
coco_sick_plot + scale_y_continuous(limits = c(0.42, 0.83)) + theme(legend.text = element_text(size=28), 
                                             legend.title=element_blank(), 
                                             axis.text = element_text(size=30), 
                                             axis.title.y = element_blank(), 
                                             axis.title.x = element_blank(), 
                                             legend.position="bottom", 
                                             legend.direction = "vertical",
                    panel.grid.minor = element_line(colour="white", size=1.5),
                    panel.grid.major = element_line(colour="white", size=1.5))