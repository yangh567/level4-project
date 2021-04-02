# This file is used to plot the maf summary graphs
library(maftools)
library(ggplot2)
library(data.table)

# read the data
path<-dir()[grep("gz",dir())]
result<-data.table()
for(k in path){
  organ_name<-unlist(strsplit(k,"[.]"))[2]
  print(organ_name)
  # assign the read content to the organ variable
  organ<-read.maf(k)

  # plot the summary graph
  pdf(file=paste0("summary_maf.",organ_name,".pdf"))
   plotmafSummary(maf = organ, rmOutlier = TRUE, addStat = 'median', dashboard = TRUE, titvRaw = FALSE)
  dev.off()
}
