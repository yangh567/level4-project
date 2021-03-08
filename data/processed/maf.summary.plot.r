library(maftools)
library(ggplot2)
library(data.table)
path<-dir()[grep("gz",dir())]
result<-data.table()
for(k in path){
  organ_name<-unlist(strsplit(k,"[.]"))[2]
  print(organ_name)
  organ<-read.maf(k)
  
  pdf(file=paste0("summary_maf.",organ_name,".pdf"))
   plotmafSummary(maf = organ, rmOutlier = TRUE, addStat = 'median', dashboard = TRUE, titvRaw = FALSE)
  dev.off()
}
