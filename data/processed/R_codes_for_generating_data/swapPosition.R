# this is the file used to swap the columns(deprecated)
rm(list = ls())
library(data.table)
library(dplyr)
data<-fread("sample_id.sbs.organ.csv")
sbs<-colnames(data)[grep("SBS",colnames(data))]
gene<-colnames(data)[-grep("SBS",colnames(data))]
gene<-gene[!gene%in%c("organ","clinic")]
date1<-data%>%select(Sample_ID,sbs) 
date2<-data%>%select(gene,organ)
date<-cbind(date1,date2)
fwrite(date,"sample_id.sbs.gene.organ.csv",row.names = F)


# do the checking here
datachecking<-fread("sample_id.sbs.gene.organ.csv")
datachecking
