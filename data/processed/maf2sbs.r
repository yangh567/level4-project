# original maf file to sample_id x sbs signatures matrix
rm(list = ls())
install.packages("devtools")
install.packages("remotes")
library("devtools")
chooseBioCmirror()
BiocManager::install("sigminer", dependencies = TRUE)
BiocManager::install("BSgenome.Hsapiens.UCSC.hg19")
BiocManager::install("BSgenome.Hsapiens.UCSC.hg38")
BiocManager::install("maftools")
BiocManager::install("PoisonAlien/TCGAmutations")

library(sigminer)
library(maftools)
library(dplyr)
library(data.table)
library(plyr)
library(tidyr)
save_pheatmap_pdf <- function(x, filename, width=20, height=10) {
  stopifnot(!missing(x))
  stopifnot(!missing(filename))
  pdf(filename, width=width, height=height)
  grid::grid.newpage()
  grid::grid.draw(x$gtable)
  dev.off()
}
sim_name<-function(x){
  sbs_name<-colnames(t(x[x==max(x)]))
  return(sbs_name)
  }


path<-dir()[grep("gz",dir())]
result<-data.table()
for(k in path){
  #k=path[3]
  organ_name<-unlist(strsplit(k,"[.]"))[2]
  print(organ_name)
  organ<-read.maf(k)
  b<-organ@data
  sampleid_gene<-as.data.frame.array(t(table(b$Hugo_Symbol,b$Tumor_Sample_Barcode)))
  sampleid_gene$Sample_ID<-row.names(sampleid_gene)
  mt_tally <- sig_tally(
    organ,
    ref_genome = "BSgenome.Hsapiens.UCSC.hg38",
    useSyn = TRUE
  )
  
  mt_sig2 <- sig_auto_extract(mt_tally$nmf_matrix,
                              K0 = 10, nrun = 10,
                              strategy = "stable"
  )
  
  
  sim_v3 <- get_sig_similarity(mt_sig2, sig_db = "SBS")
  a<-sim_v3$similarity
  
  matrix<-get_sig_exposure(mt_sig2)
  colnames(matrix)<-c("Sample_ID",apply(a, 1, sim_name))
  map<-pheatmap::pheatmap(sim_v3$similarity)
  save_pheatmap_pdf(map,paste0("class.",organ_name,".pdf"))
  res<-merge.data.frame(matrix,sampleid_gene,by.x = "Sample_ID")
  res<-res%>%mutate(organ=organ_name)%>%
       mutate(clinic=as.integer(substr(res$Sample_ID,14,15)))%>%
    filter(clinic<10)
  write.csv(res,paste0("res.",organ_name,".csv"),row.names = F)
  result<-rbind.fill(result,res)
}

write.csv(result,"sample_id.sbs.organ.csv",row.names = F)

