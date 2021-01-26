# this is the file to generate the csv file of (sample x sbs signatures) matrix of
# all cancers using bayesian extractor




rm(list = ls())
install.packages("devtools")
install.packages("remotes")
install.packages("pheatmap")
library("devtools")
chooseBioCmirror()
BiocManager::install("sigminer", dependencies = TRUE)
BiocManager::install("BSgenome.Hsapiens.UCSC.hg19")
BiocManager::install("BSgenome.Hsapiens.UCSC.hg38")
BiocManager::install("maftools")
BiocManager::install("PoisonAlien/TCGAmutations")
library(pheatmap)


library(sigminer)
library(maftools)
library(dplyr)
library(data.table)
library(plyr)
save_pheatmap_pdf <- function(x, filename, width=7, height=7) {
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
  organ_name<-unlist(strsplit(k,"[.]"))[2]
  print(organ_name)
  organ<-read.maf(k)
  
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
  res<-matrix%>%mutate(organ=organ_name)
  result<-rbind.fill(result,res)
}

write.csv(result,"sample_id.sbs.organ.csv",row.names = F)
