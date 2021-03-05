# used to append pdf generation and failed to remove the duplicated columns
rm(list = ls())
### step1 install packages--------
if(F){
install.packages("devtools")
install.packages("remotes")
library("devtools")
chooseBioCmirror()
BiocManager::install("sigminer")
BiocManager::install("BSgenome.Hsapiens.UCSC.hg19")
BiocManager::install("BSgenome.Hsapiens.UCSC.hg38")
BiocManager::install("maftools")
BiocManager::install("PoisonAlien/TCGAmutations")
} 

### step2 library packages & function--------
library(sigminer)
library(maftools)
library(dplyr)
library(data.table)
library(plyr)
library(tidyr)
library(ggplot2)
save_pheatmap_pdf <- function(x, filename, width=20, height=20 ){
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

### step3 load data & do some analysis--------
path<-dir()[grep("gz",dir())]
result<-data.table()
for(k in path[1:3]){
  #k=path[1]
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
  
  mt_sig <- sig_extract(mt_tally$nmf_matrix,
                        n_sig = nrow(a),
                        nrun = 10,
                        cores = 4
  )
  pdf(file=paste0("consensus.",organ_name,".pdf"))
  show_sig_consensusmap(mt_sig)
  dev.off()
  
  show_sig_profile(mt_sig, mode = "SBS", style = "cosmic", x_label_angle = 90)
  ggsave(paste0("sbs.",organ_name,".pdf"))
  freq<-mt_sig$Signature.norm
  write.csv(freq,paste0("freq.",organ_name,".csv"),row.names = T)
  
  matrix<-get_sig_exposure(mt_sig2)
  colnames(matrix)<-c("Sample_ID",apply(a, 1, sim_name))
  sim<-t(sim_v3$similarity)
  write.csv(sim,paste0("sim.",organ_name,".csv"),row.names = T)
  
  map3<-pheatmap::pheatmap(sim_v3$similarity)
  save_pheatmap_pdf(map3,paste0("class.",organ_name,".pdf"))
  
  res<-merge.data.frame(matrix,sampleid_gene,by.x = "Sample_ID")
  res<-res%>%mutate(organ=organ_name)%>%
       mutate(clinic=as.integer(substr(res$Sample_ID,14,15)))%>%
    filter(clinic<10)
  write.csv(res,paste0("res.",organ_name,".csv"),row.names = F)
  result<-rbind.fill(result,res)
}

write.csv(result,"sample_id.sbs.organ.csv",row.names = F)
