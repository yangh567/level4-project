# used for practices for the samples with genome reference set as hg19 (DEPRECATED)
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

library(TCGAmutations)
library(sigminer)
library(maftools)
library(dplyr)
library(data.table)
library(plyr)

# function to save heatmap
save_pheatmap_pdf <- function(x, filename, width=7, height=7) {
  stopifnot(!missing(x))
  stopifnot(!missing(filename))
  pdf(filename, width=width, height=height)
  grid::grid.newpage()
  grid::grid.draw(x$gtable)
  dev.off()
}

#
sim_name<-function(x){
  sbs_name<-colnames(t(x[x==max(x)]))
  return(sbs_name)
}

# read the files just for taking name and download the remote data from other's repository
path<-dir()[grep("gz",dir())]
result<-data.table()
for(k in path){
  organ_name<-unlist(strsplit(k,"[.]"))[2]
  print(organ_name)
  organ19<-tcga_load(organ_name)

  # see the clinic data here
  organ_clinic_data <- organ19@clinical.data
  
  organ_clinic_data $organ_name<-organ_name
  write.csv(organ_clinic_data,paste0("class.",organ_name,".csv"))
  
  organ19<-maftools::subsetMaf(organ19,tsb=organ19@variants.per.sample$Tumor_Sample_Barcode)
  mt_tally <- sig_tally(
    organ19,
    ref_genome = "BSgenome.Hsapiens.UCSC.hg19",
    useSyn = TRUE
  )
  
  mt_sig2 <- sig_auto_extract(mt_tally$nmf_matrix,
                              K0 = 90, nrun = 10,
                              strategy = "stable"
  )
  
  # find the signature similarities
  sim_v3 <- get_sig_similarity(mt_sig2, sig_db = "SBS")
  a<-sim_v3$similarity
  # obtain sbs signature exposure
  matrix<-get_sig_exposure(mt_sig2)
  colnames(matrix)<-c("Sample_ID",apply(a, 1, sim_name))
  map<-pheatmap::pheatmap(sim_v3$similarity)
  save_pheatmap_pdf(map,paste0("class.",organ_name,".pdf"))
  res<-matrix%>%mutate(organ=organ_name)
  result<-rbind.fill(result,res)
}
# sav ethe result
write.csv(result,"sample_id.sbs.organ19.csv",row.names = F)

