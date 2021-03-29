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

# function to save the heatmap
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

# read maf files
path<-dir()[grep("gz",dir())]
result<-data.table()
for(k in path){
  organ_name<-unlist(strsplit(k,"[.]"))[2]
  print(organ_name)
  organ<-read.maf(k)
  b<-organ@data
  sampleid_gene<-as.data.frame.array(t(table(b$Hugo_Symbol,b$Tumor_Sample_Barcode)))
  sampleid_gene$Sample_ID<-row.names(sampleid_gene)

  # tally the 96 components
  mt_tally <- sig_tally(
    organ,
    ref_genome = "BSgenome.Hsapiens.UCSC.hg38",
    useSyn = TRUE
  )
  # extract the signatures with bayesian NMF
  mt_sig2 <- sig_auto_extract(mt_tally$nmf_matrix,
                              K0 = 10, nrun = 10,
                              strategy = "stable"
  )
  
  # obtain the signature similarities between extracted signatures and cosmic signatures
  sim_v3 <- get_sig_similarity(mt_sig2, sig_db = "SBS")
  a<-sim_v3$similarity

  # obtain the sbs signature exposure
  matrix<-get_sig_exposure(mt_sig2)
  # concatenate the sample id column to it
  colnames(matrix)<-c("Sample_ID",apply(a, 1, sim_name))
  # draw the similarity heatmap
  map<-pheatmap::pheatmap(sim_v3$similarity)
  save_pheatmap_pdf(map,paste0("class.",organ_name,".pdf"))

  # merge the data generated for each cancer classes
  res<-merge.data.frame(matrix,sampleid_gene,by.x = "Sample_ID")
  res<-res%>%mutate(organ=organ_name)%>%
       mutate(clinic=as.integer(substr(res$Sample_ID,14,15)))%>%
    # filter out those that are not tumour patients
    filter(clinic<10)
  # save into csv
  write.csv(res,paste0("res.",organ_name,".csv"),row.names = F)
  result<-rbind.fill(result,res)
}
 # save the processed file
write.csv(result,"sample_id.sbs.organ.csv",row.names = F)

