# the version of maf to sbs that fixed the duplicated sbs signatures
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

# the function to save the generated pdf files
save_pheatmap_pdf <- function(x, filename, width=20, height=20 ){
  stopifnot(!missing(x))
  stopifnot(!missing(filename))
  pdf(filename, width=width, height=height)
  grid::grid.newpage()
  grid::grid.draw(x$gtable)
  dev.off()
}

# get the columns that have the max similarity of sbs signatures
sim_name<-function(x){
  sbs_name<-colnames(t(x[x==max(x)]))
  return(sbs_name)
  }

### step3 load data & do some analysis--------
path<-dir()[grep("gz",dir())]
result<-data.table()
for(k in path){
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
  # tally the 96 components
  mt_sig2 <- sig_auto_extract(mt_tally$nmf_matrix,
                              K0 = 10, nrun = 10,
                              strategy = "stable"
  )

  # obtain the signature similarities between extracted signatures and cosmic signatures
  sim_v3 <- get_sig_similarity(mt_sig2, sig_db = "SBS")
  a<-sim_v3$similarity

  # extract the signatures with bayesian NMF
  mt_sig <- sig_extract(mt_tally$nmf_matrix,
                        n_sig = nrow(a),
                        nrun = 10,
                        cores = 4
  )
  
  # we don't draw the graph here
  if(F){
    pdf(file=paste0("consensus.",organ_name,".pdf"))
    show_sig_consensusmap(mt_sig)
    dev.off()
    
    show_sig_profile(mt_sig, mode = "SBS", style = "cosmic", x_label_angle = 90)
    ggsave(paste0("sbs.",organ_name,".pdf"))
    
    map3<-pheatmap::pheatmap(sim_v3$similarity)
    save_pheatmap_pdf(map3,paste0("class.",organ_name,".pdf"))
  }

  # obtain the similarity
  freq<-mt_sig$Signature.norm
  write.csv(freq,paste0("freq.",organ_name,".csv"),row.names = T)
  
  sim<-t(sim_v3$similarity)
  write.csv(sim,paste0("sim.",organ_name,".csv"),row.names = T)

  # get signature exposure
  matrix<-get_sig_exposure(mt_sig2)
  id<-apply(a, 1, sim_name)
  
  # eliminating the dupicates and concatenate the sample id column
  if(length(id[duplicated(id)])!=0){
  
  test<-sim[,id==id[duplicated(id)]]
  
  pos<-arrayInd(order(test,decreasing=TRUE)[1],dim(test))
  rr<-row.names(test)[pos[1,1]]
  cc<-colnames(test)[pos[1,2]]
  pass<-colnames(test)[colnames(test)!=cc]
  a2<-a[which(row.names(a)!=pass),]
  name2<-c("sample",row.names(a2))
  
  matrix1<-matrix%>%select(-colnames(test))
  matrix2<-matrix%>%select(cc)
  matrix3<-data.table(matrix1,matrix2)%>%select(name2)
  colnames(matrix3)<-c("Sample_ID",apply(a2, 1, sim_name))
  } else{
    matrix3<-matrix
    colnames(matrix3)<-c("Sample_ID",apply(a, 1, sim_name))
  }
  # merge the gene mutation status
  res<-merge.data.frame(matrix3,sampleid_gene,by.x = "Sample_ID",all = F)
  res<-res%>%mutate(organ=organ_name)%>%
       mutate(clinic=as.integer(substr(res$Sample_ID,14,15)))%>%
    # filter out the non-patient samples
    filter(clinic<10)
  fwrite(res,paste0("res.",organ_name,".csv"),row.names = F)
  result<-rbind.fill(result,res)
}
# swap the column position here
sbs_name<-colnames(result)[grep("SBS",colnames(result))]
sample.sbs<-c("Sample_ID",sbs_name)
sampleid<-result%>%select("Sample_ID")
SBS<-result%>%select(sbs_name)
gene<-result%>%select(-sample.sbs)

final.data<-data.table(sampleid,SBS,gene)
colnames(final.data)

# obtain the final file
fwrite(final.data,"sample_id.sbs.organ.csv",row.names = F)

