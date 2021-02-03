BiocManager::install("maftools")
BiocManager::install("sigminer")
#BiocManager::install("PoisonAlien/TCGAmutations")
BiocManager::install("BSgenome.Hsapiens.UCSC.hg38")
BiocManager::install("BSgenome.Hsapiens.UCSC.hg19")
BiocManager::install("BSgenome.Hsapiens.NCBI.GRCh38")


install.packages("devtools")
install.packages("reticulate")

library(sigminer)
library(maftools)
library(NMF)

data<-readRDS("C:\\Users\\23590\\Desktop\\SBS\\TCGA_MutSig_gene01.rds")

data

write.csv(data,"C:\\Users\\23590\\Desktop\\SBS\\TestResult.csv",row.names = F)