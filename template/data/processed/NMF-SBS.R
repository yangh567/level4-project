BiocManager::install("maftools")
BiocManager::install("sigminer")
#BiocManager::install("PoisonAlien/TCGAmutations")
BiocManager::install("BSgenome.Hsapiens.UCSC.hg38")
BiocManager::install("BSgenome.Hsapiens.UCSC.hg19")
library(sigminer)

library(TCGAmutations)
laml.maf<-system.file("C:/Users/23590/Desktop/SBS","TCGA.LAML.muse.0cdf3c70-ad58-462d-b6ba-5004b26c618e.DR-10.0.somatic.maf.gz",package = "maftools",mustWork = T)
library(maftools)

test<-read.maf(maf="TCGA.LAML.muse.0cdf3c70-ad58-462d-b6ba-5004b26c618e.DR-10.0.somatic.maf.gz")
head(test@data)



mt_tally$nmf_matrix[1:5,1:5]

set.seed(1234)
brca <- tcga_load("BRCA")
brca <- maftools::subsetMaf(brca,
                            tsb = as.character(sample(brca@variants.per.sample$Tumor_Sample_Barcode, 100))
)
saveRDS(brca, file = "brca.rds")

brca<-readRDS("brca.rds")
mt_tally <- sig_tally(
  test,
  ref_genome = "BSgenome.Hsapiens.UCSC.hg38",
  useSyn = TRUE
)

str(mt_tally$all_matrices,max.level = 1)


mt_tally_ALL <- sig_tally(
  brca,
  ref_genome = "BSgenome.Hsapiens.UCSC.hg19",
  useSyn = TRUE,
  mode = "ALL",
  add_trans_bias = TRUE
)


#str(mt_tally_ALL$all_matrices,max.level = 1) look into the structure

library(NMF)


mt_est <- sig_estimate(mt_tally$nmf_matrix,
                       range = 2:6,
                       nrun = 10, # increase this value if you wana a more stable estimation
                       use_random = FALSE, # if TRUE, add results from randomized input
                       cores = 4,
                       #pConstant = 1e-13,
                       verbose = TRUE
)

show_sig_number_survey2(mt_est$survey)
#library(dplyr)
#res3<-data.table(t(res2))
#res4<-res3%>%mutate(organ="brca")
#head(res3)
#rbind(res,res2)