# this is the experiment file used to extract the file downloaded from tcga platform

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


# Read the file from the maf file

laml<-read.maf(maf="TCGA.LAML.muse.0cdf3c70-ad58-462d-b6ba-5004b26c618e.DR-10.0.somatic.maf.gz")
head(laml@data)

# generate the sample-by-component matrix
mt_tally_LAML <- sig_tally(
  laml,
  ref_genome = "BSgenome.Hsapiens.UCSC.hg38",
  useSyn = TRUE
)
# print the sample_gene mutation matrix (The most used matrix is stored in nmf_matrix ,all matrix are stored in all_matrices)
mt_tally_LAML$nmf_matrix
str(mt_tally_LAML$all_matrices,max.level = 1)
#the sample_gene mutation matrix(generated with 96 mutations)
mt_tally_LAML$all_matrices[2]


# manully try to extract the sbs

'''mt_LAML_est <- sig_estimate(mt_tally_LAML$nmf_matrix,
                            range = 2:10,
                            nrun = 10,
                            use_random = FALSE,
                            cores = 4,
                            verbose = TRUE
                            )

show_sig_number_survey2(mt_LAML_est$survey)'''

# automatically extract feature sbs using sigProfile

# first ,we ensure the R interpreter recognizes the path to python installation
'''
library("reticulate")
use_python("C:\\Users\\23590\\Anaconda3\\python36")
py_config()


library("devtools")
install_github("AlexandrovLab/SigProfilerExtractorR")
library("SigProfilerExtractorR")

help("sigprofilerextractor")
'''

mt_LAML_sig2 <- sig_auto_extract(mt_tally_LAML$nmf_matrix,
                                 nrun = 10,
                                 strategy = "stable"
                                 )
#show us the info of each run
knitr::kable(mt_LAML_sig2$Raw$summary_run)

# see the similarity
sim <- get_sig_similarity(mt_LAML_sig2,sig_db = "SBS")
#The signature matrix
sig_signature(mt_LAML_sig2)
#get the signature exposure matrix
get_sig_exposure(mt_LAML_sig2)
