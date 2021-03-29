### step1 install packages--------
url<-"https://www.stats.bris.ac.uk/R/"
if(T){
  install.packages("devtools",repos = url)
  install.packages("remotes",repos = url)
  install.packages("BiocManager",repos = url)
  library("devtools")
  #chooseBioCmirror()
  BiocManager::install("sigminer")
  BiocManager::install("BSgenome.Hsapiens.UCSC.hg38")
  BiocManager::install("maftools")
  install.packages("data.table",repos = url)
  install.packages("dplyr",repos = url)
  install.packages("ggplot2",repos = url)
  BiocManager::install("PoisonAlien/TCGAmutations")
}