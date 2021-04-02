## Data
Keep the data (e.g. from evaluations here)


* All data used in experiments is free from ethical check

* Description of the data, and what is contained:

  * *`cross_valid`* -- Storing the cross validation and validation data extracted from `sample_id.sbs.organ.csv`using stratified sampling
  * *`similarity_heatmap`* -- Generate the similarity comparison graph for signatures we extracted and the sbs signatures from cosmic website
  * *`processed`*  -- Storing the R files to extract the matrix we needed from the TCGA maf files which can be downloaded below,Storing the R files to summarize graphs for the maf files,Storing the requirement.r to install the required library for doing Bayesian NMF  for gaining the sample_sbs.organ.csv from TCGA maf files
                   -- Storing the similarity data (sim.[cancer].csv) between extracted sbs signature and Cosmic SBS signatures as well as the heatmap for explanation(class_graphs)
    
  * *`processed/sample_id.sbs.organ.csv`* -- The Ultimate processed file we needed for later AI experiments
  * *`processed/summary_maf.[cancer].pdf`* -- Storing the summary graphs for each cancer's maf files
  * *`processed/class_graphs`* -- Storing the similarity heatmap between generated signatures and cosmic signatures
  * *`processed/maf.summary.plot.r`* -- The file is used to plot the maf summary graphs(summary of gene mutation frequencies,gene mutation types,etc)
  * *`processed/maf2sbs_change.fixID.r`* -- The version of maf to sbs that fixed the duplicated sbs signatures and generated the sbs signatures using non-negative matrix factorization
  * *`processed/requirement.r`* -- The requirements you needed to run the R codes for experiments
  * *`processed/code.R`* -- Used for generating the sample_id vs mutation type matrix(**DEPRECATED and DELETED**)
  * *`processed/experiment.R`* -- Used for practice on maftools(**DEPRECATED and DELETED**)
  * *`processed/maf2sbs.r`* --  Original maf file to sample_id x sbs signatures matrix(**DEPRECATED and DELETED**)
  * *`processed/NMF-SBS.R`* --  The research file tried to do the NMF on provided library of hg19 reference somatic mutation file(**DEPRECATED and DELETED**)
  * *`processed/organ-19.R`* -- Used for practices for the samples with genome reference set as hg19 (**DEPRECATED and DELETED**)
  * *`processed/swapPosition.R`* -- This is the file used to swap the columns for generated sample_id.sbs.organ.csv(**DEPRECATED and DELETED**)
  

* The raw data is too large to fit into repository,However,the link to download could be found here:
  
  * *For dowmloading maf files : 
  https://drive.google.com/drive/folders/1oDvK3zpNO8zhLv4k9I1woT0xthrKGdAW?usp=sharing*

  * *For downloading processed sbs x sample_id x gene mutation status matrix:
  https://drive.google.com/file/d/1kk0PercSGP2-vewuwpFat-9zs5fixhzm/view?usp=sharing*
    


