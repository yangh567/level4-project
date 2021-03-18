# User manual 

### The project is research type.It examined the classification performance of cancers and gene mutation status using single-based-substitution signatures.

### Part 1 : To obtain data:

* *1. you can skip this part as the data is separated and stored under "data/cross_valid"*
   

      The processed file is already generated using the R script called maf2sbs_change.fixID.r with the TCGA maf files for each cancer types,all running step will be based on the cross validation data set and validation dataset file generated from the processed file generated in 
      "data/processed" folder called "sample_id.sbs.organ.csv", The reason why those maf file and "sample_id.sbs.organ.csv"were not uploaded is that the 32 TCGA maf files are too large to put into directory also,the generation of sbs signatures by sample_id matrix using 
      non-negative matrix factorization requires several hours to complete, also, the similarity of the NMF generated signatures, and the sbs signatures from Cosmic website is also provided as sim.cancer_type.csv to help with visualization.For the below tasks, 
      you can just used the cross_validation data set and validation data set under "data/cross_valid" to do the job.



      (note: However,if you want to run maf2sbs_change.fixID.r, you will need to obtain TCGA maf format files for different cancers from https://portal.gdc.cancer.gov/repository or here:
   
      https://drive.google.com/drive/folders/1oDvK3zpNO8zhLv4k9I1woT0xthrKGdAW?usp=sharing
   
      and put them at data/processed folder,then,get into processed folder and run "sudo apt install r-base-core" to install Rscript and run "Rscript requirement.r" to install necessary packages and "Rscript maf2sbs_change.fixID.r" to generate the "sample_id.sbs.organ.csv" file 
      which you can find here:
   
      https://drive.google.com/file/d/1kk0PercSGP2-vewuwpFat-9zs5fixhzm/view?usp=sharing.
   
      (you will need to install devtools,remotes,BiocManager,sigminer,BSgenome.Hsapiens.UCSC.hg38,maftools and PoisonAlien/TCGAmutations)
      
      also,if you have used r to generate "sample_id.sbs.organ.csv",in order to generate the cross_validation data set and validation data set for validation, 
      you will need to run the prepared_data.py under "src/statistics/" to generate the stratified sampled files for performing 
      5-fold cross validation for evaluation (5 validation data set and 1 validation dataset)using command "python prepared_data.py".)


* *2. Run heatmap_similarity.py under "data/similarity_heatmap" to visualize the similarity of mutational signatures and store the file at "processed/class_graphs" folder using command "python heatmap_similarity.py".*

   
* *3. Run "python generate_small_data.py" under "src/statistics" to obtain the matrix only contains the gene mutation status of the driver gene we need to investigate on*


* *4. Run the static_gene_prob.py under "src/statistics" directory using "python statistic_gene_prob.py" to get the mutation frequency of each gene in each cancer to help with extracting the frequently mutated driver gene in each cancer for future experiments.* 



### Part 2 : Experiment 1 : To see the classification result of gene without using feature extraction

- **(This is just instruction of running the research files to investigate on the analysis of cancer classification and gene mutation status classification as well as the ranking of gene in each cancer)**
 
* *1. Run "python classify_cancer_type_pytorch.py" under "src/classification_cancer_gene_analysis" to performing the training of classifier on 
   the cancer and evaluate on the model.*
  
   
* *2. Run "python classify_gene_type.py" under "src/classification_cancer_gene_analysis" to performing the training of classifier on 
   the gene mutation status and evaluate on the model.*
  
   
* *3. Run "python analysis.py" under "src/classification_cancer_gene_analysis" to normalize the selected weights and performing ranking top 5 frequently mutated gene for each cancer types.*
   by finding intersections.*
  

* *4. Run "python heatmap_generator.py" under "src/classification_cancer_gene_analysis" to generate the heatmap of the sbs signature weights in each cancer and each gene for this experiment.*
   
   


   
### Part 3 : Experiment 2 : To see the classification result of gene using the feature extraction of sbs sig weights obtained from cancer classification

- **(This is instruction of running the research file of using top 10 weighted sbs signature in each cancer to classify on the top 5 frequently mutated driver gene in that cancer)**

* *1. Run "python classify_gene_type.py" under "src/classification-gene-feature-extraction" to performing the training of classifier on 
   the gene mutation status and evaluate on the model as well as extracting signature's weights".*
  
   
* *3. Run "python normalization_gene_cancer.py" under "src/classification-gene-feature-extraction" to normalize the selected weights for each gene and each cancer types.*


* *4. Run "python heatmap_ploter.py" under "src/classification-gene-feature-extraction" to generate the heatmaps of the sbs signature weights in each cancer and each gene.*



   


### Part 4 : Experiment 3 : To see the classification result of top 1 frequently mutated driver gene in specific cancer using feature extraction of sbs sig weights obtained from cancer classification

- **(This is instruction of running the research file of using top 10 weighted sbs signature in the cancer to classify on the top 1 frequently mutated driver gene in that cancer to validate the idea that sbs signatures might be infeasible to predict on the gene mutation status in that cancer)** 


* *1. Run "python classify_gene_type.py" under "src/single_top_driver_gene_prediction" to performing the training of classifier on 
   the single top driver gene's mutation status and evaluate on the model as well as extracting signature weights.*
   

