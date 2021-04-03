# User manual 

### 1. The Description

#### The project is research type.It examined the classification performance of cancers and gene mutation status using single-based-substitution signatures.



#### (Note here, The automation for experiments is made , simply run `python automation_run.py` to finish all the steps below, or you can do it separately according to steps below, it takes some times to finish all experiments as the complexity of CNN and there are 5 fold for 32 cancers,which means 32 models in each fold, so, please wait)
#### (Also, note that the line to perform most frequently mutated driver gene,stratified sampling is commented unless you have got the sample_id.sbs.organ.csv file)

### 2. The Steps to Run The Code Manually
#### Part 1 : To obtain data:

* *1. you could skip this part as the data is separated by stratified sampling and stored under `data/cross_valid`, if you don't want to perform stratified sampling and generate data again*
   

      The processed file is already generated using the R script called "maf2sbs_change.fixID.r" with the TCGA maf files for each cancer types,all running step will be based on the cross validation data set and validation dataset file generated from the processed file generated in 
      "data/processed" folder called "sample_id.sbs.organ.csv", The reason why those maf file and sample_id.sbs.organ.csv were not uploaded is that the 32 TCGA maf files are too large to put into directory also,the generation of sbs signatures by sample_id matrix using 
      non-negative matrix factorization in R requires several hours to complete, also, the similarity file of the NMF generated signatures, and the sbs signatures from Cosmic website is also provided as "sim.cancer_type.csv" to help with visualization.For the below tasks, 
      you can just used the cross_validation data set and validation data set under "data/cross_valid" to see the results or download the "sample_id.sbs.organ.csv" from the link provided below.



      (Note: However,if you want to run maf2sbs_change.fixID.r to generate your own sbs x sample_id x gene mutation status x organ(sample_id.sbs.organ.csv) file, you will need to obtain TCGA maf format files for different cancers from https://portal.gdc.cancer.gov/repository or here:
   
      https://drive.google.com/drive/folders/1oDvK3zpNO8zhLv4k9I1woT0xthrKGdAW?usp=sharing
   
      and put them at "data/processed/" folder,then,get into "processed" folder and run "sudo apt install r-base-core" to install Rscript and run "Rscript requirement.r" to install necessary packages and "Rscript maf2sbs_change.fixID.r" to generate the "sample_id.sbs.organ.csv" file 
      which you can find here without running the procedures above:
   
      https://drive.google.com/file/d/1kk0PercSGP2-vewuwpFat-9zs5fixhzm/view?usp=sharing.
   
      (you will need to install devtools,remotes,BiocManager,sigminer,BSgenome.Hsapiens.UCSC.hg38,maftools and PoisonAlien/TCGAmutations)
      
      (# Note: you will need to keep "sample_id.sbs.organ.csv" under "data/processed" to use in the later AI processes)


* *2. When you have downloaded the `sample_id.sbs.organ.csv` from the provided link or generated yourself using R codes and ensured it is in `data/processed`.*

  
* *3. Run heatmap_similarity.py under `data/similarity_heatmap` to visualize the similarity of generated signatures and cosmic mutational signatures store the file at `processed/class_graphs` folder using command `python heatmap_similarity.py`.*

   
* *4. Run `python generate_small_data.py` under `src/statistics` to obtain the matrix only contains the gene mutation status of the driver gene we need to investigate on*


* *5. Run the static_gene_prob.py under `src/statistics` directory using `python statistic_gene_prob.py` to get the mutation frequency of each gene in each cancer to help with extracting the frequently mutated driver gene in each cancer for future experiments.* 


* *6. Run the prepared_data.py under `src/statistics` directory using `python prepared_data.py` to performing stratified sampling to obtain the cross validation data for evaluation and validation data for validation*


#### Part 2 : Experiment 1 : To see the classification result of cancer 

- **(This is just instruction of running the research files to investigate on the analysis of cancer classification)**
 
* *1. Run `python classify_cancer_type_pytorch.py` under `src/classification_cancer_gene_analysis` to performing the training of classifier on 
   the cancer and evaluate on the model.*
  

* *2. Run `python heatmap_generator.py` under `src/classification_cancer_analysis` to generate the heatmap of the normalized sbs signature weights in each cancer.*


* *3. Run `python heatmap_of_top_10_sbs.py` under `src/classification_cancer_analysis` to draw the top 10 sbs signatures contribution location in each cancer to support the idea of feeding those features to CNN.*
   
   
* *4. Run `python sbs_similarity_between_cancers.py` under `src/classification_cancer_analysis` to find the similarity between cancers using their sbs signature weights (some cancers shared same sbs signature might be more similar) to analyse on why there are some mis-classifications.*


   
#### Part 3 : Experiment 2 : To see the classification result of top 1 frequently mutated driver gene using the feature extraction of sbs sig weights obtained from cancer classification as well as the simple CNN implementation

- **(This is instruction of running the research file of using top 10 weighted sbs signature in each cancer to classify on the top 1 frequently mutated driver gene in that cancer)**

* *1. Run `python classify_gene_type.py` under `src/Simple-CNN-implement` to performing the training of classifier on 
   the gene mutation status and evaluate on the model.*


  

#### Part 4 : Experiment 3 : To see the classification result of top 1 frequently mutated driver gene in specific cancer using feature extraction of sbs sig weights obtained from cancer classification as well as using the complex CNN implementation

- **(This is instruction of running the research file of using top 10 weighted sbs signature in the cancer to classify on the top 1 frequently mutated driver gene in that cancer to validate the idea that sbs signatures might be infeasible to predict on the gene mutation status in that cancer)** 


* *1. Run `python classify_gene_type.py` under `src/CNN-implement` to performing the training of classifier on 
   the single top driver gene's mutation status and evaluate on the model as well as extracting signature weights.*
   

