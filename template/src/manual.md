# User manual 

The project is research type on examining the classification performance of using single-based-substitution signatures

to see the result:

1. The processed file is already generated using the R script called maf2sbs_change.fixID.r 
   with the TCGA maf files for each cancer types,all running step will be based on the file 
   generated in "data/processed" folder called "sample_id.sbs.organ.csv", The reason is that the 
   32 TCGA maf files are too large to put into directory also,the generation of sbs signatures 
   by sample_id matrix using non-negative matrix factorization requires several hours to complete,
   also, the similarity of the NMF generated signatures, and the sbs signatures from Cosmic website 
   is also provided as sim.cancer_type.csv to help with visualization.
   
2. run heatmap_similarity.py under "data/similarity" to visualize the similarity of mutational signatures
   in "processed/class_graphs" folder using command "python heatmap_similarity.py".  
   
3. run the static_gene_prob.py under "src/statistics" directory using 
   "python statistic_gene_prob.py" to get the mutation frequency of each gene 
   in each cancer to help with extracting the type of gene that are not mutated in specific cancer. 

4. run the prepared_data.py under "src/classification/" to generate the stratified sampled files 
   for performing 5-fold cross validation for evaluation includes(5 validation data set and 1 validation dataset)
   using command "python prepared_data.py".
   
5. run classify_cancer_type_pytorch.py under "src/classification/" to performing the training of classifier on 
   the cancer and evaluate on the model using "python classify_cancer_type_pytorch.py".
   
6. run classify_gene_type.py under "src/classification/" to performing the training of classifier on 
   the gene and evaluate on the model using "python classify_gene_type.py".
   
7. run analysis.py under "src/classification/" to performing finding top 10 gene for each cancer types
   using "python analysis.py".

8. run read_npy.py to generate the heatmap of the sbs signature weights in each cancer and each gene
   using "python read_npy.py".
   
   

