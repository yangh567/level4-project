
## Source

The whole repository contains:

*  `cancer-classify-gene` -- The validation experiment of using cancer type to classify on gene mutation status **(Deprecated Research,deleted)**


*  `classification-gene-feature-extraction` -- The validation experiment of classifying cancer types using sbs signatures and classify gene as well as analyse on ranking **(Deprecated Research,deleted)**


*  `classification_cancer_analysis` -- The validation experiment of classifying cancers and extract the useful signature for gene classification, and also analyse on cancer classification
*  `classification_cancer_analysis/result` -- Saving all of the visualizations and evaluation results for cancer classification
*  `classification_cancer_analysis/classify_cancer_type_pytorch.py` -- The evaluation file to validating the idea of classifying cancers using mutational signatures
*  `classification_cancer_analysis/heatmap_generator.py` -- The code to generate the heatmap to describe the contribution of different sbs signatures in each cancers
*  `classification_cancer_analysis/heatmap_of_top_10_sbs.py` -- This file is used to draw the heatmap for display the top 10 weighted sbs signatures in each cancer types to support the idea of using spatial features to conduct convolutional neural network
*  `classification_cancer_analysis/sbs_similarity_between_cancers.py` -- This file is used to investigate on why has the sbs signature has caused some of the samples of a class classified to other classes,the reason is that some of the classes are sharing the similar sbs signature exposures



*  `CNN-implement` -- The Convolutional neural network complex implementation for gene classification
*  `CNN-implement/result` -- The total visualizations and evaluation results for gene mutation status prediction
*  `CNN-implement/classify_gene_type.py` -- This file is used to test on the self-build complex CNN model on the classification_cancer_analysis of genes based on mutation signature (SBS) using 5 fold cross validation


*  `individual-classification` -- Validate the idea that sbs signatures can be used to identify two cancers to expand the idea that it can be used to classify more cancers
*  `individual-classification/result`-- Saving all of the visualizations and evaluation results for two cancer classification
*  `individual-classification/classify_BLCA-LGG.py`-- This file is used to test on the self-build model on classification_cancer_analysis of BLCA and BRCA based on mutation signature (SBS) using 5 fold cross validation to ensure the possibility of classifying 32 cancers
*  `individual-classification/normalization-of-sbs-weights.py`-- The normalization of sbs weights in BLCA and LGG
*  `individual-classification/read_npy.py`-- This file is used to draw the heatmap for display the weight of sbs signatures in two cancer types(BLCA,LGG)


*  `ml_method` -- The research on the HRDetect model **(Deprecated Research,deleted)**


*  `my_utilities` -- Contains the models,configurations,tools and tools for visualization
*  `my_utilities/my_confusion_matrix.py` -- This file is used to generate the confusion matrix for the multiclass classification_cancer_analysis
*  `my_utilities/my_model.py` -- This file is used to provide all of the models built
*  `my_utilities/my_tools.py` -- This file provides the tools for analytical visualization and label finding as well as feature extraction and etc



*  `preprocessing` -- The preprocessing of the data from HRDetect model **(Deprecated Research,deleted)**


*  `Simple-CNN-implement` -- The Convolutional neural network simple implementation for gene classification
*  `Simple-CNN-implement/result` -- The total visualizations and evaluation results for gene mutation status prediction
*  `Simple-CNN-implement/classify_gene_type.py` --  This file is used to test on the self-build simple CNN model on the classification_cancer_analysis of genes based on mutation signature (SBS) using 5 fold cross validation

*  `single_top_driver_gene_prediction` -- The validation experiment of classifying to frequently mutated driver gene using the feature extracted from cancer classification **(Deprecated Research,deleted)**


*  `statistics` -- Provides the preprocessing and stratified sampling of the raw data as well as doing statistic analysis on the gene mutation in each cancer
*  `statistics/cross_valid_static` -- Provides the visualization of the sample quantities in each fold
*  `statistics/gene_distribution` -- Storing the 224 driver gene's mutation frequency across all cancers
*  `statistics/small_data` -- Storing the extracted s224 driver gene across all cancers in all samples from the sample_id_sbs.organ.csv instead of all gene
*  `statistics/driver-gene-in-cancers.csv` -- Provides driver gene in each cancer (taken from paper table)
*  `statistics/finding-driver-gene.py` -- The code to find the list of driver gene in each cancer for future experiments
*  `statistics/generate_small_data.py` -- The code to extracted 224 driver gene across all cancers in all samples from the sample_id_sbs.organ.csv instead of all gene
*  `statistics/prepared_data.py` -- The code to perform the stratified sampling
*  `statistics/static_gene_prob.py` -- The code is used to statistically summarize number of mutation of the 224 driver gene in each cancer and express it in probability form



