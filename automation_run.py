"""

This is the file used to automate all experiments as well as generating required data

"""
import os
import subprocess
import time

start = time.time()
# display the similarity between the signature and the cosmic signatures
os.chdir('data/similarity_heatmap')
execute_heatmap_sim = subprocess.run(["python", "heatmap_similarity.py"])
print("\n")
print("please see the result graph under 'data/processed/class_graphs'")
print(5*'\n')

os.chdir("..")
os.chdir("..")

# preparing data
os.chdir("src/statistics")
# commented until you downloaded the sample_id.sbs.organ.csv processed file from link provided in manual,md
# execute_prepare_data = subprocess.run(["python", "prepared_data.py"])
execute_generate_small_data = subprocess.run(["python", "generate_small_data.py"])
execute_static_gene_prob = subprocess.run(["python", "static_gene_prob.py"])

print("\n")
print("The data and related graphs are generated")
print("please see the result graph under 'src/statistics/cross_valid_static'")
print(5*'\n')

os.chdir("..")
os.chdir("..")

# Experiment 1
os.chdir("src/classification_cancer_gene_analysis")
execute_classify_cancer = subprocess.run(["python", "classify_cancer_type_pytorch.py"])
execute_classify_gene = subprocess.run(["python", "classify_gene_type.py"])
execute_analysis = subprocess.run(["python", "analysis.py"])
execute_heatmap_generator = subprocess.run(["python", "heatmap_generator.py"])

print("\n")
print("Experiment 1 finished")
print("please see the result graph under 'src/classification_cancer_gene_analysis/result'")
print(5*'\n')

os.chdir("..")
os.chdir("..")

# Experiment 2
os.chdir("src/classification-gene-feature-extraction")
execute_classify_gene_2 = subprocess.run(["python", "classify_gene_type.py"])
execute_normalization_2 = subprocess.run(["python", "normalization_gene_cancer.py"])
execute_heatmap_ploter_2 = subprocess.run(["python", "heatmap_ploter.py"])

print("\n")
print("Experiment 2 finished")
print("please see the result graph under 'src/classification-gene-feature-extraction/result'")
print(5*'\n')

os.chdir("..")
os.chdir("..")

# Experiment 3
os.chdir("src/single_top_driver_gene_prediction")
execute_classify_gene_3 = subprocess.run(["python", "classify_gene_type.py"])

print("\n")
print("Experiment 3 finished")
print("please see the result graph under 'src/single_top_driver_gene_prediction/result'")
print(5*'\n')

os.chdir("..")
os.chdir("..")

# finished
print("The automation run finished")

end = time.time()

# print the time interval
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("The total to run the experiments is : {:0>2}hr:{:0>2}min:{:05.2f}s".format(int(hours),int(minutes),seconds))

# reminding here to tell you where to look for results
print("Here is the reminding")
print("please check the result and visualizations under 'data/processed/class_graphs'")
print("please check the result and visualizations under 'src/statistics/cross_valid_static'")
print("please check the result and visualizations under 'src/classification_cancer_gene_analysis/result'")
print("please check the result and visualizations under 'src/classification-gene-feature-extraction/result'")
print("please check the result and visualizations under 'src/single_top_driver_gene_prediction/result'")