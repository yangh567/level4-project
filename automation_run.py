"""

This is the file used to automate all experiments as well as generating required data under the result folder

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

# TODO commented until you downloaded the sample_id.sbs.organ.csv processed file from link provided in manual.md or
#  generated yourself

# we first take only the samples in each cancers that have 244 driver gene label and corresponding cancer label to
# save memory
execute_generate_small_data = subprocess.run(["python", "generate_small_data.py"])
# we statistically analyse mutation frequency of each gene in each cancers for later finding the top frequently
# mutated driver gene in each cancer
execute_static_gene_prob = subprocess.run(["python", "static_gene_prob.py"])
# stratified sampling for gene mutation status in each cancer
execute_prepare_data = subprocess.run(["python", "prepared_data.py"])

print("\n")
print("The data and related graphs are generated")
print("please see the result graph under 'src/statistics/cross_valid_static'")
print(5*'\n')

os.chdir("..")
os.chdir("..")

# Experiment 1
os.chdir("src/classification_cancer_analysis")
# performing the cancer classification using sbs signatures
execute_classify_cancer = subprocess.run(["python", "classify_cancer_type_pytorch.py"])
# generate the heatmap to show the contribution of normalized sbs signatures in each cancer
execute_heatmap_generator = subprocess.run(["python", "heatmap_generator.py"])
# draw the top 10 sbs signatures contribution location in each cancer to form the idea of feeding those features to CNN
execute_top10_sbs_heatmap_generator = subprocess.run(["python", "heatmap_of_top_10_sbs.py"])
# find the similarity between cancers using their sbs signature weights (some share same sbs signature might be more
# similar) to analyse on why there are some mis classification
execute_cancer_cancer_similarity = subprocess.run(["python", "sbs_similarity_between_cancers.py"])

print("\n")
print("Experiment 1 finished")
print("please see the result graph under 'src/classification_cancer_analysis/result'")
print(5*'\n')

os.chdir("..")
os.chdir("..")


# Experiment 2
os.chdir("src/Simple-CNN-implement")
# classify the gene mutation status using simple CNN
execute_classify_gene_1 = subprocess.run(["python", "classify_gene_type.py"])

print("\n")
print("Experiment 2 finished")
print("please see the result graph under 'src/Simple-CNN-implement/result'")
print(5*'\n')

os.chdir("..")
os.chdir("..")

# Experiment 3
os.chdir("src/CNN-implement")
# classify the gene mutation status using complex CNN
execute_classify_gene_2 = subprocess.run(["python", "classify_gene_type.py"])

print("\n")
print("Experiment 3 finished")
print("please see the result graph under 'src/CNN-implement/result'")
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
print("please check the result and visualizations under 'src/classification_cancer_analysis/result'")
print("please check the result and visualizations under 'src/Simple-CNN-implement/result'")
print("please check the result and visualizations under 'src/CNN-implement/result'")