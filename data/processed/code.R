#install.packages("BiocManager")
# used for generating the sample_id vs mutation type matrix
options(warn =-1)
library(BiocManager)
#chooseBioCmirror() ## pick mirror where you are

### install some useful packages
if(F){
BiocManager::install("BSgenome.Hsapiens.UCSC.hg37")
BiocManager::install("GenomeInfoDbData")
BiocManager::install("data.table")
BiocManager::install("future.apply")
BiocManager::install("SomaticSignatures")
}



library("BSgenome.Hsapiens.UCSC.hg38")
library(data.table)
library(future.apply)

genome <- BSgenome.Hsapiens.UCSC.hg38
head(seqlengths(genome))

## load demo date

path<-dir()[grep("tsv",dir())]

for(k in path){
  print(k)
  data<-fread(k)

head(data)

if(F){
test<-c()
geneall<-c()

for(i in 1:500){
a<-as.character(data[i,3])
b<-as.integer(data[i,4])
c<-as.integer(data[i,5])
gene<-as.character(data[i,2])
before<-as.character(genome[[a]][(b-1)])
end<-as.character(genome[[a]][(c+1)])
ref<-as.character(data[i,6])
alt<-as.character(data[i,7])
type<-paste0(ref,">",alt)
type
type_1<-c("C>A","G>T")
type_2<-c("C>G","G>C")
type_3<-c("C>T","G>A")
type_4<-c("T>A","A>T")
type_5<-c("T>C","A>G")
type_6<-c("T>G","A>C")


if (type%in%type_1){
  type="C>A"
} else if(type%in%type_2){
  type="C>G"
} else if(type%in%type_3){
  type="C>T"
} else if(type%in%type_4){
  type="T>A"
} else if(type%in%type_5){
  type="T>C"
} else {
  type="T>G"
}

type<-paste0(before,"[",type,"]",end)
  test<-c(test,type)
  geneall<-c(geneall,gene)
}
head(date)
date<-data.frame(test,geneall)
date<-melt(date)

res<-table(date)
}


sum_sbs<-function(x){
  a<-as.character(x[3])
  b<-as.integer(x[4])
  c<-as.integer(x[5])
  gene<-as.character(x[2])
  before<-as.character(genome[[a]][(b-1)])
  end<-as.character(genome[[a]][(c+1)])
  ref<-as.character(x[6])
  alt<-as.character(x[7])
  type<-paste0(ref,">",alt)
  
  ## there are 6 type of basepair
  type_1<-c("C>A","G>T")
  type_2<-c("C>G","G>C")
  type_3<-c("C>T","G>A")
  type_4<-c("T>A","A>T")
  type_5<-c("T>C","A>G")
  type_6<-c("T>G","A>C")
  
  ## figures out  which type gene SNV belongs to
  if (type%in%type_1){
    type="C>A"
  } else if(type%in%type_2){
    type="C>G"
  } else if(type%in%type_3){
    type="C>T"
  } else if(type%in%type_4){
    type="T>A"
  } else if(type%in%type_5){
    type="T>C"
  } else {
    type="T>G"
  }
  
  type<-paste0(before,"[",type,"]",end)
  out<-list(type,gene)
  return(out)
}

#plan(multisession)
res<-apply(data[1:5000,], 1, sum_sbs) ## statistic SBS with gene 
print("finish sum!!")
res2<-as.data.frame.matrix(table(rbindlist(res)))

name<-paste0(paste(unlist(strsplit(k,"[.]"))[1:2],collapse = "." ),".csv")
## output data
write.csv(res2,name,row.names = T)
}
