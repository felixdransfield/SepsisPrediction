# HEADER ---------------------------------------------------------------------------------
#
# Author: Felix Dransfield
# Email:  felix.j.dransfield@kcl.ac.uk
# 
# Date: `r paste(Sys.Date())`
#
# Script Name: Sepsis SOM clustering
#
# Script Description: Evaluates static cohort information of sepsis patients (SOFA) and find the optimal number of clusters
#                     then uses hierchical clustering to label cases in each cluster.
# 
# ---------------------------------------------------------------------------------------

# Intial Script Setup ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Loading Libraries
library("kohonen")
library("tibble")
library("magrittr")
library("dplyr")
library("tidyverse")
library("factoextra")
library("NbClust")
library("here")
library("ggplot2")
rm(list=ls())

# Setting working directory (assumes required data files are in the same directory as the script)
here::here()

# Setting seed for replication
set.seed(7777)

# Helper functions
Modes <- function(x) {
  ux <- unique(x)
  tab <- tabulate(match(x, ux))
  ux[tab == max(tab)]
}

# SCRIPT --------------------------------------------------------------------------------


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# 1: Loading/Cleaning Data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Reading CSV file
sepsis = read_csv(here('FinalNonSepsisCohort.csv'), show_col_types=FALSE)
no.sepsis = read_csv(here('FinalSepsisCohort.csv'), show_col_types=FALSE)
patient.data = rbind(sepsis, no.sepsis)

# Checking for outliers visually with boxplots
patient.data %>%
  select(age:cns) %>%
  gather(Measure, Value) %>%
  ggplot(aes(x = " ", y = Value)) +
  geom_boxplot() +
  facet_wrap(~Measure, scales = "free_y") 

# Error occured in data retrieval where some people are recorded as being Age 300
# replacing these values with the average age value
cleaned.age <- patient.data %>% filter(age < 120)
patient.data <- patient.data %>% mutate(age = replace(age, age > 120, mean(cleaned.age$age)))

# Scaling Variables
patient.data$AgeUnscaled = patient.data$age
patient.data$age = as.vector(scale(patient.data$age))
patient.data$comorbidityUnscaled = patient.data$comorbidity
patient.data$comorbidity = as.vector(scale(patient.data$comorbidity))

# Creating Matrix of clustering data
clustering.data = cbind(patient.data$age, patient.data$gender, patient.data$comorbidity, patient.data$respiration, patient.data$coagulation, patient.data$liver, patient.data$renal, patient.data$cardiovascular, patient.data$cns)
colnames(clustering.data) = c("Age", "Gender", "Comorbidities", "Respiration", "Coagulation", "Liver", "Renal", "Cardiovascular", "CNS")
# Recoding NA to 0
clustering.data[is.na(clustering.data)]<-0
#Converting to Matrices
clustering.data = as.matrix(clustering.data)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 2: SOM Clustering ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Creating a output directory for figures
output <- paste("1.CLUSTERING OUTPUT")
dir.create(here(output), showWarnings = FALSE)

# Creating SOM grid
som_grid <- somgrid(xdim = 7, ydim=7, topo="hexagonal")
knox.som = kohonen::supersom(clustering.data, grid = som_grid, rlen=1100,alpha=c(0.05,0.01),keep.data = TRUE)

# Plotting clustering performance metrics
plot.types <- list("changes", "count", "dist.neighbours", "quality", "codes")
plot.titles <- list("Training Progress", "Node Counts", "SOM neighbour distances", "Clustering Quality", "Codes")
for (plot in seq_along(plot.types)){
    pdf(here(output, paste(plot.titles[plot], ".pdf")))
    plot(knox.som, type=as.character(plot.types[plot]), main=plot.titles[plot])
    dev.off()
}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 3: Hierarchical clustering with contiguity constraints ~~~~
# From: https://rpubs.com/erblast/SOM
# Preparation and calculating distances
codes = tibble( layers = colnames(knox.som$codes[[1]]), codes = knox.som$codes ) %>%
    mutate( codes = purrr::map(codes, as_tibble) ) %>%
    spread( key = layers, value = codes) %>%
    apply(1, bind_cols) %>%
    .[[1]] %>%
    as_tibble()

# Generate distance matrix for codes
dist_m = dist(codes) %>%
as.matrix()

# Generate seperate distance matrix for map location
dist_on_map = kohonen::unit.distances(som_grid)
dist_adj = dist_m ^ dist_on_map
    
# Displaying optimal number of clusters with Elbow method
pdf(here(output, 'optimalNoOfClustersElbow.pdf'))
factoextra::fviz_nbclust(dist_adj, factoextra::hcut, method = "wss", hc_method = 'ward.D2', k.max = 15)
dev.off()

# Displaying optimal number of clusters with Silhouette method
pdf(here(output, 'optimalNoOfClustersSilhouette.pdf'))
factoextra::fviz_nbclust(dist_adj, factoextra::hcut, method = "silhouette", hc_method = "ward.D2", k.max =  15)
dev.off()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# 4: Calculating optimal number of clusters: voting among many methods ~~~~
indexes = c( "wss","silhouette","gap", "kl", "ch", "hartigan", "ccc", "scott", "marriot", "trcovw", "tracew", "friedman", "rubin", "cindex", "db", "duda", "pseudot2", "beale", "ratkowsky")

results_nb = list()
safe_nb = purrr::safely(NbClust::NbClust)
# we will time the execution time of each step
best.number.of.clusters = vector()

for(i in 1:length(indexes) ){
    t = lubridate::now()
    nb = safe_nb(as.dist(dist_adj), distance = "manhattan", min.nc = 4, max.nc = 15, method = "ward.D2", index = indexes[i])
    print(paste("at index", i, " doing method: ",indexes[i]))
    results_nb[[i]] = nb
    best.number.of.clusters = c(best.number.of.clusters,nb$result$Best.nc[1])
}

# Final number of clusters selected
final.number.of.clusters = max(Modes(best.number.of.clusters))


df_clust = tibble( indexes = indexes, nb = results_nb) %>%
                    mutate( results = purrr::map(nb,'result'), error = purrr::map(nb, 'error'), is_ok = purrr::map_lgl(error, is_null))

df_clust_success = df_clust %>%
    filter( is_ok ) %>%
    mutate( names      = purrr::map(results, names)
                        ,all_index = purrr::map(results, 'All.index')
                        ,best_nc   = purrr::map(results, 'Best.nc')
                        ,best_nc   = purrr::map(best_nc, function(x) x[[1]])
                        ,is_ok     = !purrr::map_lgl(best_nc, is_null)) %>%
    filter(is_ok) %>%
    mutate( best_nc    = purrr::flatten_dbl(best_nc))

save(df_clust_success, file = here(output, 'NumberOfClustersVotes.Rdata'))

#plotting votes
pdf(here(output, 'VotesOnNoOfClusters.pdf'))
df_clust_success %>%
    filter(!is_null(best_nc) )%>%
    ggplot( aes(x = as.factor(best_nc))) + geom_bar()+
    labs(title = 'Votes on optimal number of clusters', x = 'Best No of Clusters')

dev.off()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#5: Clustering using hierarchical clustering ~~~~~~~~~~

dist_adj =  dist_m ^ dist_on_map
clust_adj = hclust(as.dist(dist_adj), 'ward.D2')

# Cluster using final.number.of.clusters clusters (generated by analysis above).
som_clusters = cutree(clust_adj, final.number.of.clusters) 
pdf(here(output, 'Clusters.pdf'))
plot(knox.som, type = "property", property=som_clusters,main="Clusters",palette.name=rainbow, heatkeywidth = 0.9)
add.cluster.boundaries(knox.som, som_clusters,lwd=3)
dev.off()

# get vector with cluster value for each original data sample
cluster_assignment = vector()
cluster_assignment <- som_clusters[knox.som$unit.classif]    #  knox.som$unit.classif is the som node for each data point in the original data.
# make the cluster assignment a column in the original data for ease of retrieval.
patient.data$cluster_assignment = cluster_assignment

# Writing CSV with clustering assignments
write.csv(patient.data, here(output, 'ClusteredDataDemographics.csv'))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#6: Plotting all features ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
clusters = list()
clusters[[1]] = som_clusters
i = 1
k=max(unique(som_clusters))

# Colour palettes for plotting
myPalette <- function(n=20, alpha = 1) { heat.colors(n,  alpha=alpha)[n:1] }

# Selecting the variables that will be ploteted
plotting.variables <- patient.data %>%
                      select(AgeUnscaled, gender, comorbidityUnscaled, sofa, respiration, 
                            coagulation, liver, renal, cardiovascular, cns
                            )

pdf(here('1.CLUSTERING OUTPUT', 'AllFeatures.pdf'))

layout(matrix(c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16),4,4,  byrow = TRUE))##, widths=c(3,3,3,3))

for (column in 1:ncol(plotting.variables)){
    tes <- aggregate(as.numeric(plotting.variables[[column]]), by=list(knox.som$unit.classif), FUN=mean, simplify=TRUE)
    names(tes) <- c("Node", "Value")
    plot(knox.som, type = "property", property=tes$Value,main="",palette.name=myPalette, heatkeywidth = 0.9)
    title(colnames(plotting.variables)[column], line=1)
    add.cluster.boundaries(knox.som, clusters[[i]],lwd=5)
}

dev.off()


# missingNodes <- which(!(seq(1,nrow(knox.som$codes[[1]])) %in% Age.unscaled$Node))
# names(Age.unscaled) = names(data.frame(Node=missingNodes, Value=NA))
# Age.unscaled<- rbind(Age.unscaled, data.frame(Node=missingNodes, Value=NA))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ---------------------------------------------------------------------------------------




