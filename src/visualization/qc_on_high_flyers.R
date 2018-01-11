library(data.table)
library(ggplot2)
library(ggridges)
library(gridExtra)

### geometric mean impl taken from SO
gm_mean = function(x, na.rm=TRUE){
  exp(sum(log(x[x > 0]), na.rm=na.rm) / length(x))
}

### plot normalized peak heights for those peaks associated with high flyer genes
peaks_dir="/Users/zamparol/projects/SeqDemote/data/ATAC/corces_heme/peaks"

# load the list of annotated peaks, select just those for this lisf of high flyers
hf_vec = c("FOXP1", "TBC1D5", "AUTS2", "RAD51B", "FHIT", "RUNX1", "PTPRD", "PDE4D", "PRKG1", "LRBA", "ZBTB20", "ARID1B", "PLCB1")
annotated_atlas = data.table(read.csv(paste(peaks_dir,"annotated_atlas_reduced.csv",sep="/")))
hf_peaks = annotated_atlas[nearest.gene %in% hf_vec,]
setkey(hf_peaks, seqnames, start, end)

# load the normalized peak heights lists per CT
norm_summit_beds = Sys.glob(file.path(peaks_dir, "*", "*normalized_summit_heights.bed"))

for (myfile in norm_summit_beds){
  ct_summits = data.table(read.delim(myfile, sep="\t", header=FALSE))
  colnames(ct_summits) = c("chrom", "start", "end", "height")
  ct_name = unlist(strsplit(basename(myfile), "_", fixed=TRUE))[1]
  ct_summits$celltype = ct_name
  setkey(ct_summits, chrom, start, end)
  
  # reserve only those peaks associated with hf genes
  ct_hf_summits = merge(ct_summits, hf_peaks, by.x=c("chrom", "start", "end"), by.y=c("seqnames", "start", "end"))
  ct_hf_summits = ct_hf_summits[,.(chrom, start, end, height, celltype, width, annot, nearest.gene, nearest.gene.dist)]
  
  if (!exists("all_cthf_summits")){
    all_cthf_summits = ct_hf_summits
    all_ct_summits = ct_summits
  }
  else {
    all_cthf_summits = rbind(all_cthf_summits,ct_hf_summits)
    all_ct_summits = rbind(all_ct_summits, ct_summits)
  }
}

setkey(all_ct_summits, chrom, start, end)

setwd("/Users/zamparol/projects/SeqDemote/results/diagnostic_plots/ATAC/")

### For the peaks associated with high flyer genes, do we see multi-modality in their max peak heights?

# Plot the distribution of library scaled peak heights for each gene as a ridge plot
hf_lib_scaled_peak_heights <- ggplot(all_cthf_summits, aes(x = height, y = nearest.gene)) +
  geom_density_ridges(stat = "binline",bins=200) + 
  xlab("Library size normalized peak height") + ylab("Gene") + 
  theme_ridges(grid = FALSE) + 
  ggtitle("Normalized peak heights for peaks associated with high-flyer genes")

# Same as above, but this time correct for peak height by dividing by peak width
all_cthf_summits[, length_corrected_height := height / width]
hf_width_corrected_lib_scaled_peak_heights <- ggplot(all_cthf_summits, aes(x = length_corrected_height, y = nearest.gene)) +
  geom_density_ridges(stat = "binline",bins=300) + 
  xlab("Width adjusted library size normalized peak height") + ylab("Gene") + 
  theme_ridges(grid = FALSE) + 
  ggtitle("Width-adjusted normalized peak heights for peaks associated with high-flyer genes")

#### Not really.  What about for peaks in general across cell types?
all_ct_summits[, width := end - start]
all_ct_summits[, length_corrected_height := height / width]

all_lib_scaled_peak_heights <- ggplot(all_ct_summits, aes(x = height, y = celltype)) +
  geom_density_ridges(stat = "binline",bins=200) + 
  xlab("Library size normalized peak height") + ylab("Gene") + 
  theme_ridges(grid = FALSE) + 
  xlim(0,10) + 
  ggtitle("Normalized peak heights for all peaks")

# Same as above, but this time correct for peak height by dividing by peak width
all_width_corrected_lib_scaled_peak_heights <- ggplot(all_ct_summits, aes(x = length_corrected_height, y = celltype)) +
  geom_density_ridges(stat = "binline",bins=300) + 
  xlab("Width adjusted library size normalized peak height") + ylab("Gene") + 
  theme_ridges(grid = FALSE) + 
  xlim(0,0.025) + 
  ggtitle("Width-adjusted normalized peak heights for all peaks")


pdf(file = "peak_height_diagnostic_plots.pdf", width = 15, height = 13)

# compile plots into a list
pltList <- list()
pltList[[1]] <- hf_lib_scaled_peak_heights
pltList[[2]] <- hf_width_corrected_lib_scaled_peak_heights
pltList[[3]] <- all_lib_scaled_peak_heights
pltList[[4]] <- all_width_corrected_lib_scaled_peak_heights

# display the plots in a grid
grid.arrange(grobs=pltList, ncol=2)
dev.off()

