#! /bin/bash

# calculate the median library size corrected peak height for all celltypes in the Corces data set
set -o nounset -o pipefail -o errexit

# submit for all CTs
basedir=/data/leslie/zamparol/heme_ATAC/data
peaks_root="peaks"

for ct in $(find $basedir/$peaks_root -mindepth 1 -maxdepth 1 -type d)
do
	celltype=$(basename $ct)
	bsub -env "all, ct=$celltype" < submit_scripts/run_peak_height.lsf
done
