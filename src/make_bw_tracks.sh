#! /bin/bash

### try to catch a tonne of errors (h/t Michael Hoffman)
set -o nounset -o pipefail -o errexit

basedir=$1
bamroot="$basedir/bam"


# combine bams from all reps into one CT file
# for ct in $(find $bamroot -mindepth 1 -maxdepth 1 -type d)
# do
#   cd $ct
#   ctname=$(basename $ct)
#   bams=$(ls *sorted.bam)
#   suff="_all_merged.bam"
#   merged_name=$(echo $ctname$suff)
#   samtools merge -@ 8 $merged_name $bams
#   samtools index $merged_name
# done

cd $bamroot
outdir="$basedir/tracks"

# make a bw track for each file
# just the ones we don't have yet
declare -a mytypes=("./CD8Tcell" "./Bcell" "./CD4Tcell" "./CD34_Bone_Marrow" "./Ery")
#for ct in $(find $bamroot -mindepth 1 -maxdepth 1 -type d)
for ct in "${mytypes[@]}"
do
	cd $ct
	myct=$(basename $ct)
	outfile=$(echo $myct"_RPM_normalized.bw")
	bam=$(echo $myct"_all_merged.bam")
	echo "turning $bam into $outdir/$outfile..."
	#bamCoverage -b $bam --normalizeUsingRPKM -p 8 -o $outdir/$outfile
done
echo "ready to be loaded into IGV"

cd $bamroot
# remove CT combined bam files
for mb in $(find . -mindepth 1 -maxdepth 2 -name *all_merged.bam)
do
    echo "stub to do rm $mb"
    mybai=$(echo $mb".bai")
    echo "stub to do rm $mybai"
done