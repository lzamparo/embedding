#! /bin/bash

$basedir=$1
$bamroot="$basedir/bam"

# combine bams from all reps into one CT file
for ct in $(ls -1 $bamroot)
do
  cd $bamroot/$ct
  bams=$(find . -name *.bam)
  suff="_all_merged.bam"
  merged_name=$(echo $ct$suff)
  echo "samtools merge -@ 16 $merged_name $bams"
  echo "samtools index $merged_name"
done

cd $bamroot
outdir="$basedir/tracks"

# make a bw track for each file
for bam in $(find . -name *all_merged.bam)
do
	ct=$(basename $bam | cut -d'_' -f 1)
	outfile=$(echo $ct"_RPM_normalized.bw")
	echo "turning $bam into $outdir/$outfile..."
	#bamCoverage -b $bam --normalizeUsingRPKM -p 8 -o $outdir/$outfile
done
echo "ready to be loaded into IGV"

cd $bamroot
# remove CT combined bam files
for mb in $(find . -name *all_merged.bam)
do
    echo "stub to do rm $mb"
done