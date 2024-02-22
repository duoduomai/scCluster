#!/bin/bash

##########1.download##########
#paired-fastq
ftp='era-fasp@fasp.sra.ebi.ac.uk:vol1/fastq'
cat SRR.txt | while read i
do
	SRR=$(echo ${i:0:6})
	pre="00"
        late=$(echo ${i:9:10})
        index=${pre}${late}
	while [ ! -f "raw_data/"${i}"_1.fastq.gz" ];do
		echo "************Begin Download************"
		echo ""
		echo "Tring download"${i}"_1.fastq.gz"
		echo ""
		ascp -QT -l 300m -P 33001 -i /path/to/.conda/envs/gatk/etc/asperaweb_id_dsa.openssh ${ftp}/fastq/${SRR}/${index}/${i}/${i}_1.fastq.gz ./raw_data
		echo "************Finished Download************"
		echo ""
	done
	while [ ! -f "raw_data/"${i}"_2.fastq.gz" ];do
		echo "************Begin Download************"
		echo ""
		echo "Tring download"${i}"_2.fastq.gz"
		echo ""
		ascp -QT -l 300m -P 33001 -i /path/to/.conda/envs/gatk/etc/asperaweb_id_dsa.openssh ${ftp}/${SRR}/${index}/${i}/${i}_2.fastq.gz ./raw_data
		echo "************Finished Download************"
		echo ""
	done
done
#single-fastq
ftp='era-fasp@fasp.sra.ebi.ac.uk:vol1/fastq'
cat SRR.txt | while read i
do
	SRR=$(echo ${i:0:6})
	pre="00"
        late=$(echo ${i:9:10})
        index=${pre}${late}
	while [ ! -f "raw_data/"${i}".fastq.gz" ];do
		echo "************Begin Download************"
		echo ""
		echo "Tring download"${i}".fastq.gz"
		echo ""
		ascp -QT -l 300m -P 33001 -i /path/to/.conda/envs/gatk/etc/asperaweb_id_dsa.openssh ${ftp}/${SRR}/${index}/${i}/${i}.fastq.gz ./raw_data
		echo "************Finished Download************"
		echo ""
	done
done
#download-test
cat SRR.txt | while read i
do
	if [ -f "./study1/clean_data/${i}_paired_clean_1.fastq.gz" ];then
		echo "success download"
	else
		echo "false download"
	fi
done
#md5-test
find /raw_data -type f -print0 | xargs -0 md5sum >> md5.txt

##########2.QC##########
#trimmomatic&&fastqc
data=./study1/raw_data
clean_data=./study1/clean_data_2
fastqc_data=./study1/fastqc_data_2
clip=../sc/software/Trimmomatic-0.39/adapters
export path=SRR.txt
for i in $(cat $path)
do
  echo "************Begin Filtering************"
  echo '***********'${i}'*************'
  java -jar ../sc/software/Trimmomatic-0.39/trimmomatic-0.39.jar PE -threads 20 -phred33 \
  ${data}/${i}_1.fastq.gz ${data}/${i}_2.fastq.gz \
  ${clean_data}/${i}_paired_clean_1.fastq.gz \
  ${clean_data}/${i}_unpair_clean_1.fastq.gz \
  ${clean_data}/${i}_paired_clean_2.fastq.gz \
  ${clean_data}/${i}_unpair_clean_2.fastq.gz \
  ILLUMINACLIP:${clip}/TruSeq3-PE.fa:2:30:10:8:true \
  LEADING:3 \
  TRAILING:3 \
  MAXINFO:40:0.6 \
  SLIDINGWINDOW:4:15 MINLEN:50
  fastqc ${clean_data}/${i}_paired_clean_1.fastq.gz -o ${fastqc_data}/
  fastqc ${clean_data}/${i}_paired_clean_2.fastq.gz -o ${fastqc_data}/
done

##########3.alignement##########
#STAR
export path=./study1/SRR.txt
for i in $(cat $path)
do
	mkdir align_${i}_out
	STAR \
		--runThreadN 12 \
		--genomeDir ../sc/ref/star/hg38/ \
		--readFilesIn ./study1/clean_data/${i}_paired_clean_1.fastq.gz ./study1/clean_data/${i}_paired_clean_2.fastq.gz \
		--outFileNamePrefix ./study1/align_data/align_${i}_out/${i}_ \
		--readFilesCommand gunzip -c \
		--outSAMtype BAM SortedByCoordinate \
		--quantMode GeneCounts 
done

##########4.preprocess##########
#preprocess-AddOrReplaceReadGroups
mkdir ./study1/raw_bam
mkdir ./study1/bam
mkdir ./study1/sub_bam

export path=./study1/SRR.txt
for i in $(cat $path)
do
	mv ./study1/align_data/align_${i}_out/${i}_Aligned.sortedByCoord.out.bam ${i}.raw.bam
	samtools index ${i}.raw.bam
  picard AddOrReplaceReadGroups \
		I=${i}.raw.bam \
		O=${i}.bam \
		RGID=${i} \
		RGLB=lib${i} \
		RGPL=illumina \
		RGPU=unit \
		RGSM=${i} 
	samtools index ${i}.bam
	samtools view -h -b ${i}.bam chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX chrY > ${i}.sub_bam.bam
	samtools index ${i}.sub_bam.bam
	mv ${i}.raw.bam ./study1/raw_bam
	mv ${i}.bam ./study1/bam
	mv ${i}.sub_bam.bam ./study1/sub_bam
done

#sort && MarkDuplicates && index
bam=./study1/bam
cat SRR.txt|while read i
do
	time samtools sort -@ 18 -o ${bam}/${i}.bam ${bam}/${i}.bam
	time picard MarkDuplicates -Xmx32g I=${bam}/${i}.bam O=${bam}/${i}.markdup.bam M=${bam}/${i}.markdup.txt REMOVE_DUPLICATES=true
	time picard BuildBamIndex -Xmx32g I=${bam}/${i}.markdup.bam
	time samtools flagstat -@ 30 ${bam}/${i}.markdup.bam >${bam}/${i}.markdup.stat
	cat ${bam}/${i}.markdup.stat
done

#BQSR
ref=../sc/ref/genome/hg38/hg38.fa
known=../sc/GATK/gatkdoc
data=./study1/bam
output_bam=./study1/sub_bam
cat SRR.txt | while read i
do
time gatk BaseRecalibrator \
    -R ${ref} \
    -I ${data}/${i}.markdup.bam \
    --known-sites ${known}/Homo_sapiens_assembly38.dbsnp138.vcf \
    --known-sites ${known}/Mills_and_1000G_gold_standard.indels.hg38.vcf \
    --known-sites ${known}/1000G_phase1.snps.high_confidence.hg38.vcf \
    -O ${data}/recal_data_${i}.table
time gatk ApplyBQSR --bqsr-recal-file ${data}/recal_data_${i}.table \
    -R ${ref} \
    -I ${data}/${i}.markdup.bam \
    -O ${data}/${i}.markdup.BQSR.bam
done
time gatk SplitNCigarReads \
   -R ${ref} \
   -I ${data}/${i}.markdup.BQSR.bam \
   -O ${data}/${i}.markdup.BQSR.split.bam
   samtools view -h -b ${data}/${i}.markdup.BQSR.split.bam chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX chrY > ${output_bam}/${i}.markdup.BQSR.split.sub_bam.bam
   samtools index ${output_bam}/${i}.markdup.BQSR.split.sub_bam.bam

##########5.featureCounts##########

DATA_DIR="./study1/sub_bam"
GTF_FILE="annotation.gtf"
OUTPUT_MATRIX="expression_matrix.csv"

# featureCounts 
featureCounts -a "$GTF_FILE" -o counts_all_samples.txt $(find "$DATA_DIR" -name '*.markdup.BQSR.split.sub_bam.bam')
# expression_matrix
cut -f 1 counts_all_samples.txt > gene_names.txt
cut -f 7- counts_all_samples.txt | tail -n +3 | awk '{print $0"\t"}' | paste -s -d "\t" - > sample_counts.tmp
paste gene_names.txt sample_counts.tmp > "$OUTPUT_MATRIX"








