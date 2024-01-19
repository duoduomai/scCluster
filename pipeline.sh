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

##########5.call_variants##########
#gvcf
ref=../sc/ref/genome/hg38/hg38.fa
data=./study1/bam
sub_bam=./study1/sub_bam
gvcf=./study1/gvcf
mkdir gvcf
cat SRR.txt | while read i
do
	 time gatk HaplotypeCaller \
   -R ${ref} \
   -I ${sub_bam}/${i}.markdup.BQSR.split.sub_bam.bam \
   -ERC GVCF \
   -O ${gvcf}/${i}.erc.g.vcf
done

#joint
ref=../sc/ref/genome/hg38/hg38.fa
output_vcf=./study1/output_vcf
for chr in chr{1..22} chrX chrY
do
  gatk GenomicsDBImport \
	-R ${ref} \
	-V input.list \    #input.list=*.g.vcf list
	--genomicsdb-workspace-path ${output_vcf}/${chr}.db \
	-L ${chr}
gatk GenotypeGVCFs \
	-R ${ref} \
	-V gendb://${output_vcf}/${chr}.db \
	-O ./study1/genotype_gvcfs/gvcfs_${chr}.vcf
done

##########6.vqsr##########
#snp
known=../sc/GATK/gatkdoc
ref=../sc/ref/genome/hg38/hg38.fa
time gatk VariantRecalibrator \
  -R ${ref} \
  -V ./study1/genotype_gvcfs/merge.vcf \
  -resource:hapmap,known=false,training=true,truth=true,prior=15.0 /${known}/hapmap_3.3.hg38.vcf \
  -resource:omini,known=false,training=true,truth=false,prior=12.0 /${known}/1000G_omni2.5.hg38.vcf \
  -resource:1000G,known=false,training=true,truth=false,prior=10.0 /${known}/1000G_phase1.snps.high_confidence.hg38.vcf \
  -resource:dbsnp,known=true,training=false,truth=false,prior=2.0 /${known}/Homo_sapiens_assembly38.dbsnp138.vcf \
  -an DP -an QD -an FS -an SOR -an ReadPosRankSum -an MQRankSum \
  -mode SNP \
  -tranche 100.0 -tranche 99.9 -tranche 99.0 -tranche 95.0 -tranche 90.0 \
  -O snp.recal \
  --tranches-file snp.tranches \
  --rscript-file snp.plots.R  &&\
time gatk ApplyVQSR \
  -R ${ref} \
  --variant ./study1/genotype_gvcfs/merge.vcf \
  --ts-filter-level 99.0 \
  --tranches-file snp.tranches \
  --recal-file snp.recal \
  --mode SNP 
  --output /snps.VQSR.vcf  && echo"********SNPs VQSR done*********"
  
#merge_vcf_1
#ref=../sc/ref/genome/hg38/hg38.fa
#output_vcf=./study1/output_vcf
#for chr in chr{1..22} chrX chrY
#do
#  bgzip ./study1/genotype_gvcfs/gvcfs_${chr}.vcf
#  tabix -p vcf  ./study1/genotype_gvcfs/gvcfs_${chr}.vcf.gz
#done
#merge_vcf_2
gatk MergeVcfs \
   $(for i in `cat ./study1/chr.list | cut -f 1 `; do echo "-I ${i}.VQSR.vcf.gz " ;done) \
   -O merge.vcf

#vcf-filter
vcftools --vcf merge.vcf --max-missing 0.9 --maf 0.05 --recode --recode-INFO-all --out merge09maf05
#012
vcftools --vcf merge09maf05.recode.vcf --012 --out snp_matrix





