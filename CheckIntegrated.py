#!/usr/bin/env python3

import os
import argparse
import logging
import multiprocessing as mp
from pathlib import Path
from itertools import islice
from Bio import SeqIO
from Bio.Blast.Applications import NcbimakeblastdbCommandline, NcbiblastnCommandline
import pyhmmer
from pyhmmer import easel, plan7
import pyrodigal_gv
import collections

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Check if a set of query viral sequences are integreated within a set of subject genome sequences.")
    parser.add_argument("--db_dir", type=str, required=True, help="Path to CheckAMG database files (Required).")
    parser.add_argument("--output", type=str, required=True, help="Output directory for all generated files and folders (Required).")
    parser.add_argument("--wdir", type=str, required=True, help="Working directory for intermediate files (Required).")
    parser.add_argument("--genomes", type=str, required=True, help="Input viral genome(s) in nucleotide fasta format (.fna or .fasta, expects one contig per genome).")
    parser.add_argument("--host_genomes_dir", type=str, required=True, help="Directory containing host genome files (.fa, .fna, or .fasta).")
    parser.add_argument("--min_id", type=float, default=1.0, help="Minimum percent identity of a BLASTn alignment required to consider a query (virus) sequence detected in a subject (host) sequence (0.0-1.0) (default: 1.0).")
    parser.add_argument("--min_cov", type=float, default=1.0, help="Minimum fraction of the query (virus) sequence aligned by BLASTn required to consider the query (virus) sequence detected in a subject (host) sequence (0.0-1.0) (default: 1.0).")
    parser.add_argument("--min_annot", type=float, default=0.20, help="Minimum percentage (0.0-1.0) of genes with a valid V-score/VL-score asisgnment in a region required to have that region checked (default: 0.20).")
    parser.add_argument("--max_flank", type=int, default=25000, help="Maximum length (in base pairs) to the left/right of an aligned subject sequence to check if it is viral/non-viral (default: 25000).")
    parser.add_argument("--max_kegg_lscore", type=float, default=3.0, help="Maximum average KEGG VL-score of a sequence region to be considered non-viral (default: 3) (Floating Point 0.0-10.0).")
    parser.add_argument("--max_pfam_lscore", type=float, default=3.0, help="Maximum average Pfam VL-score of a sequence region to be considered non-viral (default: 3) (Floating Point 0.0-10.0).")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads to use (default: 1).")
    return parser.parse_args()
    
def prepare_blast_db(host_genomes_dir, output):
    combined_fasta = os.path.join(output, "combined_host_genomes.fasta")
    
    file_patterns = ["*.fasta", "*.fa", "*.fna"]
    
    with open(combined_fasta, "w") as outfile:
        for pattern in file_patterns:
            for filepath in Path(host_genomes_dir).rglob(pattern):
                basename = os.path.basename(filepath).rsplit('.', 1)[0]
                with open(filepath, "r") as infile:
                    for record in SeqIO.parse(infile, "fasta"):
                        record.id = f"{basename}__{record.id}"
                        record.description = ""
                        SeqIO.write(record, outfile, "fasta")
    
    blast_db = os.path.join(output, "combined_host_genomes")
    makeblastdb_cline = NcbimakeblastdbCommandline(input_file=combined_fasta, dbtype="nucl", out=blast_db)
    stdout, stderr = makeblastdb_cline()
    logging.info(f"BLAST database created at {blast_db}")
    return blast_db

def run_blast(query, db, num_threads, outfmt="6 qseqid sseqid pident length evalue bitscore qstart qend qlen sstart send slen"):
    output_file = "blast_output.tsv"
    blastn_cline = NcbiblastnCommandline(query=query, db=db, outfmt=outfmt, out=output_file, num_threads=num_threads)
    stdout, stderr = blastn_cline()
    return output_file

def process_blast_output(blast_output, query_lengths, min_cov, min_id):
    blast_df = pl.read_csv(blast_output, separator="\t", has_header=False, infer_schema_length=100000, new_columns=[
        "qseqid", "sseqid", "pident", "length", "evalue", "bitscore", "qstart", "qend", "qlen", "sstart", "send", "slen"
    ])
    blast_df = blast_df.with_columns([
        (pl.col("length").cast(pl.Float64) / pl.col("qseqid").map_elements(lambda x: query_lengths.get(x, float('nan')), return_dtype=pl.Float64)).alias("alignment_coverage"),
        pl.col("qlen").cast(pl.Int32),
        pl.col("slen").cast(pl.Int32)
    ])
    best_hits = (
        blast_df
        .filter((pl.col("alignment_coverage") >= min_cov) & (pl.col("alignment_coverage") <= 1.0) & (pl.col("pident") >= min_id * 100))
        .sort(by=["qseqid", "sseqid", "bitscore", "length"], descending=[False, False, True, True])
        .group_by(["qseqid", "sseqid"])
        .agg(pl.col("*").first())
    )
    best_hits = best_hits.with_columns([
        pl.col("qseqid").map_elements(lambda x: query_lengths.get(x, float('nan')), return_dtype=pl.Float64).alias("qlen")
    ])
    return best_hits

def extract_host_regions(best_hits, max_flank, host_sequences, outfasta):
    with open(outfasta, "w") as outfile:
        for hit in best_hits.iter_rows(named=True):
            start = max(0, hit["sstart"] - max_flank)
            end = hit["send"] + max_flank
            sequence = str(host_sequences[hit["sseqid"]].seq[start:end])
            # Write to output FASTA file
            outfile.write(f">{hit['qseqid']}_{hit['sseqid']}__{start}_{end}\n{sequence}\n")
            
def predict_genes(seq_tuple):
    seq_id, seq_seq, is_vmag, vmag_name = seq_tuple  # Unpack the tuple
    orf_finder = pyrodigal_gv.ViralGeneFinder(meta=True, mask=True)
    return (seq_id, orf_finder.find_genes(str(seq_seq)))

def chunk_sequences(sequences, chunk_size):
    it = iter(sequences)
    for first in it:
        yield [first] + list(islice(it, chunk_size - 1))

def process_chunk(chunk):
    chunk_results = []
    for seq in chunk:
        chunk_results.append(predict_genes(seq))
    return chunk_results

def run_parallel_prodigal(sequences_fasta, threads, chunk_size=100):    
    input_sequences = [(seq.id, seq.seq, False, None) for seq in SeqIO.parse(sequences_fasta, 'fasta')]
            
    chunks = list(chunk_sequences(input_sequences, chunk_size))
    with mp.Pool(threads) as pool:
        results = pool.map(process_chunk, chunks)
        pool.close()
        pool.join()
    
    all_results = [item for sublist in results for item in sublist]
    return all_results

def parse_faa_file(file_path):
    sequence_data = ''
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('>'):
                if sequence_data:  # Yield previous entry
                    entry['sequence'] = sequence_data.replace('\n', '')
                    yield entry
                    sequence_data = ''

                parts = line.strip().split(' # ')
                contig_and_gene = parts[0].split()[0].lstrip('>')  # Remove '>'
                
                contig = "_".join(contig_and_gene.rsplit('_', 1)[:-1])
                gene_number = contig_and_gene.rsplit('_', 1)[-1]
                genome = contig
                contig_pos_start, contig_pos_end, frame = parts[1:4]
                attributes = dict(item.split('=') for item in parts[4].split(';') if item)

                entry = {
                    'member': contig_and_gene,
                    'contig': contig,
                    'gene_number': gene_number,
                    'genome': genome,
                    'contig_pos_start': contig_pos_start,
                    'contig_pos_end': contig_pos_end,
                    'frame': frame,
                }
                entry.update(attributes)
            else:
                sequence_data += line.strip()
    
    if sequence_data:  # Don't forget the last entry
        entry['sequence'] = sequence_data.replace('\n', '')
        yield entry

def write_results(all_results, output_dir):
    output_faa_path = os.path.join(output_dir, "host_prophage_regions_proteins.faa")
    gene_to_genome_output_path = os.path.join(output_dir, "gene_to_genome.tsv")

    with open(output_faa_path, 'w') as non_vmag_output_faa, \
        open(gene_to_genome_output_path, 'w') as gene_to_genome_output:

        gene_to_genome_output.write("gene\tcontig\tgenome\tgene_number\n")  # Updated header

        for seq_id, predicted_genes in all_results:
            for gene_i, gene in enumerate(predicted_genes, 1):
                gene_name = f"{seq_id}_{gene_i}"
                header = (
                    f">{gene_name} # {gene.begin} # {gene.end} # "
                    f"{gene.strand} # ID={seq_id}_{gene_i};"
                    f"partial={int(gene.partial_begin)}{int(gene.partial_end)};"
                    f"start_type={gene.start_type};rbs_motif={gene.rbs_motif};"
                    f"rbs_spacer={gene.rbs_spacer};"
                    f"genetic_code={gene.translation_table};"
                    f"gc_cont={gene.gc_cont:.3f}"
                )
                prot = f"{header}\n{gene.translate(include_stop=False)}\n"
                
                non_vmag_output_faa.write(prot)  # Write to non-vMAG file
                genome_name = seq_id

                gene_to_genome_output.write(f"{gene_name}\t{seq_id}\t{genome_name}\t{gene_i}\n")

def process_data(single_contig_prots):
    # Accumulate all faa data in chunks to avoid large memory usage
    faa_data = []

    data = parse_faa_file(single_contig_prots)
    faa_data.extend(data)

    # Convert the list of dictionaries (faa_data) to a Polars DataFrame
    faa_dataframe = pl.DataFrame(faa_data)

    return faa_dataframe
    
def load_hmms(hmmdbs):
    hmms = {}
    for db_path in hmmdbs:
        with plan7.HMMFile(db_path) as hmm_file:
            hmms[db_path] = list(hmm_file)
    return hmms

def filter_results(results):
    best_results = {}
    for result in results:
        key = (result.sequence, result.db_path)
        if key not in best_results or result.score > best_results[key].score:
            best_results[key] = result
    return list(best_results.values())

def run_hmmsearch_on_sequences(sequence_files, hmms, e_value_threshold=1e-5, num_cpus=1):
    """Perform an HMM search against a list of sequences using HMM profiles from each database."""
    Result = collections.namedtuple("Result", ["hmm_id", "sequence", "score", "db_path"])

    all_results = []
    for sequence_file in sequence_files:
        # Validate the sequence file format using biopython
        try:
            with open(sequence_file, 'r') as f:
                sequences = list(SeqIO.parse(f, "fasta"))
                if not sequences:
                    raise ValueError(f"No sequences found in file {sequence_file}.")
        except Exception as e:
            raise ValueError(f"Error reading sequence file {sequence_file}: {e}")
        
        # Create the amino acid alphabet
        aa_alphabet = easel.Alphabet.amino()
        
        # Read the sequences using pyhmmer.easel after validation
        with easel.SequenceFile(sequence_file, format="fasta", digital=True, alphabet=aa_alphabet) as seqs_file:
            proteins = list(seqs_file)

        for db_path, hmm_list in hmms.items():
            results = []
            for hits in pyhmmer.hmmsearch(queries=hmm_list, sequences=proteins, cpus=num_cpus, E=e_value_threshold):
                for hit in hits:
                    if "Pfam" in db_path:
                        results.append(Result(hits.query_accession.decode(), hit.name.decode(), hit.score, db_path))
                    elif "eggNOG" in db_path:
                        results.append(Result(hits.query_name.decode().split(".")[0], hit.name.decode(), hit.score, db_path))
                    else:
                        results.append(Result(hits.query_name.decode(), hit.name.decode(), hit.score, db_path))
            # Filter results for this database and add to all results
            all_results.extend(filter_results(results))
    
    return all_results

def assign_db(db_path):
    if "KEGG" in db_path or "kegg" in db_path:
        return "KEGG"
    elif "Pfam" in db_path or "pfam" in db_path:
        return "Pfam"
    else:
        return None 

def prefix_columns(dataframe, prefix):
    # Select 'sequence' only once, and prefix other columns as needed
    cols_to_select = [pl.col("sequence")]
    cols_to_select.extend(
        pl.col(col).alias(f"{prefix}_{col}") for col in dataframe.columns if col not in ["sequence", "db"]
    )
    return dataframe.select(cols_to_select)

def widen_hmm_results(hmm_results_df):
    # Create a new column for each combination of `db` and the associated scores/ids/V-scores
    hmm_results_df = hmm_results_df.with_columns([
        pl.when(pl.col("db") == db).then(pl.col(col)).otherwise(None).alias(f"{db}_{col}")
        for db in hmm_results_df.select("db").unique().to_series().to_list()
        for col in ["score", "hmm_id"]
    ])

    # Remove the original score, hmm_id, V-score, and db columns as they are now redundant
    hmm_results_df = hmm_results_df.drop(["score", "hmm_id", "db"])

    # Pivot the DataFrame to widen it, aggregating by sequence and filling missing values as needed
    hmm_results_df_wide = hmm_results_df.group_by("sequence").agg([
        pl.max(col).alias(col) for col in hmm_results_df.columns if col != "sequence"
    ])

    return hmm_results_df_wide

def add_vscores(processed_data, hmm_df_wide, vscores_df):
    merged_df = processed_data.join(hmm_df_wide, left_on='member', right_on='sequence', how='left')
    cols_to_remove = ['KEGG_score', 'Pfam_score']
    cols_to_remove += [col for col in merged_df.columns if col.endswith('_right')]
    merged_df = merged_df.drop(cols_to_remove)

    # Split the DataFrame by 'db' value
    df_pfam = vscores_df.filter(pl.col("db") == "Pfam")
    df_kegg = vscores_df.filter(pl.col("db") == "KEGG")

    # Add prefixes
    pfam_prefixed = prefix_columns(df_pfam, "Pfam")
    kegg_prefixed = prefix_columns(df_kegg, "KEGG")

    # Join on 'sequence'
    wide_df = pfam_prefixed.join(kegg_prefixed, on="sequence", how="outer")
    wide_df = wide_df.drop(["sequence_right", "Pfam_hmm_id", "Pfam_db_right", "KEGG_score", "KEGG_db_right"])

    # Merge V-scores and L-scores with input TSV, and save results
    merged_df = merged_df.join(wide_df, left_on='member', right_on='sequence', how='left')
    
    return merged_df
    
def calculate_gene_lengths(data):
    data = data.with_columns([
        (pl.col('contig_pos_end').cast(pl.Int64) - pl.col('contig_pos_start').cast(pl.Int64) + 1).alias('gene_length_bases'),
        ((pl.col('contig_pos_end').cast(pl.Int64) - pl.col('contig_pos_start').cast(pl.Int64) + 1) / 3).cast(pl.Int32).alias('prot_length_AAs')
    ])
    return data

def summarize_gene_group(region):
    num_genes = len(region)
    if num_genes == 0:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

    annotated_genes = region.filter(pl.col("KEGG_hmm_id").is_not_null() | pl.col("Pfam_hmm_id").is_not_null()).shape[0]
    percent_annotated = annotated_genes / num_genes if num_genes > 0 else float('nan')

    avg_kegg_vscore = region.select(pl.col("KEGG_V-score").filter(pl.col("KEGG_V-score").is_not_null())).mean().item()
    avg_pfam_vscore = region.select(pl.col("Pfam_V-score").filter(pl.col("Pfam_V-score").is_not_null())).mean().item()
    avg_kegg_lscore = region.select(pl.col("KEGG_L-score").filter(pl.col("KEGG_L-score").is_not_null())).mean().item()
    avg_pfam_lscore = region.select(pl.col("Pfam_L-score").filter(pl.col("Pfam_L-score").is_not_null())).mean().item()

    return percent_annotated, avg_kegg_vscore, avg_pfam_vscore, avg_kegg_lscore, avg_pfam_lscore

def process_annotated_gene_index(gene_index_annotated, best_hits):
    summary_data = []

    for hit in best_hits.iter_rows(named=True):
        genome = hit["qseqid"]
        host_contig = hit["sseqid"]
        start = hit["sstart"]
        end = hit["send"]
        slen = int(hit["slen"])
        qlen = int(hit["qlen"])

        # Ensure proper filtering by extracting the relevant part of the contig
        contig_name = f"{genome}_{host_contig}__{start}_{end}".split('__')[0]
        contig_data = gene_index_annotated.filter(pl.col("contig").str.contains(contig_name))

        # Convert contig_pos_start and contig_pos_end to integers
        contig_data = contig_data.with_columns([
            pl.col("contig_pos_start").cast(pl.Int32),
            pl.col("contig_pos_end").cast(pl.Int32)
        ])

        # Prophage genes
        prophage_genes = contig_data.filter((pl.col("contig_pos_start") >= start) & (pl.col("contig_pos_end") <= end))
        percent_annotated_prophage, avg_kegg_vscore_prophage, avg_pfam_vscore_prophage, avg_kegg_lscore_prophage, avg_pfam_lscore_prophage = summarize_gene_group(prophage_genes)

        # Left genes
        left_genes = contig_data.filter(pl.col("contig_pos_end") < start)
        percent_annotated_left, avg_kegg_vscore_left, avg_pfam_vscore_left, avg_kegg_lscore_left, avg_pfam_lscore_left = summarize_gene_group(left_genes)

        # Right genes
        right_genes = contig_data.filter(pl.col("contig_pos_start") > end)
        percent_annotated_right, avg_kegg_vscore_right, avg_pfam_vscore_right, avg_kegg_lscore_right, avg_pfam_lscore_right = summarize_gene_group(right_genes)

        summary_data.append({
            "query": genome,
            "query_length": qlen,
            "subject": host_contig,
            "subject_genome": host_contig.split('__')[0],
            "subject_contig": host_contig.split('__')[1],
            "subject_start": start,
            "subject_end": end,
            "subject_length": slen,
            "percent_annotated_left": percent_annotated_left,
            "avg_kegg_vscore_left": avg_kegg_vscore_left,
            "avg_pfam_vscore_left": avg_pfam_vscore_left,
            "avg_kegg_lscore_left": avg_kegg_lscore_left,
            "avg_pfam_lscore_left": avg_pfam_lscore_left,
            "percent_annotated_center": percent_annotated_prophage,
            "avg_kegg_vscore_center": avg_kegg_vscore_prophage,
            "avg_pfam_vscore_center": avg_pfam_vscore_prophage,
            "avg_kegg_lscore_center": avg_kegg_lscore_prophage,
            "avg_pfam_lscore_center": avg_pfam_lscore_prophage,
            "percent_annotated_right": percent_annotated_right,
            "avg_kegg_vscore_right": avg_kegg_vscore_right,
            "avg_pfam_vscore_right": avg_pfam_vscore_right,
            "avg_kegg_lscore_right": avg_kegg_lscore_right,
            "avg_pfam_lscore_right": avg_pfam_lscore_right
        })

    summary_df = pl.DataFrame(summary_data)
    return summary_df

def summarize_regions(summary_df, max_kegg_lscore, max_pfam_lscore, min_annot):
    def check_region(percent_annotated, avg_kegg_lscore, avg_pfam_lscore):
        if percent_annotated < min_annot:
            return float('nan')
        
        if avg_kegg_lscore is None and avg_pfam_lscore is None:
            return float('nan')
        elif avg_kegg_lscore is None and avg_pfam_lscore is not None:
            return 1.0 if avg_pfam_lscore <= max_pfam_lscore else 0.0
        elif avg_pfam_lscore is None and avg_kegg_lscore is not None:
            return 1.0 if avg_kegg_lscore <= max_kegg_lscore else 0.0
        else:
            return 1.0 if (avg_kegg_lscore <= max_kegg_lscore and avg_pfam_lscore <= max_pfam_lscore) else 0.0

    flank_data = []
    for row in summary_df.iter_rows(named=True):
        likely_viral = check_region(row["percent_annotated_center"], row["avg_kegg_lscore_center"], row["avg_pfam_lscore_center"])
        left_flank_host = check_region(row["percent_annotated_left"], row["avg_kegg_lscore_left"], row["avg_pfam_lscore_left"])
        right_flank_host = check_region(row["percent_annotated_right"], row["avg_kegg_lscore_right"], row["avg_pfam_lscore_right"])

        flank_data.append({
            "query": row["query"],
            "likely_viral": (not likely_viral) if (likely_viral == likely_viral) else float('nan'),
            "left_flank_host": left_flank_host if left_flank_host == left_flank_host else float('nan'),
            "right_flank_host": right_flank_host if right_flank_host == right_flank_host else float('nan')
        })

    flank_df = pl.DataFrame(flank_data)
    return flank_df

def main(args):
    # Set up the output directory
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    
    # Prepare BLAST database from host genome files
    if os.path.exists(os.path.join(args.output, "combined_host_genomes.fasta")) and os.path.exists(os.path.join(args.output, "combined_host_genomes.ndb")):
        logging.info(f"BLAST database already exists at {os.path.join(args.output, 'combined_host_genomes')}.")
        blast_db = os.path.join(args.output, "combined_host_genomes")
    else:
        logging.info("Preparing BLAST database from host genome files.")
        blast_db = prepare_blast_db(args.host_genomes_dir, args.output)
    
    # Run BLASTn to align prophage sequences with host sequences
    if os.path.exists(os.path.join(args.output, "blast_output.tsv")) and os.path.exists(os.path.join(args.output, "blast_best_hits.tsv")):
        logging.info(f"BLAST hits already exist at {os.path.join(args.output, 'blast_output.tsv')} and {os.path.join(args.output, 'blast_best_hits.tsv')}.")
        blast_output = os.path.join(args.output, "blast_output.tsv")
        query_lengths = {record.id: len(record.seq) for record in SeqIO.parse(args.genomes, "fasta")}
        best_hits = pl.read_csv(os.path.join(args.output, "blast_best_hits.tsv"), separator="\t")
    else:
        logging.info("Running BLAST to align prophage sequences with host sequences.")
        blast_output = run_blast(args.genomes, blast_db, args.threads)
        query_lengths = {record.id: len(record.seq) for record in SeqIO.parse(args.genomes, "fasta")}
        best_hits = process_blast_output(blast_output, query_lengths, args.min_cov, args.min_id)
        best_hits.write_csv(os.path.join(args.output, "blast_best_hits.tsv"), separator="\t")

    # Extract host regions from BLAST hits (with extra flanking sequences)
    if os.path.exists(os.path.join(args.output, "host_prophage_regions.fna")):
        logging.info(f"Host regions already extracted at {os.path.join(args.output, 'host_prophage_regions.fna')}.")
    else:
        logging.info("Extracting host regions from BLAST hits.")
        host_sequences = SeqIO.to_dict(SeqIO.parse(os.path.join(args.output, "combined_host_genomes.fasta"), "fasta"))
        extract_host_regions(best_hits, args.max_flank, host_sequences, os.path.join(args.output, "host_prophage_regions.fna"))
    
    # Predict genes in the extracted host regions    
    if os.path.exists(os.path.join(args.output, "host_prophage_regions_proteins.faa")):
        logging.info(f"Predicted genes already exist at {os.path.join(args.output, 'host_prophage_regions_proteins.faa')}.")
    else:
        logging.info("Predicting genes in the extracted host regions.")
        all_results = run_parallel_prodigal(os.path.join(args.output, "host_prophage_regions.fna"), args.threads, 100)
        write_results(all_results, args.output)
    
    # Run HMM search to assign V-scores and L-scores and write results
    if os.path.exists(os.path.join(args.output, "hmm_results.tsv")) and os.path.exists(os.path.join(args.output, "vscores.tsv")):
        logging.info(f"HMM search results already exist at {os.path.join(args.output, 'hmm_results.tsv')} and {os.path.join(args.output, 'vscores.tsv')}.")
        vscores_df = pl.read_csv(os.path.join(args.output, "vscores.tsv"), separator="\t")
        hmm_df = pl.read_csv(os.path.join(args.output, "hmm_results.tsv"), separator="\t")
    else:
        prots = [os.path.join(args.output, "host_prophage_regions_proteins.faa")]
        hmms_loc = [os.path.join(args.db_dir, db) for db in os.listdir(args.db_dir) if ('KEGG' in db or 'kegg' in db or 'Pfam' in db or 'pfam' in db) and (db.endswith(".H3M") or db.endswith(".h3m"))]
        logging.info(f"Loading HMM profiles from {hmms_loc}.")
        hmms = load_hmms(hmms_loc)
        
        logging.info(f"Running HMM search to assign V-scores and L-scores as they appear in {os.path.join(args.wdir, 'vscores.csv')}.")
        all_results = run_hmmsearch_on_sequences(prots, hmms, 1E-5, args.threads)
        
        hmm_df = pl.DataFrame([{"hmm_id": r.hmm_id, "sequence": r.sequence, "score": r.score, "db_path": r.db_path} for r in all_results])
        hmm_df = hmm_df.rename({"hmm_id": "id"}).with_columns(pl.col("db_path").map_elements(assign_db, return_dtype=pl.Utf8).alias("db")).drop('db_path')
        hmm_df = hmm_df.rename({"id": "hmm_id"}).sort(['sequence', 'score', 'db', 'hmm_id'])
        hmm_df.write_csv(os.path.join(args.output, "hmm_results.tsv"), separator="\t")

        vscores = pl.read_csv(os.path.join(args.wdir, "files", "vscores.csv"), schema_overrides={"id": pl.Utf8, "V-score": pl.Float64, "L-score": pl.Float64, "db": pl.Categorical, "name": pl.Utf8})
        vscores_df = hmm_df.join(vscores, left_on='hmm_id', right_on='id', how='left').filter(pl.col("V-score").is_not_null())
        vscores_df.write_csv(os.path.join(args.output, "vscores.tsv"), separator="\t")

    # Process the data to generate a gene index with annotation and information
    if os.path.exists(os.path.join(args.output, "gene_index.tsv")):
        logging.info(f"Gene index already exists at {os.path.join(args.output, 'gene_index.tsv')}.")
        merged_df = pl.read_csv(os.path.join(args.output, "gene_index.tsv"), separator="\t")
    else:
        logging.info("Generating a gene index with annotation and information.")
        processed_data = process_data(os.path.join(args.output, "host_prophage_regions_proteins.faa"))
        hmm_df_wide = widen_hmm_results(hmm_df)
        merged_df = add_vscores(processed_data, hmm_df_wide, vscores_df)
        merged_df.write_csv(os.path.join(args.output, "gene_index.tsv"), separator="\t")
    
    # Process the annotated gene index to summarize the genome-level information
    if os.path.exists(os.path.join(args.output, "genome_summary.tsv")):
        logging.info(f"Genome summary already exists at {os.path.join(args.output, 'genome_summary.tsv')}.")
        summary_df = pl.read_csv(os.path.join(args.output, "genome_summary.tsv"), separator="\t")
    else:
        logging.info("Processing annotated gene index to summarize genome-level information.")
        summary_df = process_annotated_gene_index(merged_df, best_hits)
        summary_df.write_csv(os.path.join(args.output, "genome_summary.tsv"), separator="\t")

    # Write unaligned query prophage names to a file
    if os.path.exists(os.path.join(args.output, "unaligned_queries.txt")):
        logging.info(f"Unaligned queries already exist at {os.path.join(args.output, 'unaligned_queries.txt')}.")
    else:
        logging.info(f"Writing unaligned query prophage names to {os.path.join(args.output, 'unaligned_queries.txt')}.")
        aligned_queries = set(best_hits['qseqid'].unique())
        all_queries = set(query_lengths.keys())
        unaligned_queries = all_queries - aligned_queries
        with open(os.path.join(args.output, "unaligned_queries.txt"), "w") as f:
            for query in unaligned_queries:
                f.write(f"{query}\n")
        # if unaligned_queries:
        #     unaligned_df = pl.DataFrame({"genome": list(unaligned_queries), "KEGG_flanks": [0]*len(unaligned_queries), "Pfam_flanks": [0]*len(unaligned_queries)})
        #     summary_df = pl.concat([summary_df, unaligned_df])
        #     summary_df.write_csv(os.path.join(args.output, "genome_summary.tsv"), separator="\t")
    
    # Determine flanks and write final results
    if os.path.exists(os.path.join(args.output, "final_results.tsv")):
        logging.info(f"Final summary already exists at {os.path.join(args.output, 'final_results.tsv')}. Nothing else to do here.")
    else:
        logging.info(f"Determining whether input sequences are flanked by non-viral regions and writing final results to {os.path.join(args.output, 'final_results.tsv')}.")
        final_results_df = summarize_regions(summary_df, args.max_kegg_lscore, args.max_pfam_lscore, args.min_annot)
        final_results_df.write_csv(os.path.join(args.output, "final_results.tsv"), separator="\t")
    
if __name__ == "__main__":
    args = parse_args()
    os.environ["POLARS_MAX_THREADS"] = str(args.threads)
    import polars as pl
    main(args)
