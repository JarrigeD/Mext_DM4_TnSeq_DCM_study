"""
Author:     Domitille Jarrige
Date:       2023-02-10
Title:      Counts processing
Purpose:    For each gene in a genome, search for the largest uninterrupted CDS
            closest to the START codon. Produces a CDS statistics file for
            essentiality analyses and a wig file of corrected insertion sites
            for visualisation in TRANSIT GUI. This script is intended to run after
            TRANSIT transposon mapping, on the counts file. It can be parallelized
            by using the parameter "-t or --threads".
"""

# ______________________________________________________
# Modules import
# ______________________________________________________
import sys
import os
import time
import warnings
import getopt

import multiprocessing as mp
import pandas as pd


# Ignore pandas warnings on returning view versus copy.  See link below for more information:
# https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
warnings.filterwarnings(action="ignore", module="pandas")

# ______________________________________________________
# Global variables
# ______________________________________________________

SHIFT = 8  # nt value of the shift in coordinates incurred by transposon insertion in forward sense.
FRAME_INTERVAL = 100  # nt size of the minimal interval necessary to ignore an insertion in the CDS frame analysis.
TSS_5_UTR = 250 # nt size of the potential 5' UTR to scan for insertion downstream of orphan TSS
TSS_PROMOTER = 100 # nt size of the potential promoter to scan for insertion upstream of orphan TSS

# ______________________________________________________
# Functions
# ______________________________________________________

def read_files(annot_file, counts_file):
    """Read the annotation file and the TnSeq count file.
    Returns two pandas dataframes.
    :param annot_file: annotation file
    :param counts_file: annotated counts file from Transit TPP
    :return: pandas dataframes
    """
    annot_df = pd.read_csv(annot_file, sep=",",
                           header=0, names=["1", "#Orf", "3", "start", "end", "length", "frame", "8", "9", "10",
                                            "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22"])
    counts_df = pd.read_csv(counts_file, sep="\t")
    return annot_df, counts_df


def shift_FW_insertions(counts_df):
    """Shift the FW insertion coordinates to take into account
    the duplications caused by transposon insertion.
    Assumes the studied genetic element is circular.
    :param counts_df: counts pandas dataframe.
    :return: shifted pandas dataframe
    """
    list_FW_end = counts_df["Fwd_Rd_Ct"].iloc[-SHIFT:]
    counts_df["Fwd_Rd_Ct"] = counts_df["Fwd_Rd_Ct"].shift(SHIFT)
    counts_df["Fwd_Rd_Ct"].iloc[:SHIFT] = list_FW_end.astype(int)
    counts_df["Fwd_Rd_Ct"] = counts_df["Fwd_Rd_Ct"].astype(int)
    counts_df["Tot_Rd_Ct"] = counts_df["Fwd_Rd_Ct"] + counts_df["Rev_Rd_Ct"]
    return counts_df


def assign_coordinates(annot_file, counts_file, tss_file=None, out_tss=None, no_cds=False):
    """Annotate CDS on the raw counts file genomic coordinates.
    :param annot_file: annotation file.
    :param counts_file: counts file from Transit TPP.
    :param tss_file: TSS file, optional, default None.
    :param out_tss: TSS analysis out file, optional, default None.
    :param no_cds: Deactivate CDS analyses. Default False.
    :return: annotated counts dataframe of CDS coordinates.
    """
    annot_df, count_df = read_files(annot_file, counts_file)
    count_df = shift_FW_insertions(count_df)
    if tss_file:
        tss_scan(count_df, tss_file, out_tss)

    if no_cds:
        return

    # initialisation of lists
    list_coor, list_inser, list_gene = [], [], []
    list_gene_len, list_strand, list_frame = [], [], []
    list_orientation_FW, list_orientation_RV = [], []
    list_gene_position, list_relative_gene_position = [], []

    for index, row in annot_df.iterrows():
        # recovering annotation data, one gene at a time
        gene_id = row["#Orf"]
        start = row["start"]
        end = row["end"]
        def sign(x): return "+" if x > 0 else "-" if x < 0 else 0
        strand = sign(row["frame"])
        length = end - start + 1

        # counts dataframe iteration to recover insertion data
        # python indices start at 0 not the genomic coordinates! Correction with -1
        try:
            for k in range(start - 1, end):
                relative_pos = round(((k + 1 - start + 1) * 100 / length), 2)

                list_coor.append(count_df.loc[k, "coord"])
                list_inser.append(count_df.loc[k, "Tot_Rd_Ct"])
                list_gene.append(gene_id)
                list_gene_len.append(length)
                list_strand.append(strand)
                list_frame.append(((k + 1 - start) % 3) + 1)
                list_orientation_FW.append(count_df.loc[k, "Fwd_Rd_Ct"])
                list_orientation_RV.append(count_df.loc[k, "Rev_Rd_Ct"])
                list_gene_position.append(k + 1 - start + 1)
                list_relative_gene_position.append(relative_pos)

        except (TypeError, KeyError):
            print(f"WARNING: coordinates of gene {gene_id} not found in counts file. Gene ignored.")
            continue

    dico = {"coord": list_coor,
            "nb_insertions": list_inser,
            "gene": list_gene,
            "gene_length": list_gene_len,
            "insert_position": list_gene_position,
            "relative_insert_position_%": list_relative_gene_position,
            "gene_strand": list_strand,
            "in_frame": list_frame,
            "orientation_FW": list_orientation_FW,
            "orientation_RV": list_orientation_RV}
    final_df = pd.DataFrame(dico)
    return final_df



def batch_frame_search(annotated_counts_df, outfile, threads):
    """Calculate batch size and launch batch processes
    :param annotated_counts_df: annotated counts dataframe from assign_coordinates.
    :param outfile: output CDS statistics table file for essentiality analyses.
    :param threads: number of processes to run in parallel.
    :return: genomic coordinates list of discarded insertions from the frame analysis.
    """
    counts_df = annotated_counts_df
    genes_list = list(counts_df.groupby("gene").count().index)

    # Compute batches of genes to parallelize calculations
    batch_size = len(genes_list) // threads
    batches = [genes_list[(batch_size * i):(batch_size * (i + 1) if i + 1 != threads else len(genes_list))] for i in
               range(threads)]
    print(f"Running on {threads} processes. With batch size {batch_size}.")

    # Create pool of processes to parallelize calculations
    with mp.Pool(threads) as pool:
        res = [pool.apply(search_frames, kwds={"partial_counts_df": counts_df[counts_df["gene"].isin(batch)],
                                               "batch": batch}) for batch in batches]

    list_gene_name, list_gene_length, list_gene_strand, list_longest, list_percentage = [], [], [], [], []
    list_raw_sites, list_new_sites, list_density, list_new_density, list_sites_to_ignore = [], [], [], [], []
    for di in res:
        list_gene_name.extend(di["gene"])
        list_sites_to_ignore.extend(di["list_sites_to_ignore"])
        list_gene_length.extend(di["gene_length"])
        list_gene_strand.extend(di["strand"])
        list_longest.extend(di["longest_frame"])
        list_percentage.extend(di["percentage"])
        list_raw_sites.extend(di["sites"])
        list_new_sites.extend(di["corrected_sites"])
        list_density.extend(di["density"])
        list_new_density.extend(di["new_density"])

    dico = {"gene": list_gene_name,
            "gene_length": list_gene_length,
            "gene_strand": list_gene_strand,
            "longest_uninterrupted_intern_frame": list_longest,
            "%_uninterrupted_intern_frame": list_percentage,
            "raw_insertion_sites": list_raw_sites,
            "corrected_insertion_sites": list_new_sites,
            "density": list_density,
            "corrected_density": list_new_density}

    final_df = pd.DataFrame(dico)
    final_df.to_csv(outfile, index=False, sep="\t")
    return list_sites_to_ignore



def search_frames(partial_counts_df, batch):
    """Search the annotated counts dataframe for the longest uninterrupted CDS of each gene.
    :param partial_counts_df: partial annotated counts dataframe of genes in batch.
    :param batch: list of genes to process in the batch.
    :return: dictionary of statistics from the frame analysis for the batch of genes.
    """
    counts_df = partial_counts_df
    genes_list = batch
    list_longest_frame, list_gene, list_gene_len = [], [], []
    list_strand, list_percentage, list_sites, list_sites_to_ignore = [], [], [], []
    list_corrected_sites, list_density, list_new_density = [], [], []

    for gene in genes_list:
        total_sites = 0
        total_corrected_sites = 0
        tmp_df = counts_df[counts_df["gene"] == gene]
        tmp_df.index = range(len(tmp_df))
        res = scan_gene_intern(tmp_df)
        longest_frame = res[0]
        list_coor_insert = res[1]
        gene_length = tmp_df.loc[0, "gene_length"]

        list_longest_frame.append(int(longest_frame))
        list_gene.append(gene)
        list_gene_len.append(gene_length)
        list_strand.append(tmp_df.loc[0, "gene_strand"])
        list_percentage.append(round((longest_frame / gene_length * 100), 2))

        # Compute density of insertions sites, corrected or total
        for index, row in tmp_df.iterrows():
            if row["nb_insertions"] != 0:
                total_sites += 1

                if row["insert_position"] not in list_coor_insert:
                    total_corrected_sites += 1

                elif row["insert_position"] in list_coor_insert:
                    # Total list of insertions to not take into account
                    list_sites_to_ignore.append(row["coord"])

        list_sites.append(total_sites)
        list_corrected_sites.append(total_corrected_sites)
        list_density.append(total_sites / gene_length)
        list_new_density.append(total_corrected_sites / gene_length)

    dico = {"gene": list_gene,
            "gene_length": list_gene_len,
            "strand": list_strand,
            "longest_frame": list_longest_frame,
            "percentage": list_percentage,
            "sites": list_sites,
            "corrected_sites": list_corrected_sites,
            "density": list_density,
            "new_density": list_new_density,
            "list_sites_to_ignore": list_sites_to_ignore}

    return dico



def tss_scan(counts, tss_file, out_file):
    """Analyse of orphan TSS sites. Determine whether the potential
    5' UTR downstream is interrupted by transposon insertion sites.
    :param counts: shifted counts_df.
    :param tss_file: tsv file containing TSS  indications.
    :param out_file: name of the output TSS essentiality file.
    """
    tss_df = pd.read_csv(tss_file, sep="\t")
    list_tss = [(row["Position"], row["Strand.x"]) for index, row in tss_df.iterrows()]
    list_essential_5utr_tss = []
    list_essential_prom_tss = []
    for pos, strand in list_tss:
        if strand == "Plus":
            # takes into account only transposons inserted in reverse orientation to the TSS
            if (pos + (TSS_5_UTR + 1)) <= len(counts):
                insertion_5utr_list = [counts.iloc[i, 0] for i in range(pos, pos+TSS_5_UTR) if counts.iloc[i, 1]]
            else:
                insertion_5utr_list = [counts.iloc[i, 0] for i in range(pos, len(counts)-pos) if counts.iloc[i, 1]]

            if (pos - (TSS_PROMOTER + 1)) >= 1:
                insertion_prom_list = [counts.iloc[i, 0] for i in range(pos-TSS_PROMOTER, pos) if counts.iloc[i, 1]]
            else:
                insertion_prom_list = [counts.iloc[i, 0] for i in range(1, pos) if counts.iloc[i, 1]]

        elif strand == "Minus":
            # takes into account only transposons inserted in reverse orientation to the TSS
            if (pos - (TSS_5_UTR + 1)) >= 1:
                insertion_5utr_list = [counts.iloc[i, 0] for i in range(pos-1, pos-TSS_5_UTR, -1) if counts.iloc[i, 3]]
            else:
                insertion_5utr_list = [counts.iloc[i, 0] for i in range(len(counts)-pos, pos, -1) if counts.iloc[i, 3]]

            if (pos + (TSS_PROMOTER + 1)) <= len(counts):
                insertion_prom_list = [counts.iloc[i, 0] for i in range(pos+TSS_PROMOTER, pos, -1) if counts.iloc[i, 3]]
            else:
                insertion_prom_list = [counts.iloc[i, 0] for i in range(len(counts)+1, pos, -1) if counts.iloc[i, 3]]

        list_essential_5utr_tss.append(len(insertion_5utr_list))
        list_essential_prom_tss.append(len(insertion_prom_list))

    tss_df["promoter_insertion_sites"] = list_essential_prom_tss
    tss_df["5'UTR_insertion_sites"] = list_essential_5utr_tss
    tss_df.to_csv(out_file, sep="\t", index=False)


def update_counts(raw_counts_file, list_coord, outfile):
    """Create an updated wig file with all the genomic coordinates minus the ignored
    insertion sites from the frame analysis (e.g. insertion in frame or further away
    than FRAME_INTERVAL from the next insertion).
    param raw_counts_file: raw counts file from TPP.
    param list_coord: list of genomic coordinates of insertions that were discarded
                      in the frame analysis.
    param outfile: corrected counts file path.
    """
    counts_df = pd.read_csv(raw_counts_file, sep="\t")

    # Shift the raw counts of Fwd insertions
    counts_df = shift_FW_insertions(counts_df)
    counts_df["Kept_Rd_Ct"] = counts_df["Tot_Rd_Ct"]

    # Discard selected insertions
    for coor in list_coord:
        counts_df.loc[coor-1, "Kept_Rd_Ct"] = 0

    counts_df.iloc[:, [0, 7]].to_csv("tmp.tsv", index=False, sep="\t")
    with open(outfile, "w") as f_out:
        f_out.write(f"# Corrected wig file generated by counts_processing.py from {count_file}, only insertions"
                    f" in CDS, not in frame, and not more distant than 100bp from the next one are kept.\n")
        with open("tmp.tsv", "r") as f_tmp:
            for line in f_tmp:
                f_out.write(line)
    os.remove("tmp.tsv")
    return


def scan_gene_intern(tmp_df):
    """Scan a gene to search for the biggest uninterrupted frame whatever the position.
    Ignore in frame insertions and insertions further away than FRAME_INTERVAL from the
    next insertion site. Insertion sites in the STOP codons are also ignored.
    :param tmp_df : annotated counts sub-dataframe for the gene.
    :return: the largest uninterrupted frame and a list of discarded insertion sites coordinates.
    """
    list_segments = []
    list_insertions_to_trash = []

    # Process strand + and strand - genes separately.
    if tmp_df.loc[0, "gene_strand"] == "+":
        set_insert = set(tmp_df[tmp_df["nb_insertions"] != 0]["insert_position"])
        set_in_frame_insert = set(tmp_df[(tmp_df["orientation_RV"] != 0) &
                                         (tmp_df["in_frame"] == 1)]["insert_position"])
        list_not_in_frame_insert = list(set_insert.difference(set_in_frame_insert))
        list_insertions_to_trash = list(set_in_frame_insert)
        list_insertions_to_trash.sort()

        # If there is only one or zero insertion out of frame in the gene,
        # returns gene length and in frame insertions to discard.
        if len(list_not_in_frame_insert) <= 1:
            return tmp_df.loc[0, "gene_length"], list_insertions_to_trash
        
        else:
            list_not_in_frame_insert.append(tmp_df.iloc[0, 4])
            list_not_in_frame_insert.append(tmp_df.iloc[-1, 4])
            list_not_in_frame_insert = list(set(list_not_in_frame_insert))
            list_not_in_frame_insert.sort()
            begin = list_not_in_frame_insert[0]
            for i in range(len(list_not_in_frame_insert) - 2):
                # If next not in frame insertion is closer to FRAME_INTERVAL bp.
                if (list_not_in_frame_insert[i + 2] - list_not_in_frame_insert[i + 1]) < FRAME_INTERVAL:
                    list_segments.append(list_not_in_frame_insert[i + 1] - begin)
                    begin = list_not_in_frame_insert[i + 1]

                # If next not in frame insertion is further than FRAME_INTERVAL bp.
                elif (list_not_in_frame_insert[i + 2] - list_not_in_frame_insert[i + 1]) >= FRAME_INTERVAL:
                    list_insertions_to_trash.append(list_not_in_frame_insert[i + 1])

                # What to do when reaching the end of the CDS.
                if i == (len(list_not_in_frame_insert) - 3):
                    x = list_not_in_frame_insert[i + 1] - begin
                    y = list_not_in_frame_insert[i + 2] - list_not_in_frame_insert[i + 1]
                    if y >= FRAME_INTERVAL:
                        list_insertions_to_trash.append(list_not_in_frame_insert[i + 1])
                        list_segments.append(x + y)
                    else:
                        list_segments.append(x)
                        list_segments.append(y)

    # Process strand + and strand - genes separately.
    elif tmp_df.loc[0, "gene_strand"] == "-":
        set_insert = set(tmp_df[tmp_df["nb_insertions"] != 0]["insert_position"])
        set_in_frame_insert = set(tmp_df[(tmp_df["orientation_FW"] != 0) &
                                         (tmp_df["in_frame"] == 1)]["insert_position"])
        list_not_in_frame_insert = list(set_insert.difference(set_in_frame_insert))
        list_insertions_to_trash = list(set_in_frame_insert)
        list_insertions_to_trash.sort()
        
        # If there is only one or zero insertion out of frame in the gene,
        # returns gene length and in frame insertions to discard.
        if len(list_not_in_frame_insert) <= 1:
            return (tmp_df.loc[0, "gene_length"]), list_insertions_to_trash
        
        else:
            list_not_in_frame_insert.append(tmp_df.iloc[0, 4])
            list_not_in_frame_insert.append(tmp_df.iloc[-1, 4])
            list_not_in_frame_insert = list(set(list_not_in_frame_insert))
            list_not_in_frame_insert.sort()
            list_not_in_frame_insert.reverse()
            begin = list_not_in_frame_insert[0]
            for i in range(len(list_not_in_frame_insert) - 2):
                # If next not in frame insertion is closer to FRAME_INTERVAL bp
                if (list_not_in_frame_insert[i + 1] - list_not_in_frame_insert[i + 2]) < FRAME_INTERVAL:
                    list_segments.append(begin - list_not_in_frame_insert[i + 1])
                    begin = list_not_in_frame_insert[i + 1]

                # If next not in frame insertion is further than FRAME_INTERVAL bp.
                elif (list_not_in_frame_insert[i + 1] - list_not_in_frame_insert[i + 2]) >= FRAME_INTERVAL:
                    list_insertions_to_trash.append(list_not_in_frame_insert[i + 1])

                # What to do when reaching the end of the CDS.
                if i == (len(list_not_in_frame_insert) - 3):
                    x = begin - list_not_in_frame_insert[i + 1]
                    y = list_not_in_frame_insert[i + 1] - list_not_in_frame_insert[i + 2]
                    if y >= FRAME_INTERVAL:
                        list_segments.append(x + y)
                        list_insertions_to_trash.append(list_not_in_frame_insert[i + 1])
                    else:
                        list_segments.append(x)
                        list_segments.append(y)

    # Remove insertions in stop codon
    for stop in tmp_df.iloc[-3:, 4]:
        try:
            list_insertions_to_trash.remove(stop)
        except ValueError:
            continue

    list_insertions_to_trash.sort()
    return max(list_segments), list_insertions_to_trash


# ______________________________________________________
# Main program
# ______________________________________________________

options = "hi:a:o:t:T:d:x"
long_options = ["help", "raw_counts_file=", "annotation_file=", "output_prefix=",
                "threads=", "tss_file=", "results_directory=", "no_cds_analyses"]
# Check for top-level environment
if __name__ == '__main__':
    # Create pool of processes to parallelize calculations
    #pool = mp.Pool(threads)

    # Parse arguments
    try:
        opts, args = getopt.getopt(sys.argv[1:], options, long_options)
    except getopt.GetoptError:
        print("Usage: python counts_processing.py -i <count_file> -a <annotation_file> "
              "-o <output_prefix> [optional: -t <threads> -T <tss_file> -d <results_directory> -x]\n")
        print('Try "python counts_processing.py -h [--help]" for more information.')
        sys.exit(2)

    # default if no TSS file is provided
    tss = "None provided. Option not activated."
    directory = False
    opt_cds = False
    threads = 1

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("Usage: python counts_processing.py -i <count_file> -a <annotation_file> "
                  "-o <output_prefix> [optional: -t <threads> -T <tss_file> -d <results_directory> -x]\n")
            print("""Mandatory arguments:
                -i, --raw_counts_file    input transposon counts file from TRANSIT.
                -a, --annotation_file    annotation csv file from MAGE (LABGeM).
                -o, --output             prefix output of results files.\n
                Optional arguments:"
                -t, --threads            thread number to use (default 1)
                -T, --tss_file           tsv TSS annotations table.
                -d, --results_directory  path of the desired results directory.
                -x, --no_cds_analyses    deactivate CDS analyses.\n""")
            sys.exit()

        elif opt in ("-i", "--raw_counts_file"):
            count_file = arg
        elif opt in ("-a", "--annotation_file"):
            annotation_file = arg
        elif opt in ("-o", "--output"):
            output = arg
        elif opt in ("-t", "--threads"):
            threads = int(arg)
        elif opt in ("-T", "--tss_file"):
            tss = arg
        elif opt in ("-d", "--results_directory"):
            directory = arg
        elif opt in ("-x", "--no_cds_analyses"):
            opt_cds = True

    try:
        print(f"*********************************************\n"
              f"*             Command and files:            *\n"
              f"*********************************************\n"
              f"raw_counts_file: {count_file}\n"
              f"annotation file: {annotation_file}\n"
              f"output prefix  : {output}\n"
              f"TSS file       : {tss}\n"
              f"threads        : {threads}")
        if opt_cds:
            print("CDS analyses option deactivated")
        if directory:
            print("Output directory: ", directory)

    except NameError:
        print("Usage: python counts_processing.py -i <count_file> -a <annotation_file> "
              "-o <output_prefix> [optional: -t <threads> -T <tss_file> -d <results_directory> -x]\n")
        print('Try "python counts_processing.py -h [--help]" for more information.')
        sys.exit(2)

    if opt_cds:
        print("*********************************************")
        print("*                 Working...                *")
        print("*********************************************")

        if tss != "None provided. Option not activated.":
            if directory:
                out_tss_file = directory + "/" + output + "_" + \
                               os.path.splitext(os.path.basename(tss))[0] + ".tsv"
            else:
                out_tss_file = output + "_" + os.path.splitext(os.path.basename(tss))[0] + ".tsv"
            #annotation_df = pool.apply(assign_coordinates,
            #                           kwds={"annot_file": annotation_file, "counts_file": count_file,
            #                                 "tss_file": tss, "out_tss": out_tss_file, "no_cds": opt_cds})
            #pool.close()
            annotation_df = assign_coordinates(annot_file=annotation_file, counts_file=count_file,
                                               tss_file=tss, out_tss=out_tss_file, no_cds=opt_cds)
        else:
            print("No work to be done!\n")
            print("Usage: python counts_processing.py -i <count_file> -a <annotation_file> "
                  "-o <output_prefix> [optional: -t <threads> -T <tss_file> -d <results_directory> -x]\n")
            sys.exit(2)

    else:
        print("*********************************************")
        print("*      Annotating CDS in counts file...     *")
        print("*********************************************")

        print(f"Start time: {time.localtime()[0]}-{time.localtime()[1]}-{time.localtime()[2]} "
              f"{time.localtime()[3]}:{time.localtime()[4]}:{time.localtime()[5]}")

        if tss != "None provided. Option not activated.":
            if directory:
                out_tss_file = directory + "/" + output + "_" + \
                               os.path.splitext(os.path.basename(tss))[0] + ".tsv"
            else:
                out_tss_file = output + "_" + os.path.splitext(os.path.basename(tss))[0] + ".tsv"
            #annotation_df = pool.apply(assign_coordinates,
            #                           kwds={"annot_file": annotation_file, "counts_file": count_file,
            #                                 "tss_file": tss, "out_tss": out_tss_file})
            annotation_df = assign_coordinates(annot_file=annotation_file, counts_file=count_file,
                                               tss_file=tss, out_tss=out_tss_file)
            #pool.close()
        else:
            #annotation_df = pool.apply(assign_coordinates,
            #                           kwds={"annot_file": annotation_file, "counts_file": count_file})
            annotation_df = assign_coordinates(annot_file=annotation_file, counts_file=count_file)
            #pool.close()


        print("_____\nAnnotation of CDS complete.\n_____")

        print(f"End time: {time.localtime()[0]}-{time.localtime()[1]}-{time.localtime()[2]} "
              f"{time.localtime()[3]}:{time.localtime()[4]}:{time.localtime()[5]}")

        print("*********************************************")
        print("* Computing longest uninterrupted frames... *")
        print("*********************************************")

        print(f"Start time: {time.localtime()[0]}-{time.localtime()[1]}-{time.localtime()[2]} "
              f"{time.localtime()[3]}:{time.localtime()[4]}:{time.localtime()[5]}")

        if directory:
            out_wig_file = directory + output + "_corrected.wig"
            out_cds_file = directory + output + "_cds_stats.tsv"
        else:
            out_wig_file = output + "_corrected.wig"
            out_cds_file = output + "_cds_stats.tsv"

        sites_to_ignore = batch_frame_search(annotated_counts_df=annotation_df, outfile=out_cds_file, threads=threads)
        update_counts(raw_counts_file=count_file, list_coord=sites_to_ignore, outfile=out_wig_file)

        print(f"_____\nShift of FW transposon insertion sites:       {SHIFT} bp.\n"
              f"Minimal interval for frame search analyses:   {FRAME_INTERVAL} bp.\n_____\n")
        print(f"Gene stats file:                              {out_cds_file}")
        print(f"Corrected genomic wig file for visualisation: {out_wig_file}")

    if tss != "None provided. Option not activated.":
        print(f"TSS stats file:                               {out_tss_file}")

    print(f"_____\nEnd time: {time.localtime()[0]}-{time.localtime()[1]}-{time.localtime()[2]} "
          f"{time.localtime()[3]}:{time.localtime()[4]}:{time.localtime()[5]}")
