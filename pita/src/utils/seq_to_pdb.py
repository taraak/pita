"""
Generating peptide pdb file with tLEaP. Requires `ambertools` from conda-forge
to be installed in the current env.
"""

import os
import shutil
import tempfile

aa_321 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}

infile_templ = """source oldff/leaprc.ff99SBildn
peptide = sequence {%s}
savepdb peptide output.pdb
quit
"""


def translate_1letter_to_3letter(one_letter_seq, zwitter_ion=True):
    """Generate 3 letter sequence with 1-letter seq.
    If `zwitter_ion`, adding N and C terminal special label for zwitter ion form.
    """
    three_letter_seq = []
    for i, one_letter in enumerate(one_letter_seq):
        my_res_name = aa_321[one_letter]
        if zwitter_ion:
            if i == 0:
                my_res_name = "N" + my_res_name
            elif i == len(one_letter_seq) - 1:
                my_res_name = "C" + my_res_name
        three_letter_seq.append(my_res_name)
    return three_letter_seq


def make_peptide_with_tleap(three_letter_seq, save_path):
    """Make a pdb file for a three letter amino acid sequence and save
    it to the `save_path`.

    Note:
    One is expected to specify the terminal groups with the sequence.
    E.g., for capping with ACE and NME residue, one need to put ACE
     as the first residue and NME as the last residue.
    For zwitter ion form of ending, one need to attach a N label to the front
    of the first residue, e.g., NMET; as well as a C label to the front of the
    last residue, e.g., CALA.
    """
    current_work_dir = os.getcwd()
    save_path = os.path.abspath(save_path)
    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.chdir(tmpdirname)
            with open("temp.in", "w") as f:
                script = infile_templ % (" ".join(three_letter_seq))
                f.write(script)
            os.system("tleap -s -f temp.in > /dev/null")
            os.remove("leap.log")
            os.remove("temp.in")
            shutil.copy("output.pdb", save_path)
            os.remove("output.pdb")
    finally:
        os.chdir(current_work_dir)


if __name__ == "__main__":
    for sequence in [
        "A",
        "AA",
        "AAA",
        "AAAA",
        "AAAAA",
        "AAAAAA",
        "AAAAAAA",
        "AAAAAAAAA",
    ]:
        print(f"Generating zwitter-ion form peptide {sequence} into '{sequence}.pdb'")
        make_peptide_with_tleap(translate_1letter_to_3letter(sequence), f"{sequence}.pdb")
        print(f"Generating capped peptide {sequence} into '{sequence}_capped.pdb'")
        make_peptide_with_tleap(
            ["ACE"] + translate_1letter_to_3letter(sequence, zwitter_ion=False) + ["NME"],
            f"{sequence}_capped.pdb",
        )
