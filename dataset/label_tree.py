"""
the labels of different datasets
"""
nucleus_label_codebook = {}

# NuCLS
nucleus_label_codebook["fov"] = -1
nucleus_label_codebook["tumor"] = 0
nucleus_label_codebook["fibroblast"] = 1
nucleus_label_codebook["lymphocyte"] = 2
nucleus_label_codebook["plasma_cell"] = 3
nucleus_label_codebook["macrophage"] = 4
nucleus_label_codebook["mitotic_figure"] = 5
nucleus_label_codebook["vascular_endothelium"] = 6
nucleus_label_codebook["myoepithelium"] = 7
nucleus_label_codebook["apoptotic_body"] = 8
nucleus_label_codebook["neutrophil"] = 9
nucleus_label_codebook["ductal_epithelium"] = 10
nucleus_label_codebook["eosinophil"] = 11
nucleus_label_codebook["unlabeled"] = -1

# BreCaHAD
nucleus_label_codebook["mitosis"] = 12
nucleus_label_codebook["non_mitosis"] = 13
nucleus_label_codebook["apoptosis"] = 14
nucleus_label_codebook["tumor"] = 15 # NuCLS
nucleus_label_codebook["non_tumor"] = 16
nucleus_label_codebook["lumen"] = 17
nucleus_label_codebook["non_lumen"] = 18

# CoNSeP
nucleus_label_codebook["other"] = 19
nucleus_label_codebook["inflammatory"] = 20
nucleus_label_codebook["healthy_epithelial"] = 21
nucleus_label_codebook["malignant_epithelial"] = 22
nucleus_label_codebook["fibroblast"] = 23 # NuCLS
nucleus_label_codebook["muscle"] = 24
nucleus_label_codebook["endothelial"] = 25

# MoNuSAC
nucleus_label_codebook["Epithelial"] = 26
nucleus_label_codebook["Lymphocyte"] = 27
nucleus_label_codebook["Neutrophil"] = 28 # NuCLS
nucleus_label_codebook["Macrophage"] = 29

# panNuke
nucleus_label_codebook["Neoplastic"] = 30
nucleus_label_codebook["Inflammatory"] = 31 # CoNSeP
nucleus_label_codebook["Connective"] = 32
nucleus_label_codebook["Dead"] = 33
nucleus_label_codebook["Epithelial"] = 34 # MoNuSAC
