# Text Detoxification

This repo contains the code and data of the paper: [Text Detoxification as Style Transfer in English and Hindi](https://aclanthology.org/2023.icon-1.13/)

## Overview
This repository focuses on text detoxification, which involves automatically transforming toxic text into nontoxic text, contributing to safer online communication. We frame this task as a Text Style Transfer (TST) problem, where the style changes, but the content remains intact. Our research presents three approaches: (i) knowledge transfer from related tasks, (ii) a multi-task learning approach that integrates sequence-to-sequence modeling with various toxicity classification tasks, and (iii) a delete-and-reconstruct method. We utilize a dataset from Dementieva et al. (2021), which includes multiple detoxified versions of toxic texts. Our experiments, guided by expert human annotators, create a dataset where each toxic sentence is paired with an appropriate detoxified version. Additionally, we introduce a small Hindi parallel dataset aligned with part of the English dataset for evaluation. Our results show that our methods effectively detoxify text while preserving its content and fluency.

## Data
You can find the data and all the necessary details [here](https://github.com/panlingua/multilingual_text_detoxification_datasets).

## Walkthrough
*Will add more information in this section soon.*

### Dependency

    pip install -r <requirements.txt>

## Citing
If you use this data or code, please cite the following:
  
    @inproceedings{sourabrata-etal-2023-text,
    title = "Text Detoxification as Style Transfer in {E}nglish and {H}indi",
    author = "Mukherjee, Sourabrata  and
      Bansal, Akanksha  and
      Kr. Ojha, Atul  and
      P. McCrae, John  and
      Dusek, Ondrej",
    editor = "D. Pawar, Jyoti  and
      Lalitha Devi, Sobha",
    booktitle = "Proceedings of the 20th International Conference on Natural Language Processing (ICON)",
    month = dec,
    year = "2023",
    address = "Goa University, Goa, India",
    publisher = "NLP Association of India (NLPAI)",
    url = "https://aclanthology.org/2023.icon-1.13",
    pages = "133--144",
    abstract = "",
}

## License

    Author: Sourabrata Mukherjee
    Copyright © 2023 Sourabrata Mukherjee.
    Licensed under the MIT License.

## Acknowledgements

This research was funded by the European Union (ERC, NG-NLG, 101039303) and by Charles University projects GAUK 392221 and SVV 260698. We acknowledge the use of resources provided by the LINDAT/CLARIAH-CZ Research Infrastructure (Czech Ministry of Education, Youth, and Sports project No. LM2018101). We also acknowledge Panlingua Language Processing LLP for collaborating on this research project. Atul Kr. Ojha would like to acknowledge the support of the Science Foundation Ireland (SFI) as part of Grant Number SFI/12/RC/2289_P2 Insight_2, Insight SFI Research Centre for Data Analytics.

