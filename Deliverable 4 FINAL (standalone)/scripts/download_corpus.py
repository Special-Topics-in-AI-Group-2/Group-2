"""download_corpus.py — fetch the real open-access PDF corpus for the agent.

The CSAI415 brief asks for "100-300 open-access PDFs from one topic (e.g. arXiv
cs.AI/cs.CL subset)".  We ship a curated, **coherent** ~102-paper slice of the
arXiv cs.CL / cs.LG / cs.IR literature on
*Transformers → pretraining → retrieval → parameter-efficient tuning* — the same
thread the seed data and gold Q/A already reference (Attention Is All You Need,
BERT, RAG).  14 hand-curated anchor papers carry rich metadata; ~88 more landmark
papers (EXTRA_CORPUS) are downloaded from the working arXiv PDF endpoint and
**title-verified** against each file so papers.csv always matches the PDFs.
Every file is a genuine paper downloaded from arXiv, not generated.  An optional
``--harvest N`` mode tops the corpus up via the arXiv metadata API when it is
not rate-limiting.

What it writes
--------------
  data/pdfs/<slug>.pdf          one real PDF per paper
  data/papers.csv               metadata table consumed by app/build_graph.py
                                (columns: paper id,title,authors,venue,year,topics)
  data/corpus_metadata.json     full provenance: arxiv_id, doi, url, license

Usage
-----
  python scripts/download_corpus.py                 # download all (skip existing)
  python scripts/download_corpus.py --limit 5       # first 5 only (quick demo)
  python scripts/download_corpus.py --out data/pdfs --force

Licensing / ethics
------------------
All papers are hosted open-access on arXiv.org.  arXiv distributes them under a
perpetual, non-exclusive license to redistribute; several are additionally
released by their authors under Creative Commons licenses.  We redistribute only
metadata + a download script here (the PDFs are fetched at run time from arXiv),
and we cite every paper by title, authors, venue, year and DOI.  See
reports/D4_Final_Report.md §Ethics & Licensing.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

# Windows consoles default to cp1252 and crash on non-ASCII output.  Force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


# ---------------------------------------------------------------------------
# Curated corpus — one coherent topic cluster (NLP / Transformers / RAG / PEFT)
# ---------------------------------------------------------------------------
# Each entry is a genuine arXiv paper.  `topics` drive the Neo4j (:Topic) nodes.

CORPUS: list[dict] = [
    {
        "paper_id": "P001", "arxiv_id": "1706.03762", "slug": "attention_is_all_you_need",
        "title": "Attention Is All You Need",
        "authors": "Ashish Vaswani; Noam Shazeer; Niki Parmar; Jakob Uszkoreit; Llion Jones; Aidan N. Gomez; Lukasz Kaiser; Illia Polosukhin",
        "venue": "NeurIPS", "year": 2017,
        "topics": "Transformers; Self-Attention; Sequence Modeling; NLP",
    },
    {
        "paper_id": "P002", "arxiv_id": "1810.04805", "slug": "bert",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "authors": "Jacob Devlin; Ming-Wei Chang; Kenton Lee; Kristina Toutanova",
        "venue": "NAACL", "year": 2019,
        "topics": "BERT; Pre-training; Transformers; NLP",
    },
    {
        "paper_id": "P003", "arxiv_id": "2005.11401", "slug": "rag",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": "Patrick Lewis; Ethan Perez; Aleksandra Piktus; Fabio Petroni; Vladimir Karpukhin; Naman Goyal; Heinrich Kuttler; Mike Lewis; Wen-tau Yih; Tim Rocktaschel; Sebastian Riedel; Douwe Kiela",
        "venue": "NeurIPS", "year": 2020,
        "topics": "RAG; Information Retrieval; Generation; NLP",
    },
    {
        "paper_id": "P004", "arxiv_id": "1907.11692", "slug": "roberta",
        "title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
        "authors": "Yinhan Liu; Myle Ott; Naman Goyal; Jingfei Du; Mandar Joshi; Danqi Chen; Omer Levy; Mike Lewis; Luke Zettlemoyer; Veselin Stoyanov",
        "venue": "arXiv", "year": 2019,
        "topics": "BERT; Pre-training; Transformers; NLP",
    },
    {
        "paper_id": "P005", "arxiv_id": "1910.10683", "slug": "t5",
        "title": "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
        "authors": "Colin Raffel; Noam Shazeer; Adam Roberts; Katherine Lee; Sharan Narang; Michael Matena; Yanqi Zhou; Wei Li; Peter J. Liu",
        "venue": "JMLR", "year": 2020,
        "topics": "Transfer Learning; Transformers; Pre-training; NLP",
    },
    {
        "paper_id": "P006", "arxiv_id": "2004.04906", "slug": "dpr",
        "title": "Dense Passage Retrieval for Open-Domain Question Answering",
        "authors": "Vladimir Karpukhin; Barlas Oguz; Sewon Min; Patrick Lewis; Ledell Wu; Sergey Edunov; Danqi Chen; Wen-tau Yih",
        "venue": "EMNLP", "year": 2020,
        "topics": "Information Retrieval; Dense Retrieval; Question Answering; NLP",
    },
    {
        "paper_id": "P007", "arxiv_id": "1908.10084", "slug": "sentence_bert",
        "title": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
        "authors": "Nils Reimers; Iryna Gurevych",
        "venue": "EMNLP", "year": 2019,
        "topics": "Sentence Embeddings; Dense Retrieval; BERT; NLP",
    },
    {
        "paper_id": "P008", "arxiv_id": "2106.09685", "slug": "lora",
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "authors": "Edward J. Hu; Yelong Shen; Phillip Wallis; Zeyuan Allen-Zhu; Yuanzhi Li; Shean Wang; Lu Wang; Weizhu Chen",
        "venue": "ICLR", "year": 2022,
        "topics": "Parameter-Efficient Fine-Tuning; LoRA; Transformers; NLP",
    },
    {
        "paper_id": "P009", "arxiv_id": "2305.14314", "slug": "qlora",
        "title": "QLoRA: Efficient Finetuning of Quantized LLMs",
        "authors": "Tim Dettmers; Artidoro Pagnoni; Ari Holtzman; Luke Zettlemoyer",
        "venue": "NeurIPS", "year": 2023,
        "topics": "Parameter-Efficient Fine-Tuning; Quantization; LoRA; NLP",
    },
    {
        "paper_id": "P010", "arxiv_id": "2203.02155", "slug": "instructgpt",
        "title": "Training Language Models to Follow Instructions with Human Feedback",
        "authors": "Long Ouyang; Jeff Wu; Xu Jiang; Diogo Almeida; Carroll L. Wainwright; Pamela Mishkin; Chong Zhang; Sandhini Agarwal; Katarina Slama; Alex Ray; et al.",
        "venue": "NeurIPS", "year": 2022,
        "topics": "Instruction Tuning; RLHF; Alignment; NLP",
    },
    {
        "paper_id": "P011", "arxiv_id": "2201.11903", "slug": "chain_of_thought",
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "authors": "Jason Wei; Xuezhi Wang; Dale Schuurmans; Maarten Bosma; Brian Ichter; Fei Xia; Ed Chi; Quoc Le; Denny Zhou",
        "venue": "NeurIPS", "year": 2022,
        "topics": "Reasoning; Prompting; Large Language Models; NLP",
    },
    {
        "paper_id": "P012", "arxiv_id": "2004.12832", "slug": "colbert",
        "title": "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT",
        "authors": "Omar Khattab; Matei Zaharia",
        "venue": "SIGIR", "year": 2020,
        "topics": "Information Retrieval; Dense Retrieval; BERT; NLP",
    },
    {
        "paper_id": "P013", "arxiv_id": "2312.10997", "slug": "rag_survey",
        "title": "Retrieval-Augmented Generation for Large Language Models: A Survey",
        "authors": "Yunfan Gao; Yun Xiong; Xinyu Gao; Kangxiang Jia; Jinliu Pan; Yuxi Bi; Yi Dai; Jiawei Sun; Qianyu Guo; Meng Wang; Haofen Wang",
        "venue": "arXiv", "year": 2023,
        "topics": "RAG; Information Retrieval; Large Language Models; Survey",
    },
    {
        "paper_id": "P014", "arxiv_id": "1301.3781", "slug": "word2vec",
        "title": "Efficient Estimation of Word Representations in Vector Space",
        "authors": "Tomas Mikolov; Kai Chen; Greg Corrado; Jeffrey Dean",
        "venue": "ICLR Workshop", "year": 2013,
        "topics": "Word Embeddings; Representation Learning; NLP",
    },
]

# ---------------------------------------------------------------------------
# Extended curated set — ~90 more landmark NLP / IR / LLM papers.
# Metadata is hand-authored (accurate); the PDF is fetched from the working
# arXiv PDF endpoint and title-verified at download time (mismatches are
# dropped), so papers.csv always matches the actual files even if an id is off.
# This is the API-free path used when the arXiv metadata API is rate-limited.
# ---------------------------------------------------------------------------

EXTRA_CORPUS: list[dict] = [
    # --- Foundational seq2seq / attention ---
    {"arxiv_id": "1409.0473", "title": "Neural Machine Translation by Jointly Learning to Align and Translate", "authors": "Dzmitry Bahdanau; Kyunghyun Cho; Yoshua Bengio", "venue": "ICLR", "year": 2015, "topics": "Machine Translation; Attention; NLP"},
    {"arxiv_id": "1409.3215", "title": "Sequence to Sequence Learning with Neural Networks", "authors": "Ilya Sutskever; Oriol Vinyals; Quoc V. Le", "venue": "NeurIPS", "year": 2014, "topics": "Sequence Modeling; Machine Translation; NLP"},
    {"arxiv_id": "1406.1078", "title": "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation", "authors": "Kyunghyun Cho; Bart van Merrienboer; Yoshua Bengio", "venue": "EMNLP", "year": 2014, "topics": "Machine Translation; Representation Learning; NLP"},
    {"arxiv_id": "1508.04025", "title": "Effective Approaches to Attention-based Neural Machine Translation", "authors": "Minh-Thang Luong; Hieu Pham; Christopher D. Manning", "venue": "EMNLP", "year": 2015, "topics": "Machine Translation; Attention; NLP"},
    # --- Contextual embeddings / pretraining ---
    {"arxiv_id": "1802.05365", "title": "Deep contextualized word representations", "authors": "Matthew E. Peters; Mark Neumann; Luke Zettlemoyer", "venue": "NAACL", "year": 2018, "topics": "Representation Learning; Pre-training; NLP"},
    {"arxiv_id": "1801.06146", "title": "Universal Language Model Fine-tuning for Text Classification", "authors": "Jeremy Howard; Sebastian Ruder", "venue": "ACL", "year": 2018, "topics": "Transfer Learning; Pre-training; NLP"},
    {"arxiv_id": "1906.08237", "title": "XLNet: Generalized Autoregressive Pretraining for Language Understanding", "authors": "Zhilin Yang; Zihang Dai; Quoc V. Le", "venue": "NeurIPS", "year": 2019, "topics": "Pre-training; Transformers; NLP"},
    {"arxiv_id": "1909.11942", "title": "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations", "authors": "Zhenzhong Lan; Mingda Chen; Radu Soricut", "venue": "ICLR", "year": 2020, "topics": "BERT; Pre-training; Efficiency"},
    {"arxiv_id": "1910.01108", "title": "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter", "authors": "Victor Sanh; Lysandre Debut; Thomas Wolf", "venue": "arXiv", "year": 2019, "topics": "BERT; Distillation; Efficiency"},
    {"arxiv_id": "2003.10555", "title": "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators", "authors": "Kevin Clark; Minh-Thang Luong; Christopher D. Manning", "venue": "ICLR", "year": 2020, "topics": "Pre-training; Transformers; NLP"},
    {"arxiv_id": "1910.13461", "title": "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension", "authors": "Mike Lewis; Yinhan Liu; Luke Zettlemoyer", "venue": "ACL", "year": 2020, "topics": "Pre-training; Generation; NLP"},
    {"arxiv_id": "1901.02860", "title": "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context", "authors": "Zihang Dai; Zhilin Yang; Ruslan Salakhutdinov", "venue": "ACL", "year": 2019, "topics": "Transformers; Language Modeling; NLP"},
    # --- Large language models / scaling ---
    {"arxiv_id": "2005.14165", "title": "Language Models are Few-Shot Learners", "authors": "Tom B. Brown; Benjamin Mann; Dario Amodei", "venue": "NeurIPS", "year": 2020, "topics": "Large Language Models; Few-Shot Learning; NLP"},
    {"arxiv_id": "2001.08361", "title": "Scaling Laws for Neural Language Models", "authors": "Jared Kaplan; Sam McCandlish; Dario Amodei", "venue": "arXiv", "year": 2020, "topics": "Large Language Models; Scaling Laws; NLP"},
    {"arxiv_id": "2203.15556", "title": "Training Compute-Optimal Large Language Models", "authors": "Jordan Hoffmann; Sebastian Borgeaud; Laurent Sifre", "venue": "NeurIPS", "year": 2022, "topics": "Large Language Models; Scaling Laws; Efficiency"},
    {"arxiv_id": "2204.02311", "title": "PaLM: Scaling Language Modeling with Pathways", "authors": "Aakanksha Chowdhery; Sharan Narang; Jacob Devlin", "venue": "JMLR", "year": 2023, "topics": "Large Language Models; Pre-training; NLP"},
    {"arxiv_id": "2302.13971", "title": "LLaMA: Open and Efficient Foundation Language Models", "authors": "Hugo Touvron; Thibaut Lavril; Guillaume Lample", "venue": "arXiv", "year": 2023, "topics": "Large Language Models; Pre-training; NLP"},
    {"arxiv_id": "2307.09288", "title": "Llama 2: Open Foundation and Fine-Tuned Chat Models", "authors": "Hugo Touvron; Louis Martin; Thomas Scialom", "venue": "arXiv", "year": 2023, "topics": "Large Language Models; Instruction Tuning; NLP"},
    {"arxiv_id": "2206.07682", "title": "Emergent Abilities of Large Language Models", "authors": "Jason Wei; Yi Tay; William Fedus", "venue": "TMLR", "year": 2022, "topics": "Large Language Models; Reasoning; NLP"},
    {"arxiv_id": "1909.08053", "title": "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism", "authors": "Mohammad Shoeybi; Mostofa Patwary; Bryan Catanzaro", "venue": "arXiv", "year": 2019, "topics": "Large Language Models; Efficiency; Pre-training"},
    {"arxiv_id": "2101.00027", "title": "The Pile: An 800GB Dataset of Diverse Text for Language Modeling", "authors": "Leo Gao; Stella Biderman; Connor Leahy", "venue": "arXiv", "year": 2020, "topics": "Large Language Models; Datasets; Pre-training"},
    {"arxiv_id": "2211.05100", "title": "BLOOM: A 176B-Parameter Open-Access Multilingual Language Model", "authors": "Teven Le Scao; Angela Fan; BigScience Workshop", "venue": "arXiv", "year": 2022, "topics": "Large Language Models; Multilingual; NLP"},
    {"arxiv_id": "2205.01068", "title": "OPT: Open Pre-trained Transformer Language Models", "authors": "Susan Zhang; Stephen Roller; Luke Zettlemoyer", "venue": "arXiv", "year": 2022, "topics": "Large Language Models; Pre-training; NLP"},
    # --- Efficient attention / architectures ---
    {"arxiv_id": "2205.14135", "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", "authors": "Tri Dao; Daniel Y. Fu; Christopher Re", "venue": "NeurIPS", "year": 2022, "topics": "Efficiency; Self-Attention; Transformers"},
    {"arxiv_id": "2307.08691", "title": "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", "authors": "Tri Dao", "venue": "arXiv", "year": 2023, "topics": "Efficiency; Self-Attention; Transformers"},
    {"arxiv_id": "2309.06180", "title": "Efficient Memory Management for Large Language Model Serving with PagedAttention", "authors": "Woosuk Kwon; Zhuohan Li; Ion Stoica", "venue": "SOSP", "year": 2023, "topics": "Efficiency; Serving; Large Language Models"},
    {"arxiv_id": "2312.00752", "title": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", "authors": "Albert Gu; Tri Dao", "venue": "arXiv", "year": 2023, "topics": "Sequence Modeling; Efficiency; State Space Models"},
    {"arxiv_id": "2004.05150", "title": "Longformer: The Long-Document Transformer", "authors": "Iz Beltagy; Matthew E. Peters; Arman Cohan", "venue": "arXiv", "year": 2020, "topics": "Transformers; Long Context; Efficiency"},
    {"arxiv_id": "2009.14794", "title": "Rethinking Attention with Performers", "authors": "Krzysztof Choromanski; Valerii Likhosherstov; Adrian Weller", "venue": "ICLR", "year": 2021, "topics": "Self-Attention; Efficiency; Transformers"},
    {"arxiv_id": "2006.04768", "title": "Linformer: Self-Attention with Linear Complexity", "authors": "Sinong Wang; Belinda Z. Li; Hao Ma", "venue": "arXiv", "year": 2020, "topics": "Self-Attention; Efficiency; Transformers"},
    {"arxiv_id": "2104.09864", "title": "RoFormer: Enhanced Transformer with Rotary Position Embedding", "authors": "Jianlin Su; Yu Lu; Yunfeng Liu", "venue": "arXiv", "year": 2021, "topics": "Transformers; Positional Encoding; NLP"},
    # --- Parameter-efficient fine-tuning ---
    {"arxiv_id": "1902.00751", "title": "Parameter-Efficient Transfer Learning for NLP", "authors": "Neil Houlsby; Andrei Giurgiu; Sylvain Gelly", "venue": "ICML", "year": 2019, "topics": "Parameter-Efficient Fine-Tuning; Transfer Learning; NLP"},
    {"arxiv_id": "2104.08691", "title": "The Power of Scale for Parameter-Efficient Prompt Tuning", "authors": "Brian Lester; Rami Al-Rfou; Noah Constant", "venue": "EMNLP", "year": 2021, "topics": "Parameter-Efficient Fine-Tuning; Prompting; NLP"},
    {"arxiv_id": "2101.00190", "title": "Prefix-Tuning: Optimizing Continuous Prompts for Generation", "authors": "Xiang Lisa Li; Percy Liang", "venue": "ACL", "year": 2021, "topics": "Parameter-Efficient Fine-Tuning; Generation; NLP"},
    {"arxiv_id": "2110.04366", "title": "Towards a Unified View of Parameter-Efficient Transfer Learning", "authors": "Junxian He; Chunting Zhou; Graham Neubig", "venue": "ICLR", "year": 2022, "topics": "Parameter-Efficient Fine-Tuning; Transfer Learning; NLP"},
    {"arxiv_id": "2303.16199", "title": "LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention", "authors": "Renrui Zhang; Jiaming Han; Yu Qiao", "venue": "arXiv", "year": 2023, "topics": "Parameter-Efficient Fine-Tuning; Instruction Tuning; NLP"},
    {"arxiv_id": "1804.07461", "title": "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding", "authors": "Alex Wang; Amanpreet Singh; Samuel R. Bowman", "venue": "ICLR", "year": 2019, "topics": "Benchmark; Evaluation; NLP"},
    # --- Instruction tuning / alignment / RLHF ---
    {"arxiv_id": "2109.01652", "title": "Finetuned Language Models Are Zero-Shot Learners", "authors": "Jason Wei; Maarten Bosma; Quoc V. Le", "venue": "ICLR", "year": 2022, "topics": "Instruction Tuning; Large Language Models; NLP"},
    {"arxiv_id": "2210.11416", "title": "Scaling Instruction-Finetuned Language Models", "authors": "Hyung Won Chung; Le Hou; Jason Wei", "venue": "arXiv", "year": 2022, "topics": "Instruction Tuning; Large Language Models; NLP"},
    {"arxiv_id": "2212.10560", "title": "Self-Instruct: Aligning Language Models with Self-Generated Instructions", "authors": "Yizhong Wang; Yeganeh Kordi; Hannaneh Hajishirzi", "venue": "ACL", "year": 2023, "topics": "Instruction Tuning; Alignment; NLP"},
    {"arxiv_id": "2305.11206", "title": "LIMA: Less Is More for Alignment", "authors": "Chunting Zhou; Pengfei Liu; Omer Levy", "venue": "NeurIPS", "year": 2023, "topics": "Instruction Tuning; Alignment; NLP"},
    {"arxiv_id": "2305.18290", "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", "authors": "Rafael Rafailov; Archit Sharma; Chelsea Finn", "venue": "NeurIPS", "year": 2023, "topics": "RLHF; Alignment; Large Language Models"},
    {"arxiv_id": "2204.05862", "title": "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback", "authors": "Yuntao Bai; Andy Jones; Jared Kaplan", "venue": "arXiv", "year": 2022, "topics": "RLHF; Alignment; Large Language Models"},
    {"arxiv_id": "2212.08073", "title": "Constitutional AI: Harmlessness from AI Feedback", "authors": "Yuntao Bai; Saurav Kadavath; Jared Kaplan", "venue": "arXiv", "year": 2022, "topics": "Alignment; RLHF; Large Language Models"},
    {"arxiv_id": "2009.01325", "title": "Learning to Summarize from Human Feedback", "authors": "Nisan Stiennon; Long Ouyang; Paul Christiano", "venue": "NeurIPS", "year": 2020, "topics": "RLHF; Summarization; NLP"},
    # --- Prompting / reasoning / agents ---
    {"arxiv_id": "2203.11171", "title": "Self-Consistency Improves Chain of Thought Reasoning in Language Models", "authors": "Xuezhi Wang; Jason Wei; Denny Zhou", "venue": "ICLR", "year": 2023, "topics": "Reasoning; Prompting; Large Language Models"},
    {"arxiv_id": "2205.11916", "title": "Large Language Models are Zero-Shot Reasoners", "authors": "Takeshi Kojima; Shixiang Shane Gu; Yusuke Iwasawa", "venue": "NeurIPS", "year": 2022, "topics": "Reasoning; Prompting; Large Language Models"},
    {"arxiv_id": "2210.03629", "title": "ReAct: Synergizing Reasoning and Acting in Language Models", "authors": "Shunyu Yao; Jeffrey Zhao; Yuan Cao", "venue": "ICLR", "year": 2023, "topics": "Reasoning; Agents; Large Language Models"},
    {"arxiv_id": "2305.10601", "title": "Tree of Thoughts: Deliberate Problem Solving with Large Language Models", "authors": "Shunyu Yao; Dian Yu; Karthik Narasimhan", "venue": "NeurIPS", "year": 2023, "topics": "Reasoning; Prompting; Large Language Models"},
    {"arxiv_id": "2302.04761", "title": "Toolformer: Language Models Can Teach Themselves to Use Tools", "authors": "Timo Schick; Jane Dwivedi-Yu; Thomas Scialom", "venue": "NeurIPS", "year": 2023, "topics": "Agents; Tool Use; Large Language Models"},
    {"arxiv_id": "2211.10435", "title": "PAL: Program-aided Language Models", "authors": "Luyu Gao; Aman Madaan; Graham Neubig", "venue": "ICML", "year": 2023, "topics": "Reasoning; Prompting; Large Language Models"},
    # --- Retrieval / RAG / dense IR ---
    {"arxiv_id": "2002.08909", "title": "REALM: Retrieval-Augmented Language Model Pre-Training", "authors": "Kelvin Guu; Kenton Lee; Ming-Wei Chang", "venue": "ICML", "year": 2020, "topics": "RAG; Information Retrieval; Pre-training"},
    {"arxiv_id": "2007.00808", "title": "Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval", "authors": "Lee Xiong; Chenyan Xiong; Arnold Overwijk", "venue": "ICLR", "year": 2021, "topics": "Dense Retrieval; Information Retrieval; NLP"},
    {"arxiv_id": "2112.09118", "title": "Unsupervised Dense Information Retrieval with Contrastive Learning", "authors": "Gautier Izacard; Mathilde Caron; Edouard Grave", "venue": "TMLR", "year": 2022, "topics": "Dense Retrieval; Information Retrieval; NLP"},
    {"arxiv_id": "2104.08663", "title": "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models", "authors": "Nandan Thakur; Nils Reimers; Iryna Gurevych", "venue": "NeurIPS", "year": 2021, "topics": "Information Retrieval; Benchmark; Evaluation"},
    {"arxiv_id": "2112.01488", "title": "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction", "authors": "Keshav Santhanam; Omar Khattab; Matei Zaharia", "venue": "NAACL", "year": 2022, "topics": "Dense Retrieval; Information Retrieval; Efficiency"},
    {"arxiv_id": "2212.03533", "title": "Text Embeddings by Weakly-Supervised Contrastive Pre-training", "authors": "Liang Wang; Nan Yang; Furu Wei", "venue": "arXiv", "year": 2022, "topics": "Sentence Embeddings; Dense Retrieval; NLP"},
    {"arxiv_id": "2007.01282", "title": "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering", "authors": "Gautier Izacard; Edouard Grave", "venue": "EACL", "year": 2021, "topics": "RAG; Question Answering; Information Retrieval"},
    {"arxiv_id": "2208.03299", "title": "Atlas: Few-shot Learning with Retrieval Augmented Language Models", "authors": "Gautier Izacard; Patrick Lewis; Edouard Grave", "venue": "JMLR", "year": 2023, "topics": "RAG; Few-Shot Learning; Information Retrieval"},
    {"arxiv_id": "2212.10496", "title": "Precise Zero-Shot Dense Retrieval without Relevance Labels", "authors": "Luyu Gao; Xueguang Ma; Jamie Callan", "venue": "ACL", "year": 2023, "topics": "Dense Retrieval; RAG; Information Retrieval"},
    {"arxiv_id": "2310.11511", "title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection", "authors": "Akari Asai; Zeqiu Wu; Hannaneh Hajishirzi", "venue": "ICLR", "year": 2024, "topics": "RAG; Information Retrieval; Generation"},
    {"arxiv_id": "2301.12652", "title": "REPLUG: Retrieval-Augmented Black-Box Language Models", "authors": "Weijia Shi; Sewon Min; Wen-tau Yih", "venue": "NAACL", "year": 2024, "topics": "RAG; Information Retrieval; Large Language Models"},
    {"arxiv_id": "2004.07180", "title": "Sparse, Dense, and Attentional Representations for Text Retrieval", "authors": "Yi Luan; Jacob Eisenstein; Michael Collins", "venue": "TACL", "year": 2021, "topics": "Information Retrieval; Dense Retrieval; NLP"},
    {"arxiv_id": "2010.00768", "title": "Distilling Knowledge from Reader to Retriever for Question Answering", "authors": "Gautier Izacard; Edouard Grave", "venue": "ICLR", "year": 2021, "topics": "Information Retrieval; Question Answering; Distillation"},
    # --- Embeddings / representation ---
    {"arxiv_id": "1310.4546", "title": "Distributed Representations of Words and Phrases and their Compositionality", "authors": "Tomas Mikolov; Ilya Sutskever; Jeffrey Dean", "venue": "NeurIPS", "year": 2013, "topics": "Word Embeddings; Representation Learning; NLP"},
    {"arxiv_id": "1405.4053", "title": "Distributed Representations of Sentences and Documents", "authors": "Quoc V. Le; Tomas Mikolov", "venue": "ICML", "year": 2014, "topics": "Representation Learning; Word Embeddings; NLP"},
    {"arxiv_id": "1607.04606", "title": "Enriching Word Vectors with Subword Information", "authors": "Piotr Bojanowski; Edouard Grave; Tomas Mikolov", "venue": "TACL", "year": 2017, "topics": "Word Embeddings; Representation Learning; NLP"},
    {"arxiv_id": "1908.10084", "title": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", "authors": "Nils Reimers; Iryna Gurevych", "venue": "EMNLP", "year": 2019, "topics": "Sentence Embeddings; Dense Retrieval; BERT"},
    # --- Benchmarks / evaluation / QA datasets ---
    {"arxiv_id": "1606.05250", "title": "SQuAD: 100,000+ Questions for Machine Comprehension of Text", "authors": "Pranav Rajpurkar; Jian Zhang; Percy Liang", "venue": "EMNLP", "year": 2016, "topics": "Question Answering; Benchmark; NLP"},
    {"arxiv_id": "1905.00537", "title": "SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems", "authors": "Alex Wang; Yada Pruksachatkun; Samuel R. Bowman", "venue": "NeurIPS", "year": 2019, "topics": "Benchmark; Evaluation; NLP"},
    {"arxiv_id": "2009.03300", "title": "Measuring Massive Multitask Language Understanding", "authors": "Dan Hendrycks; Collin Burns; Jacob Steinhardt", "venue": "ICLR", "year": 2021, "topics": "Benchmark; Evaluation; Large Language Models"},
    {"arxiv_id": "1809.09600", "title": "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering", "authors": "Zhilin Yang; Peng Qi; Christopher D. Manning", "venue": "EMNLP", "year": 2018, "topics": "Question Answering; Benchmark; NLP"},
    {"arxiv_id": "1705.03551", "title": "TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension", "authors": "Mandar Joshi; Eunsol Choi; Luke Zettlemoyer", "venue": "ACL", "year": 2017, "topics": "Question Answering; Benchmark; NLP"},
    {"arxiv_id": "2110.14168", "title": "Training Verifiers to Solve Math Word Problems", "authors": "Karl Cobbe; Vineet Kosaraju; John Schulman", "venue": "arXiv", "year": 2021, "topics": "Reasoning; Benchmark; Large Language Models"},
    {"arxiv_id": "2206.04615", "title": "Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models", "authors": "Aarohi Srivastava; Abhinav Rastogi; BIG-bench collaboration", "venue": "TMLR", "year": 2023, "topics": "Benchmark; Evaluation; Large Language Models"},
    {"arxiv_id": "1611.09268", "title": "MS MARCO: A Human Generated MAchine Reading COmprehension Dataset", "authors": "Tri Nguyen; Mir Rosenberg; Li Deng", "venue": "NeurIPS Workshop", "year": 2016, "topics": "Question Answering; Information Retrieval; Benchmark"},
    # --- Generation / decoding / hallucination ---
    {"arxiv_id": "1904.09751", "title": "The Curious Case of Neural Text Degeneration", "authors": "Ari Holtzman; Jan Buys; Yejin Choi", "venue": "ICLR", "year": 2020, "topics": "Generation; Decoding; NLP"},
    {"arxiv_id": "2202.03629", "title": "Survey of Hallucination in Natural Language Generation", "authors": "Ziwei Ji; Nayeon Lee; Pascale Fung", "venue": "ACM Computing Surveys", "year": 2023, "topics": "Generation; Hallucination; Survey"},
    {"arxiv_id": "2303.08774", "title": "GPT-4 Technical Report", "authors": "OpenAI", "venue": "arXiv", "year": 2023, "topics": "Large Language Models; Evaluation; NLP"},
    # --- Vision-language (related, cs.CV/cs.CL) ---
    {"arxiv_id": "2010.11929", "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", "authors": "Alexey Dosovitskiy; Lucas Beyer; Neil Houlsby", "venue": "ICLR", "year": 2021, "topics": "Transformers; Vision-Language; Representation Learning"},
    {"arxiv_id": "2103.00020", "title": "Learning Transferable Visual Models From Natural Language Supervision", "authors": "Alec Radford; Jong Wook Kim; Ilya Sutskever", "venue": "ICML", "year": 2021, "topics": "Vision-Language; Representation Learning; NLP"},
    {"arxiv_id": "2201.12086", "title": "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation", "authors": "Junnan Li; Dongxu Li; Steven Hoi", "venue": "ICML", "year": 2022, "topics": "Vision-Language; Pre-training; Generation"},
    # --- Misc influential ---
    {"arxiv_id": "1607.06450", "title": "Layer Normalization", "authors": "Jimmy Lei Ba; Jamie Ryan Kiros; Geoffrey E. Hinton", "venue": "arXiv", "year": 2016, "topics": "Optimization; Transformers; Representation Learning"},
    {"arxiv_id": "1512.03385", "title": "Deep Residual Learning for Image Recognition", "authors": "Kaiming He; Xiangyu Zhang; Jian Sun", "venue": "CVPR", "year": 2016, "topics": "Representation Learning; Vision-Language; Optimization"},
    {"arxiv_id": "1412.6980", "title": "Adam: A Method for Stochastic Optimization", "authors": "Diederik P. Kingma; Jimmy Ba", "venue": "ICLR", "year": 2015, "topics": "Optimization; Representation Learning"},
    {"arxiv_id": "1503.02531", "title": "Distilling the Knowledge in a Neural Network", "authors": "Geoffrey Hinton; Oriol Vinyals; Jeff Dean", "venue": "NeurIPS Workshop", "year": 2015, "topics": "Distillation; Efficiency; Representation Learning"},
    {"arxiv_id": "2312.10997", "title": "Retrieval-Augmented Generation for Large Language Models: A Survey", "authors": "Yunfan Gao; Yun Xiong; Haofen Wang", "venue": "arXiv", "year": 2023, "topics": "RAG; Survey; Large Language Models"},
    {"arxiv_id": "2303.18223", "title": "A Survey of Large Language Models", "authors": "Wayne Xin Zhao; Kun Zhou; Ji-Rong Wen", "venue": "arXiv", "year": 2023, "topics": "Large Language Models; Survey; NLP"},
    {"arxiv_id": "2108.07258", "title": "On the Opportunities and Risks of Foundation Models", "authors": "Rishi Bommasani; Drew A. Hudson; Percy Liang", "venue": "arXiv", "year": 2021, "topics": "Large Language Models; Survey; Alignment"},
    {"arxiv_id": "1910.03771", "title": "Transformers: State-of-the-Art Natural Language Processing", "authors": "Thomas Wolf; Lysandre Debut; Alexander M. Rush", "venue": "EMNLP", "year": 2020, "topics": "Transformers; NLP; Software"},
    {"arxiv_id": "2005.11401", "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "authors": "Patrick Lewis; Ethan Perez; Douwe Kiela", "venue": "NeurIPS", "year": 2020, "topics": "RAG; Information Retrieval; Generation"},
]

ARXIV_PDF = "https://arxiv.org/pdf/{arxiv_id}.pdf"
ARXIV_API = "http://export.arxiv.org/api/query"
USER_AGENT = "Mozilla/5.0 (CSAI415-corpus-downloader; academic use)"

ANCHOR_IDS = {p["arxiv_id"] for p in CORPUS}

# arXiv category codes -> readable Topic names for the Neo4j graph.
CATEGORY_TOPICS = {
    "cs.CL": "Computation and Language",
    "cs.IR": "Information Retrieval",
    "cs.LG": "Machine Learning",
    "cs.AI": "Artificial Intelligence",
    "cs.CV": "Computer Vision",
    "cs.NE": "Neural Computing",
    "stat.ML": "Statistical Machine Learning",
}
_ATOM = {"a": "http://www.w3.org/2005/Atom"}


def _slugify(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return s[:60] or "paper"


def harvest_arxiv(target_total: int, categories: list[str], page: int = 180) -> list[dict]:
    """Query the arXiv API for recent cs.CL/cs.IR papers to grow the corpus.

    Returns metadata dicts in the same shape as CORPUS (paper_id assigned by the
    caller).  The 14 hand-curated anchor papers are excluded here and prepended
    by the caller so the gold Q/A references stay valid.
    """
    need = max(0, target_total - len(CORPUS))
    if need <= 0:
        return []
    query = " OR ".join(f"cat:{c}" for c in categories)
    harvested: list[dict] = []
    seen: set[str] = set(ANCHOR_IDS)
    start = 0
    print(f"[harvest] querying arXiv for {need} more papers ({query}) ...")
    while len(harvested) < need and start < need * 3 + 200:
        params = urllib.parse.urlencode({
            "search_query": query, "start": start, "max_results": page,
            "sortBy": "submittedDate", "sortOrder": "descending",
        })
        url = f"{ARXIV_API}?{params}"
        xml = None
        for attempt in range(1, 6):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    xml = resp.read()
                break
            except Exception as exc:  # noqa: BLE001 (incl. HTTP 429 rate-limit)
                wait = 30 * attempt   # arXiv API blocks bursts for minutes; back off hard
                print(f"[harvest] API attempt {attempt}/5 failed ({exc}); retry in {wait}s")
                time.sleep(wait)
        if xml is None:
            print("[harvest] giving up on this page.")
            break

        entries = ET.fromstring(xml).findall("a:entry", _ATOM)
        if not entries:
            break
        for e in entries:
            raw_id = (e.findtext("a:id", default="", namespaces=_ATOM) or "").rsplit("/", 1)[-1]
            arxiv_id = re.sub(r"v\d+$", "", raw_id)        # strip version suffix
            if not arxiv_id or arxiv_id in seen:
                continue
            seen.add(arxiv_id)
            title = " ".join((e.findtext("a:title", default="", namespaces=_ATOM) or "").split())
            authors = [a.findtext("a:name", default="", namespaces=_ATOM)
                       for a in e.findall("a:author", _ATOM)]
            authors = [a for a in authors if a][:8]
            published = e.findtext("a:published", default="", namespaces=_ATOM) or ""
            year = int(published[:4]) if published[:4].isdigit() else None
            cats = [c.get("term") for c in e.findall("a:category", _ATOM) if c.get("term")]
            topics = [CATEGORY_TOPICS.get(c) for c in cats if c in CATEGORY_TOPICS]
            topics = list(dict.fromkeys(t for t in topics if t)) or ["Computation and Language"]
            harvested.append({
                "arxiv_id": arxiv_id,
                "slug": _slugify(title),
                "title": title,
                "authors": "; ".join(authors),
                "venue": "arXiv",
                "year": year,
                "topics": "; ".join(topics),
            })
            if len(harvested) >= need:
                break
        start += page
        time.sleep(3.0)  # arXiv API politeness window
    print(f"[harvest] collected {len(harvested)} papers from arXiv.")
    return harvested


def download_one(arxiv_id: str, dest: Path, *, retries: int = 3, timeout: int = 60) -> bool:
    """Download a single arXiv PDF with retries.  Returns True on success."""
    url = ARXIV_PDF.format(arxiv_id=arxiv_id)
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if not data.startswith(b"%PDF"):
                raise ValueError("response is not a PDF (got HTML — rate limited?)")
            dest.write_bytes(data)
            return True
        except Exception as exc:  # noqa: BLE001
            wait = 2 * attempt
            print(f"    attempt {attempt}/{retries} failed: {exc}  (retry in {wait}s)")
            time.sleep(wait)
    return False


def write_metadata(out_dir: Path, corpus: list[dict]) -> None:
    """Write papers.csv (for build_graph.py) and corpus_metadata.json."""
    data_dir = out_dir.parent
    csv_path = data_dir / "papers.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["paper id", "title", "authors", "venue", "year", "topics"])
        for p in corpus:
            writer.writerow([p["paper_id"], p["title"], p["authors"],
                             p["venue"], p["year"], p["topics"]])
    print(f"[meta] wrote {csv_path}")

    meta = []
    for p in corpus:
        meta.append({
            **p,
            "doi": f"10.48550/arXiv.{p['arxiv_id']}",
            "url": f"https://arxiv.org/abs/{p['arxiv_id']}",
            "pdf_url": ARXIV_PDF.format(arxiv_id=p["arxiv_id"]),
            "pdf_filename": f"{p['slug']}.pdf",
            "license": "arXiv.org perpetual non-exclusive license (open access)",
        })
    json_path = data_dir / "corpus_metadata.json"
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[meta] wrote {json_path}")


_TITLE_STOP = {"a", "an", "and", "the", "of", "for", "with", "in", "on", "to",
               "via", "from", "is", "are", "as", "by", "at", "using", "than",
               "your", "you", "we", "our", "be", "into", "over"}


def verify_title(pdf_path: Path, title: str) -> bool:
    """Best-effort check that a downloaded PDF really is the claimed paper.

    Reads the first two pages and confirms that enough significant title words
    appear there (arXiv prints the title at the top of page 1).  Guarantees
    papers.csv metadata matches the actual files even if a curated id is wrong.
    If PyMuPDF is unavailable we skip the check (return True).
    """
    try:
        import fitz
    except Exception:  # noqa: BLE001
        return True
    try:
        doc = fitz.open(str(pdf_path))
        text = " ".join((doc[i].get_text("text") or "")
                        for i in range(min(2, len(doc)))).lower()
        doc.close()
    except Exception:  # noqa: BLE001
        return False
    words = [re.sub(r"[^a-z0-9]", "", w) for w in title.lower().split()]
    content = [w for w in words if len(w) > 3 and w not in _TITLE_STOP]
    if not content:
        return True
    hits = sum(1 for w in content if w in text)
    required = max(2, (len(content) + 1) // 2)   # >= half the content words
    return hits >= min(required, len(content))


def main() -> int:
    ap = argparse.ArgumentParser(description="Download the real arXiv PDF corpus.")
    ap.add_argument("--out", type=Path, default=Path("data/pdfs"),
                    help="Directory to write PDFs into.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only download the first N papers (quick demo).")
    ap.add_argument("--anchors-only", action="store_true",
                    help="Download only the 14 hand-curated anchor papers.")
    ap.add_argument("--harvest", type=int, default=None, metavar="TOTAL",
                    help="Also top up to TOTAL papers via the arXiv API "
                         "(usually rate-limited; the curated set is API-free).")
    ap.add_argument("--categories", nargs="+", default=["cs.CL", "cs.IR"],
                    help="arXiv categories to harvest from.")
    ap.add_argument("--no-verify", action="store_true",
                    help="Skip the title-verification step.")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if the PDF already exists.")
    args = ap.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the working corpus: curated anchors first, then the curated extended
    # set (API-free), de-duplicated by arXiv id; optional API harvest on top.
    corpus = [dict(p) for p in CORPUS]
    seen = {p["arxiv_id"] for p in corpus}
    if not args.anchors_only:
        for e in EXTRA_CORPUS:
            if e["arxiv_id"] in seen:
                continue
            seen.add(e["arxiv_id"])
            corpus.append(dict(e))
    if args.harvest:
        for e in harvest_arxiv(args.harvest, args.categories):
            if e["arxiv_id"] in seen:
                continue
            seen.add(e["arxiv_id"])
            corpus.append(dict(e))
    if args.limit:
        corpus = corpus[: args.limit]

    # Normalise: ensure a slug; paper_id is assigned later over the KEPT set.
    for p in corpus:
        p.setdefault("slug", _slugify(p["title"]))
    print(f"[corpus] {len(corpus)} candidate papers → {out_dir.resolve()}\n")

    kept: list[dict] = []
    skipped, failed, mismatched = 0, [], []
    for i, p in enumerate(corpus, start=1):
        dest = out_dir / f"{p['slug']}.pdf"
        label = f"[{i:>3}/{len(corpus)}] {p['arxiv_id']:<11} {p['title'][:50]}"
        if dest.exists() and not args.force:
            print(f"{label}  (skip — exists)")
            kept.append(p)
            skipped += 1
            continue
        print(f"{label}")
        if not download_one(p["arxiv_id"], dest):
            failed.append(p["arxiv_id"])
            continue
        if not args.no_verify and not verify_title(dest, p["title"]):
            print(f"    x title mismatch — dropping {p['arxiv_id']}")
            dest.unlink(missing_ok=True)
            mismatched.append(p["arxiv_id"])
            continue
        print(f"    ok {dest.name}  ({dest.stat().st_size/1024:.0f} KB)")
        kept.append(p)
        time.sleep(1.0)  # be polite to arXiv

    # Assign contiguous paper ids over the kept set (anchors stay P001..P014).
    for idx, p in enumerate(kept, start=1):
        p["paper_id"] = f"P{idx:03d}"

    write_metadata(out_dir, kept)

    print("\n" + "=" * 60)
    print(f"  kept (in corpus): {len(kept)}")
    print(f"  skipped (exists): {skipped}")
    print(f"  download failed : {len(failed)}  {failed if failed else ''}")
    print(f"  title mismatch  : {len(mismatched)}  {mismatched if mismatched else ''}")
    print(f"  corpus dir      : {out_dir.resolve()}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
