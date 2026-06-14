# 🔬 Information Retrieval System (Assignment 1)

**Designed by IR Group 81**
* AJAY RAI (2025ab05194@wilp.bits-pilani.ac.in)
* PRATIK ROY (2025ab05187@wilp.bits-pilani.ac.in)
* VRITI MALHOTRA (2025ab05195@wilp.bits-pilani.ac.in)

---

## 📖 Project Overview
This project is an interactive, end-to-end Information Retrieval (IR) system built using Python and Streamlit. It allows users to upload a collection of text documents and dynamically explores core IR concepts including text normalization, index construction, phrase querying, dictionary search benchmarking, and tolerant retrieval.

## ✨ Core Features
* **Text Preprocessing:** Compares Stemming (Porter Stemmer) vs. Lemmatization (WordNet) and dynamically calculates vocabulary reduction percentages.
* **Phrase Query Processing:** Implements and confronts Biword Indices against Positional Indices to demonstrate how positional tracking eliminates structural false positives.
* **Dictionary Search Benchmarking:** Automatically builds a Binary Search Tree (BST) and a B-Tree (t=3) from the corpus vocabulary, running live multi-query performance benchmarks to compare Search and Retrieval speeds.
* **Tolerant Retrieval:** Features a dynamic spell-checker utilizing the Levenshtein Edit Distance algorithm to correct user typos against the indexed vocabulary.
* **Dynamic Inference Reporting:** Automatically generates data-backed inferences based on the specific document dataset uploaded by the user.

---

## 🚀 How to Run the Application

This application is deployed on the web and requires **no installation**. 

1. **Open the App:** Click this link to open the application in any modern web browser (Chrome, Safari, Edge, etc.): 
   👉 **https://ir-assignment-81.streamlit.app/**
2. **Upload Data:** On the main screen, click the "Browse files" button and upload your dataset consisting of standard `.txt` files. You can upload multiple files at once.
3. **Reset:** If you want to test a different dataset, click the "🗑️ Reset & Clear All Datasets" button in the left sidebar to wipe the system memory before uploading new files.

---

## 📊 How to Interpret the Results

Once your text files are uploaded, the system processes them and unlocks several analytical tabs. Here is how to understand the outputs in each section:

### 1. Preprocessing Analysis (Stemming vs. Lemmatization)
* **What you are looking at:** The system compares two methods of reducing words to their root forms. 
* **How to interpret it:** Look at the percentage reduction in the "Vocabulary Size" metrics. A higher percentage reduction means the system compressed the index more efficiently, which saves memory. The "Dynamic System Conclusion" box will automatically tell you which method is better for your specific uploaded files based on the balance between index compression and average document recall.

### 2. Phrase Query Processing
* **What you are looking at:** A head-to-head comparison between a Biword Index (storing pairs of words) and a Positional Index (tracking exact word coordinates).
* **How to interpret it:** Type a two-word phrase (e.g., "machine learning"). If the Biword Index returns *more* matched files than the Positional Index, those extra files are **false positives**. This proves that the Positional Index is more accurate because it ensures the words are actually next to each other, rather than just separated by a period in two different sentences.

### 3. Tree Index Performance
* **What you are looking at:** The system builds two data structures (Binary Search Tree and B-Tree) from your vocabulary and runs an automated speed race using 5 test words.
* **How to interpret it:** Look at the "Faster Total" column in the generated table. This shows which tree structure searched and retrieved the data faster (measured in fractions of a second). B-Trees generally win as your uploaded dataset gets larger due to their multi-way branching design.

### 4. Tolerant Retrieval
* **What you are looking at:** A spell-checking system using the Levenshtein Edit Distance mathematical algorithm.
* **How to interpret it:** Type a misspelled word from your text files (e.g., "computr" instead of "computer"). The "Computed Edit Distance Score" tells you exactly how many character insertions, deletions, or substitutions were required to fix your typo. The lower the score, the closer your typo was to the system's final suggestion. 

### 5. Compulsory Report (Section G)
* **What you are looking at:** A dynamically generated report answering core academic questions.
* **How to interpret it:** You don't have to guess! The system reads the mathematical metrics from the previous tabs and automatically writes a data-backed conclusion for each of the 7 compulsory assignment questions based strictly on the files you provided.
