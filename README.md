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

## ⚙️ Setup & Installation Instructions

To run this application locally or in a Virtual Lab environment, follow these steps:

### 1. Prerequisites
Ensure you have Python installed on your machine (Python 3.8+ recommended).

### 2. Install Dependencies
Clone or download this repository, navigate to the project folder in your terminal, and install the required external libraries:
```bash
pip install -r requirements.txt
