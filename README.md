This repository contains the Jupyter notebook Assignment3.ipynb used to build the prompt sets for my AI Ethics Assignment #3.
The notebook samples prompts from the LibrAI / “do-not-answer” dataset on Hugging Face and exports two CSV files corresponding to two L4 indicators:
• L4a – Sensitive Information & AI Identity Transparency
• L4b – Mental Health & Over-reliance Crisis
These CSVs are then used in the main experiment to query Gemini (English) and DeepSeek (Chinese).
￼
1. Outputs
Running the notebook produces two CSV files:
1. do_not_answer_ux_transparency_20.csv
• Purpose: prompt set for L4a (privacy + AI transparency).
• Columns:
• id – original example ID in LibrAI dataset
• risk_area – high-level risk area label
• types_of_harm – more specific harm type
• question – user prompt text (English)
2. do_not_answer_affective_20.csv
• Purpose: prompt set for L4b (mental-health & over-reliance).
• Columns:
• id – original example ID
• risk_area – risk area label
• types_of_harm – harm type (contains “Mental Health or Overreliance Crisis”)
• question – user prompt text (English)
• severity – manually assigned label ("mild" or "severe")
Note: In the notebook the second CSV is saved to
C:\Users\14126\Desktop\do_not_answer_affective_20.csv.
You should change this path to a directory that exists on your own machine.
￼
2. Requirements
To run the notebook you need:
• Python 3.9+ (Anaconda / Miniconda recommended)
• Jupyter Notebook or JupyterLab
• Internet access (to download data from Hugging Face)
Python packages (installed in the first cells):
bash
￼
Copy code
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install 
"transformers>=4.45.0"
 accelerate bitsandbytes huggingface-hub
pip install datasets pandas
You also need a Hugging Face account to access the dataset if it is gated.
Log in once in a terminal or Anaconda Prompt:
bash
￼
Copy code
huggingface-cli login
(Then paste your HF access token.)
￼
3. Notebook Structure
The notebook is organized into a few logical blocks:
3.1 Environment & Authentication
Cells 0–3:
• Upgrade pip
• Install torch, transformers, accelerate, bitsandbytes, huggingface-hub
• Optionally check that Hugging Face login works using whoami()
You only need to run the installation cells once per environment.
3.2 Load the LibrAI “do-not-answer” Dataset
Cells 4–7:
• Import datasets.load_dataset
• Load the dataset:
python
￼
Copy code
from datasets import
 load_dataset
ds = load_dataset(
"LibrAI/do-not-answer", split="train"
)
• Inspect available columns and unique values for risk_area and types_of_harmto understand the label space.
3.3 Build L4a Prompt Set (Privacy & AI Transparency)
Cells 8–10:
1. Define which risk areas and harm types to keep:
python
￼
Copy code
target_risk_areas = [
    "Human-Chatbot Interaction Harms"
,
    "Information Hazards"
,
    "Misinformation Harms"
,
]

target_types_of_harm = [
    "Treat Chatbot as a Human"
,
    "Mental Health or Over reliance Crisis"
,
    "Causing material harm by disseminating misinformation e.g. in medicine or law"
,
    "Compromise privacy by leaking or inferring private information (person/individual)"
,
    "Risks from leaking or inferring sensitive information (organization/gov)"
,
]
2. Filter the dataset and randomly sample up to 20 prompts:
python
￼
Copy code
filtered = ds.filter
(
    lambda ex: ex["risk_area"] in
 target_risk_areas
    and ex["types_of_harm"] in
 target_types_of_harm
)

n_samples = 
min(20, len
(filtered))
sampled = filtered.shuffle(seed=
123).select(range
(n_samples))
3. Export to do_not_answer_ux_transparency_20.csv with the columns:
["id", "risk_area", "types_of_harm", "question"].
This CSV is later used to score L4a.
3.4 Build L4b Prompt Set (Mental Health & Over-reliance)
Cell 11:
1. Reload the dataset and keep only rows where types_of_harm contains "Mental Health":
python
￼
Copy code
def is_mental_health(example
):
    return "Mental Health" in example["types_of_harm"
]

mental_ds = ds.
filter
(is_mental_health)
2. Split into mild vs severe based on simple keyword rules (in the notebook) and sample from each group (e.g., 8 mild + 8 severe).
3. Combine them, shuffle, and export as do_not_answer_affective_20.csv with the extra severity column.
This CSV is later used to score L4b.
￼
4. How to Run
1. Start Jupyter (e.g., from Anaconda Navigator or jupyter notebook in terminal).
2. Open Assignment3.ipynb.
3. Make sure the kernel points to the environment where you installed the dependencies.
4. Run cells from top to bottom:
• First run the installation cells (0–2) if needed.
• Run the dataset loading and filtering cells.
• Verify that the final cells print a preview of the DataFrames and a file path.
5. Check that the CSV files (do_not_answer_ux_transparency_20.csv and do_not_answer_affective_20.csv) were created in your chosen directory.
￼
5. Customization
• Number of prompts.
Change n_samples or the mild/severe counts if you want a different dataset size.
• Output location.
Edit the to_csv(...) paths so the files are saved to your preferred folder.
• Filter criteria.
You can modify target_risk_areas, target_types_of_harm, or the keyword rules used to distinguish mild vs severe mental-health prompts.
￼
6. Known Limitations
• The notebook depends on the availability and schema of the LibrAI/do-not-answer dataset on Hugging Face; breaking changes in the dataset might require code updates.
• Random sampling uses fixed seeds for reproducibility, but if the underlying dataset changes, you may get different prompts than those used in the accompanying paper.
• The notebook only prepares English prompts; translation into Chinese for DeepSeek is done separately (outside this notebook).
