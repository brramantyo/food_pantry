# Generating Failure Examples for Report

## Quick Start (Cluster)

### Step 1: Find Your Evaluation Results

First, check if you have evaluation results with per-sample predictions:

```bash
# Look for eval results
ls -lh eval_results*.json

# Check if it has per_sample_results
python3 -c "import json; d=json.load(open('eval_results_v11.json')); print('per_sample_results' in d)"
```

### Step 2: Edit the Script

Open `run_generate_failures.sh` and update the `EVAL_RESULTS` variable:

```bash
EVAL_RESULTS="eval_results_v11.json"  # Change to your actual file
```

### Step 3: Run on Cluster

```bash
# Create logs directory
mkdir -p logs

# Submit job
sbatch run_generate_failures.sh

# Check status
squeue -u $USER

# View output
tail -f logs/generate_failures_*.log
```

### Step 4: Download Results

```bash
# On your local machine:
scp -r hzb0071@aiau.eng.auburn.edu:~/food_pantry_repo/figures/failure_examples ./
```

---

## What You'll Get

After running, you'll have:

1. **5 annotated images** (`failure_example_1.png` to `failure_example_5.png`)
   - Shows ground truth vs prediction
   - Highlights missed/wrong categories
   
2. **LaTeX code** (`latex_failure_examples.tex`)
   - Ready to copy-paste into your report
   - Includes figure captions and descriptions

3. **JSON metadata** (`failure_descriptions.json`)
   - Structured data about each failure case

---

## Using in Your Report

### Option A: Copy the Entire Section

Open `figures/failure_examples/latex_failure_examples.tex` and copy everything into your report's Task 1 Error Analysis section.

### Option B: Pick Individual Examples

If you only want 1-2 examples instead of all 5:

```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.7\textwidth]{figures/failure_examples/failure_example_1.png}
    \caption{\textbf{Multi-Item Omission.} Ground truth: \texttt{Granola} (2), \texttt{Dairy} (1). Prediction: \texttt{Granola} (2). The model predicted only \texttt{Granola} and missed \texttt{Dairy}.}
    \label{fig:failure_example_1}
\end{figure}
```

---

## Troubleshooting

### "FileNotFoundError: eval_results_v11.json"

Your evaluation results file has a different name. Find it:

```bash
find . -name "eval_*.json" -o -name "*results*.json"
```

Then update `EVAL_RESULTS` in the script.

### "KeyError: per_sample_results"

Your evaluation results don't have per-sample predictions. You need to:

**Option 1:** Re-run evaluation with `--save-predictions`:

```bash
python3 evaluate_florence2.py \
    --checkpoint checkpoints/v11_best \
    --base-model microsoft/Florence-2-large-ft \
    --test-jsonl florence2_data/test.jsonl \
    --output eval_results_v11_with_predictions.json \
    --bf16
```

**Option 2:** Run inference directly (slower):

Edit `run_generate_failures.sh` and remove the `if [ -f "$EVAL_RESULTS" ]` check, so it always runs inference.

### "No module named 'matplotlib'"

Install matplotlib:

```bash
pip install matplotlib pillow --user
```

Or if using conda:

```bash
conda install matplotlib pillow
```

---

## Alternative: Run Locally (Without Cluster)

If you have the test data locally:

```bash
python3 generate_failure_examples_cluster.py \
    --predictions eval_results_v11.json \
    --test-jsonl florence2_data/test.jsonl \
    --data-dir . \
    --output-dir figures/failure_examples \
    --n-examples 5
```

---

## Customization

### Change Number of Examples

```bash
--n-examples 3  # Generate only 3 examples instead of 5
```

### Different Model/Checkpoint

```bash
--checkpoint checkpoints/contrastive_best \
--base-model microsoft/Florence-2-large-ft
```

### Different Failure Types

The script automatically selects diverse failure types:
1. Multi-item omission (most common)
2. Category confusion
3. Count errors
4. False positives

If you want to manually select specific examples, edit the `select_representative_examples()` function in the script.

---

## Expected Output

```
==========================================
FAILURE EXAMPLE GENERATOR
==========================================

Loading test data...
✓ Loaded 166 test samples

Loading predictions...
✓ Loaded 166 predictions

Categorizing failures...

Failure statistics:
  • multi_item_omission: 23 cases
  • category_confusion: 8 cases
  • count_error: 5 cases
  • false_positive: 3 cases

Selecting 5 representative examples...

Generating visualizations...
✓ Saved: figures/failure_examples/failure_example_1.png
✓ Saved: figures/failure_examples/failure_example_2.png
✓ Saved: figures/failure_examples/failure_example_3.png
✓ Saved: figures/failure_examples/failure_example_4.png
✓ Saved: figures/failure_examples/failure_example_5.png

==========================================
✓ COMPLETE
==========================================

Generated 5 failure examples
Output directory: figures/failure_examples

Files created:
  • 5 PNG images
  • figures/failure_examples/failure_descriptions.json
  • figures/failure_examples/latex_failure_examples.tex

Next steps:
  1. Review the images in figures/failure_examples
  2. Copy the LaTeX code from latex_failure_examples.tex into your report
  3. Adjust the figure placement and captions as needed
```

---

## Questions?

If you run into issues, check:
1. Do you have test.jsonl in `florence2_data/`?
2. Do you have evaluation results with predictions?
3. Are the image paths in test.jsonl correct?
4. Is matplotlib installed?

For more help, check the script's `--help`:

```bash
python3 generate_failure_examples_cluster.py --help
```
