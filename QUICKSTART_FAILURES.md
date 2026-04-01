# 🎯 Quick Start: Generate 5 Failure Examples

## TL;DR

```bash
# 1. Check if you have everything
bash check_requirements.sh

# 2. Edit the script (set your eval results file)
nano run_generate_failures.sh

# 3. Run on cluster
sbatch run_generate_failures.sh

# 4. Download results
scp -r hzb0071@aiau:~/food_pantry_repo/figures/failure_examples ./

# 5. Copy LaTeX code into your report
cat figures/failure_examples/latex_failure_examples.tex
```

---

## What You'll Get

✅ **5 annotated images** showing ground truth vs prediction  
✅ **LaTeX code** ready to paste into your report  
✅ **JSON metadata** with failure descriptions  

---

## Files Created

| File | Description |
|------|-------------|
| `generate_failure_examples_cluster.py` | Main script (runs on cluster) |
| `run_generate_failures.sh` | SLURM submission script |
| `check_requirements.sh` | Check if you have required files |
| `FAILURE_EXAMPLES_README.md` | Full documentation |
| `EXAMPLE_OUTPUT.md` | Shows what the output looks like |

---

## Two Ways to Run

### Option 1: Use Existing Eval Results (FAST ⚡)

If you have `eval_results_v11.json` with per-sample predictions:

```bash
python3 generate_failure_examples_cluster.py \
    --predictions eval_results_v11.json \
    --test-jsonl florence2_data/test.jsonl \
    --data-dir . \
    --output-dir figures/failure_examples
```

### Option 2: Run Inference (SLOWER 🐢)

If you don't have eval results:

```bash
python3 generate_failure_examples_cluster.py \
    --checkpoint checkpoints/v11_best \
    --base-model microsoft/Florence-2-large-ft \
    --test-jsonl florence2_data/test.jsonl \
    --data-dir . \
    --output-dir figures/failure_examples \
    --run-inference
```

---

## Output Structure

```
figures/failure_examples/
├── failure_example_1.png          # Multi-item omission
├── failure_example_2.png          # Multi-item omission
├── failure_example_3.png          # Category confusion
├── failure_example_4.png          # Count error
├── failure_example_5.png          # Multi-item omission
├── failure_descriptions.json      # Metadata
└── latex_failure_examples.tex     # Ready-to-use LaTeX code
```

---

## Using in Your Report

### Full Version (All 5 Examples)

```latex
\subsubsection{Example Failure Cases}

[Copy entire content from latex_failure_examples.tex]
```

### Minimal Version (1-2 Examples)

```latex
\subsubsection{Example Failure Cases}

A few typical failure cases help explain the remaining errors:

\begin{figure}[H]
    \centering
    \includegraphics[width=0.7\textwidth]{figures/failure_examples/failure_example_1.png}
    \caption{Multi-item omission: model predicted \texttt{Granola} but missed \texttt{Dairy}.}
\end{figure}

This is consistent with the main error pattern: omission in multi-item scenes.
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: test.jsonl` | Check path: `florence2_data/test.jsonl` |
| `KeyError: per_sample_results` | Use `--run-inference` instead |
| `No module named 'matplotlib'` | `pip install matplotlib pillow` |
| Images not found | Check `--data-dir` path |

---

## Need Help?

📖 Read `FAILURE_EXAMPLES_README.md` for full documentation  
👀 Check `EXAMPLE_OUTPUT.md` to see what you'll get  
🔍 Run `check_requirements.sh` to diagnose issues  

---

## Next Steps

1. ✅ Generate the 5 examples
2. ✅ Review the images
3. ✅ Copy LaTeX code into report
4. ✅ Adjust captions if needed
5. ✅ Compile report and check figure placement

**This addresses Prof Sun's feedback about "insufficient reasoning" and "failure examples"!** 🎉
