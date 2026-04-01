# Example Output: latex_failure_examples.tex

This is what the generated LaTeX code will look like. You can copy-paste this directly into your report.

---

```latex
\subsubsection{Example Failure Cases}

To better understand the remaining errors, we examine a few typical failure cases:

\begin{figure}[H]
    \centering
    \includegraphics[width=0.7\textwidth]{figures/failure_examples/failure_example_1.png}
    \caption{\textbf{Multi-Item Omission.} Ground truth: \texttt{Granola} (2), \texttt{Dairy} (1). Prediction: \texttt{Granola} (2). The model predicted only \texttt{Granola} and missed \texttt{Dairy}.}
    \label{fig:failure_example_1}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.7\textwidth]{figures/failure_examples/failure_example_2.png}
    \caption{\textbf{Multi-Item Omission.} Ground truth: \texttt{Ready Meals} (1), \texttt{Carbohydrate Meal} (2). Prediction: \texttt{Carbohydrate Meal} (2). The model predicted only \texttt{Carbohydrate Meal} and missed \texttt{Ready Meals}.}
    \label{fig:failure_example_2}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.7\textwidth]{figures/failure_examples/failure_example_3.png}
    \caption{\textbf{Category Confusion.} Ground truth: \texttt{Savory Snacks and Crackers} (3). Prediction: \texttt{Granola} (2), \texttt{Savory Snacks and Crackers} (1). The model incorrectly predicted \texttt{Granola} instead of \texttt{Savory Snacks and Crackers}.}
    \label{fig:failure_example_3}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.7\textwidth]{figures/failure_examples/failure_example_4.png}
    \caption{\textbf{Count Error.} Ground truth: \texttt{Canned Fruit} (4). Prediction: \texttt{Canned Fruit} (2). The model predicted the correct categories but wrong counts.}
    \label{fig:failure_example_4}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.7\textwidth]{figures/failure_examples/failure_example_5.png}
    \caption{\textbf{Multi-Item Omission.} Ground truth: \texttt{Vegetables Canned} (2), \texttt{Canned Fruit} (1). Prediction: \texttt{Canned Fruit} (1). The model predicted only \texttt{Canned Fruit} and missed \texttt{Vegetables Canned}.}
    \label{fig:failure_example_5}
\end{figure}

These examples are consistent with the main pattern in the confusion analysis: 
the dominant issue is \textbf{omission in multi-item scenes} rather than 
large-scale confusion across unrelated categories. The model tends to predict 
the visually dominant item while missing secondary categories that occupy less 
visual space or have lower contrast.
```

---

## How to Use This in Your Report

### Full Integration (All 5 Examples)

Add this entire section after your confusion matrix analysis in Task 1:

```latex
\subsection{Error Analysis}

\subsubsection{Confusion Matrix}
[Your existing confusion matrix content...]

\subsubsection{Example Failure Cases}
[Paste the generated LaTeX code here]
```

### Minimal Version (1-2 Examples Only)

If you want to keep it short, pick just 1-2 examples:

```latex
\subsubsection{Example Failure Cases}

A few typical failure cases help explain the remaining errors:

\begin{figure}[H]
    \centering
    \includegraphics[width=0.7\textwidth]{figures/failure_examples/failure_example_1.png}
    \caption{\textbf{Multi-Item Omission.} Ground truth: \texttt{Granola} (2), \texttt{Dairy} (1). Prediction: \texttt{Granola} (2). The model predicted only \texttt{Granola} and missed \texttt{Dairy}.}
    \label{fig:failure_example_1}
\end{figure}

This example is consistent with the main pattern: the dominant issue is 
\textbf{omission in multi-item scenes}. The model tends to predict the 
visually dominant item while missing secondary categories.
```

### Text-Only Version (No Figures)

If you're worried about page count, you can describe the failures without images:

```latex
\subsubsection{Example Failure Cases}

A few typical failure cases help explain the remaining errors:

\begin{itemize}
    \item \textbf{Multi-item omission:} In one image containing both \texttt{Granola} products and \texttt{Dairy} items, the model predicted only \texttt{Granola} and missed the secondary \texttt{Dairy} category. Similarly, in another image with both \texttt{Ready Meals} and \texttt{Carbohydrate Meal} items, only \texttt{Carbohydrate Meal} was predicted.
    
    \item \textbf{Visually similar categories:} Some confusion occurs between \texttt{Carbohydrate Meal} and \texttt{Ready Meals}, which often share similar packaging and visual appearance in pantry settings.
\end{itemize}

These examples are consistent with the main pattern in the confusion analysis: 
the dominant issue is \textbf{omission in multi-item scenes} rather than 
large-scale confusion across unrelated categories.
```

---

## Customizing Captions

You can edit the captions to be more concise:

**Before:**
```latex
\caption{\textbf{Multi-Item Omission.} Ground truth: \texttt{Granola} (2), \texttt{Dairy} (1). Prediction: \texttt{Granola} (2). The model predicted only \texttt{Granola} and missed \texttt{Dairy}.}
```

**After (shorter):**
```latex
\caption{Multi-item omission: model predicted \texttt{Granola} but missed \texttt{Dairy}.}
```

---

## Figure Placement Tips

- `[H]` = "Here" (exactly where you put it)
- `[t]` = Top of page
- `[b]` = Bottom of page
- `[p]` = Separate page for figures

If your figures are jumping around, try:
```latex
\begin{figure}[!htb]  % Force "here, top, or bottom"
```

Or use `\FloatBarrier` (requires `\usepackage{placeins}`):
```latex
\FloatBarrier
\begin{figure}[H]
...
\end{figure}
\FloatBarrier
```

---

## Expected Visual Output

Each generated PNG will show:

```
┌─────────────────────────────────────────┐
│                                         │
│         [Pantry Shelf Image]            │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │ Ground Truth:                     │  │
│  │   • Granola: 2                    │  │
│  │   • Dairy: 1                      │  │
│  │                                   │  │
│  │ Prediction:                       │  │
│  │   • Granola: 2                    │  │
│  │                                   │  │
│  │ ⚠ MISSED: Dairy                   │  │
│  └───────────────────────────────────┘  │
│                                         │
└─────────────────────────────────────────┘
     Failure Example 1: Multi-Item Omission
```

The white text box shows:
- Ground truth (what should be detected)
- Prediction (what the model actually predicted)
- Highlighted differences (missed or wrong categories)

---

## JSON Output Example

`failure_descriptions.json` will contain:

```json
[
  {
    "example_num": 1,
    "failure_type": "multi_item_omission",
    "image_path": "test_data/images/shelf_042.jpg",
    "output_filename": "failure_example_1.png",
    "gt_counts": {
      "Granola": 2,
      "Dairy": 1
    },
    "pred_counts": {
      "Granola": 2
    },
    "description": "The model predicted only \\texttt{Granola} and missed \\texttt{Dairy}."
  },
  ...
]
```

This is useful if you want to programmatically process the results or create custom visualizations.
