# Week 1 exercise

## 1. Measure the strawman

Pick 5 questions you'd actually ask your second brain. Examples from the
bundled vault to seed your thinking:

- *Lookup:* "What was the chunker bug?"
- *Synthesis:* "Summarize what I've written about evals."
- *Exploration:* "Show me notes connecting Anchor to retrieval papers."
- *Recency:* "What was I thinking about yesterday?"
- *Comparison:* "What's the difference between BM25 and dense retrieval?"

Run each against the baseline:

```bash
uv run anchor ask "<your question>" --vault tests/fixtures/vault --show-hits
```

For each question, write down:

| Question | Did top-3 contain the right note? | Did the answer cite correctly? | Did it miss a note you knew was relevant? |
|----------|-----------------------------------|--------------------------------|-------------------------------------------|

You now have a baseline scorecard. Save it — week 4 will compare against it.

## 2. Find a vector-only failure

Find one question where you *know* the right answer is a note that the
vector retriever misses, and you can prove it from the graph. Hint: pick a
note that's heavily linked but whose body uses different vocabulary than
your question.

Why does vector retrieval miss it? What signal would have caught it? (You'll
build that signal in week 3.)

## 3. (Optional) Point at your own vault

Set `ANCHOR_VAULT=~/your/notes` and run the same 5 questions. The synthetic
vault is too clean to expose real failure modes — your actual notes will.
