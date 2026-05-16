# vif-banana

# (a) Run judge (Gemini judge × all generators × all tasks)
`python3 judge.py --judge gemini`

# Filter by generator / filter by task
```
python3 judge.py --judge gpt --generator nano_banana_pro
python3 judge.py --judge gemini --task n3_103
python3 judge.py --judge gpt --overwrite    # Overwrite existing judgments
```

# (b) Aggregate
```
python3 aggregate_judge.py
# → judge_results/{overall_summary, by_main_count, per_task}.{csv,json}
```
