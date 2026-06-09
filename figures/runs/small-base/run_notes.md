# Run: small-base

**Architecture:** d_model=768, 12 heads, 12 layers, d_ff=3072, seq_len=512  
**Config:** use_rope=False, n_kv_heads=None (standard MHA), GELU, Pre-LN, weight tying  
**Steps:** 100,000  
**Hardware:** RTX 5070 Laptop  
**Runtime:** ~12 hours  

## Results
- Best val loss: **4.7861**
- Best val perplexity: **~119**
- Final val loss: ~4.79

## Notes
- The run_log.pt for this run was overwritten by the subsequent RoPE run before it could be archived.
- Figures above were saved from the committed state (commit 2e67305) before the next run started.
- Key metrics are preserved in `notes/Experiments/2026-06-07-small-run.md` in the Obsidian vault.
