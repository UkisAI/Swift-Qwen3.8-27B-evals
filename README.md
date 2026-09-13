# Swift-Qwen3.8-27B evaluation logs

Raw artifacts for the nine benchmarks on the [Swift-Qwen3.8-27B](https://huggingface.co/ukisai/Swift-Qwen3.8-27B) model card: Qwen3.8-27B BF16 base vs base + Swift adapter. Every per-sample response, score, config, and run log is included as produced.

| Folder | Benchmark |
|---|---|
| 01-gpqa-diamond | GPQA-Diamond (198 x 5 seeds) |
| 02-mmlu-pro | MMLU-Pro (12,032 x 5 seeds) |
| 03-c-eval | C-Eval (1,346 x 5 seeds) |
| 04-ifbench | IFBench (k=5) |
| 05-aime-2026 | AIME 2026 (MathArena, 5 seeds) |
| 06-hmmt-nov-2025 | HMMT Nov 2025 (MathArena, 5 seeds) |
| 07-erqa | ERQA (400 x 5 seeds) |
| 08-terminal-bench-2.1 | Terminal-Bench 2.1 (Harbor, terminus-2, k=5) |
| 09-livecodebench-v6 | LiveCodeBench release_v6 (1,055 x 5 seeds) |

Arm labels: `base` = Qwen3.8-27B BF16, `swift` = the same base with the Swift LoRA. Two Terminal-Bench files over GitHub's size limit were omitted (`08-terminal-bench-2.1/swift/qemu-startup__bcXSxBZ/agent/recording.cast` 926 MB and `terminus_2.pane` 729 MB). Token-looking strings inside Terminal-Bench trials belong to the `sanitize-git-repo` task itself.
