# 示例视频

把待编译的视频放在这个目录，然后运行：

```bash
python compile.py --input examples/
```

视频文件本身不会提交到仓库（见 `.gitignore`）。可以放：

- 自己拍摄的第一人称操作视频
- 从公开数据集下载并解压出的视频
- 任何包含连续操作动作的长视频（建议 2 分钟以上，太短的片段切不出周期）

## 建议的测试顺序

1. 先放一小段（2~5 分钟）跑通全流程，确认密钥和依赖没问题
2. 用 `--stage segment` 检查切分是否合理
3. 调整 `--min-peak-ratio` 到满意后，再跑完整编译

```bash
# 第一步：只切分，不花钱
python compile.py --input examples/your_clip.mp4 --stage segment

# 第二步：确认片段数和边界合理后，完整编译
python compile.py --input examples/your_clip.mp4
```
