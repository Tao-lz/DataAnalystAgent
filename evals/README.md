# 模型评测（evals）

`tests/` 里的 29 个离线用例保证**逻辑正确**；这里的评测集回答的是**答得准不准**：
意图判对没、工具选对没、关键数字说对没。

## 快速开始

```powershell
cd C:\Users\李自涛\Desktop\code\DataAnalysisAgent

# 先跑 3 条试水（约 30 秒）
python evals/run_evals.py --limit 3

# 跑全部 20 条
python evals/run_evals.py

# 只跑某一类
python evals/run_evals.py --only viz
python evals/run_evals.py --only cleaning

# 每条跑 3 次，降低随机性
python evals/run_evals.py --repeat 3

# 清空评测存储后重来
python evals/run_evals.py --fresh
```

评测使用独立存储 `evals/.storage/`，**不会污染日常使用的数据**。
每次运行会在 `evals/reports/` 下生成 `eval-<时间>.md` 和 `.json`。

## 指标定义

| 指标 | 怎么算 | 看什么 |
| --- | --- | --- |
| 用例通过率 | 所有检查项都通过的用例占比 | 整体健康度 |
| 意图准确率 | 实际意图 ∈ 期望意图列表的用例占比 | 路由节点准不准 |
| 工具召回率 | 命中的必需工具数 / 必需工具总数 | 该调的有没有调 |
| 工具精确率 | 调用落在允许集合内的比例 | 有没有乱调 |
| 参数准确率 | 关键参数（列名、分组字段）填对的用例占比 | 参数填得对不对 |
| 关键数字命中率 | 期望数字出现在答案里的比例（相对误差 ≤1%） | 有没有算错/说错 |
| 平均轮次 / 延迟 / token | 全部用例的均值 | 成本与体验 |

**通过标准**：意图对 + 必需工具齐全 + 关键参数对 + 关键数字命中 +（如需要）报告生成成功。
只比工具集合、不比调用顺序——先调概览还是先调统计都算对。

## 用例格式

`cases.jsonl` 一行一条：

```json
{
  "id": "viz-01",
  "tag": "visualization",
  "question": "各地区的销售额对比，用柱状图展示",
  "intents": ["visualization"],
  "required_tools": ["plot_bar"],
  "allowed_tools": ["dataset_overview", "compute_statistics", "plot_bar"],
  "expected_args": {"plot_bar": {"x": "地区", "y": "销售额"}},
  "numbers": [2260964.83]
}
```

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `id` | 是 | 唯一标识，也是 `--only` 的匹配目标 |
| `tag` | 是 | 分类：chat / overview / statistics / cleaning / visualization / report / adversarial |
| `question` | 是 | 发给 Agent 的自然语言 |
| `intents` | 是 | 允许的意图（列表，允许多个） |
| `required_tools` | 否 | 必须调用的工具 |
| `allowed_group` | 否 | 允许调用的意图分组（overview / statistics / cleaning / visualization），比逐个列工具更好维护 |
| `allowed_tools` | 否 | 精确指定允许调用的工具；不写则不限制 |
| `expected_args` | 否 | 关键参数，例如 `{"plot_bar": {"x": "地区"}}` |
| `numbers` | 否 | 答案里必须出现的关键数字（支持 1% 相对误差） |
| `expect_keywords` | 否 | 答案里至少要出现其中一个词（用于"该拒绝时是否拒绝"） |
| `expect_report` | 否 | 是否必须产出 `report_id` |
| `no_tools` | 否 | 为 true 时，调用任何工具都算失败（用于闲聊） |

### 数字匹配规则

答案里的 `2,260,964.83`、`226.1万`、`2260964.83` 都能命中期望值 `2260964.83`：

- 先去掉千分位逗号
- 识别 `万 / 亿 / 千` 并换算
- 相对误差 ≤1% 视为命中

百分比按原值比较（`29.9%` 对应期望 `29.9`），不要把 29.9 写成 0.299。

## 怎么加用例

1. 在 `cases.jsonl` 末尾加一行（必须是合法 JSON，不能有注释）
2. 先用 `--only <新 id 前缀>` 单独跑一条，确认期望值是对的
3. 期望数字建议从**真实工具输出**里抄，不要手算

## 注意事项

### 2026-11-20 基线（deepseek-chat，20 条用例）

| 指标 | 结果 |
| --- | --- |
| 用例通过率 | 100%（20/20） |
| 意图准确率 | 100% |
| 工具召回率 | 100% |
| 工具精确率 | 100% |
| 参数准确率 | 100% |
| 关键数字命中率 | 100% |
| 平均轮次 / 延迟 | 2.2 轮 / 11.3 s |
| 单条最多工具调用 | 27 次（adv-02 预测未来时反复 filter_rows） |

这个基线只说明**主路径没翻车**，不代表模型已经很好：20 条偏简单，下一步应该扩到 50+ 条、
加更刁的对抗用例，并用 `--repeat 3` 看稳定性。`adv-02` 的 27 次调用是最明显的优化点。

- **每条用例前会复位数据**：清洗类用例会改工作副本，评测脚本会自动切回 `_v0`。
- **随机性**：即使 temperature=0，模型输出也不完全一致。重要结论用 `--repeat 3` 再下。
- **对抗用例的意图是放宽的**：问一个不存在的字段，回答"没有这个字段"属于正确答案，走 chat 或 statistics 都算对。
- **平均 token 是累计值**：等于本轮所有模型调用的 token 之和（每轮都会重发历史），不是单次请求的 token。
- **成本**：20 条一轮大约 40~60 次模型调用，跑一轮几分钟、几分钱。
- **别过拟合**：调 prompt 时留 5~6 条不参与调试，作为 holdout。
- **评测存储可删**：`evals/.storage/` 全是评测产生的数据，随时可以 `--fresh` 重建。

## 常见问题

**跑完数字命中率很低？**
先看报告里"未命中数字"是哪几个。常见原因：模型把 `2260964.83` 写成 `226.1 万`（能被换算命中），
或者答案里根本没引用这个数字（说明工具 summary 没给到关键值，需要改工具返回）。

**某个用例总判失败？**
看 `evals/reports/eval-*.md` 里的"失败用例的回答片段"。先判断是模型错了，
还是你的 `required_tools` / `numbers` 标注错了——标注错比模型错更常见。

**想对比不同模型？**
改 `.env` 里的 `LLM_MODEL` / `LLM_BASE_URL`，重新跑一遍，对比两份报告即可。
