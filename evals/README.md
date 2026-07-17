# Agent Eval

该目录比较基础 ReAct（`baseline`）与事件驱动 Task Harness。评测调用真实配置的
LLM，但工具全部为确定性 Fake Tools：不会联网、读取真实文件、写 Excel 或发送邮件，
也不包含当前不可用的微信能力。

## 快速运行

先安装项目和测试依赖，并在 `.env` 配置模型 API Key：

```bash
pip install -r requirements.txt -r requirements-dev.txt

# 先用 12 个 case 验证链路
python -m evals.runner --limit 12 --repeats 1

# 正式对比：30 个任务各运行 3 次
python -m evals.runner --repeats 3
```

默认模型为 `glm-4-flash`。结果写入：

- `evals/results/runs.jsonl`：每次运行的完整工具轨迹、得分和失败标签。
- `evals/results/comparison.md`：完成率、平均调用数、延迟和失败类型对比。

Eval 默认关闭 LangSmith，避免批量运行产生大量 trace；需要时设置
`EVAL_LANGSMITH_TRACING=1` 再运行。

可单独运行一个版本：

```bash
python -m evals.runner --variants harness --limit 5
```

## 公平性约束

两个版本使用相同模型、主提示词、Fake Tools、输入和调用预算。Harness 版本使用数据集
中的固定计划，以避免 planner 随机性污染对比；唯一主要变量是显式步骤状态、阶段 gate、
重锚定与自动续跑。baseline 只运行基础 ReAct，不执行 Harness 自动续跑。

当前评分以确定性规则为主：必需/禁止工具、调用顺序、参数、调用预算、答案关键事实、
重复外发和虚假成功。后续可以另加 LLM Judge，但不应让它替代可直接检查的工具轨迹。
