# 数学建模 AI 论文审计 Skill

继承于math model agent的ai论文结果，用于审计 AI 生成的数学建模成果，定位影响模型正确性、合理性、稳健性和最终决策的问题，并给出可执行的验证、修复与再验证路线。

## 主要内容

- `math_model_paper_audit_skill_v0.2.md`：当前版本的审计 Skill。
- `math_model_paper_audit_skill_v0.1.md`：上一版本，便于对照。
- `24c/`、`25c/`：测试材料、审计报告、代码与结果样例。

## 使用方式

优先阅读 `math_model_paper_audit_skill_v0.2.md`。该版本支持快速分诊（FAST / TRIAGE）和深度审计（DEEP AUDIT）两种模式。

仓库已排除 Python 虚拟环境、缓存和临时审计产物，以保持版本历史精简。
