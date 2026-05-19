# PRD: Expert-Acquisition Agent System (EAAS)

**Version**: 1.2-spec

**Status**: Engineering-Ready

**Change log**: v1.2 新增 §2（产品表面、角色与旅程、REST/Webhook 契约、可见交付物）；原 §2–§16 顺延为 §3–§17。

**Audience**: §2 描述产品表面、集成方式与可见交付物（业务/集成/运维）；§3 起为工程语义与实现契约。

---

## 1. System Overview

### 1.1 Purpose

将「能在陌生领域内快速成长为专家」的**可形式化元能力**，以**可量化、可验证**方式内置于 LLM-Agent 系统。系统接收：领域规约、测试任务集、资源预算；输出满足领域专家阈值的交付结果。

### 1.2 Job-To-Be-Done (Locked)

Given:

- `domain` ∈ **FormallyVerifiableDomain** ∪ **KnowledgeIntensiveDomain**
- `DomainTestTaskSet`：与该 `domain` 对齐的基准/现实交付任务集
- `ResourceBudget`：`tokenCap` / `roundCap` / `toolCallCap`

Do: 在 `ResourceBudget` 内完成专业能力构建循环。

Deliver: 对 `DomainTestTaskSet` 的解集，使 `TaskPassRate(solutions, DomainTestTaskSet) ≥ DomainExpertThreshold[domain]`。

Oracle（锁定）:

- 公认基准得分 **与**
- 现实任务交付可验证结果

### 1.3 Out-of-Scope（硬边界）

- 无客观 oracle 的开放世界领域 → 拒绝
- 超低延迟（如单次 < 1s）场景 → 非本 PRD 目标
- 以人类主观满意为唯一终止条件 → 拒绝
- 无法绑定 sandbox / verifier 的封闭域 → 拒绝

**说明**：本产品「长什么样、怎么用」见 **§2 Product Surface, Experience & External Contract**；§1 定义能力与 JTBD，不假定具体 UI 框架。

---

## 2. Product Surface, Experience & External Contract

本节定义首版对外边界：**谁在什么环境里触发一次 Expert Run、系统返回什么、成功/失败如何被人类理解**。内部 Phase 机与模块名保持不变。

### 2.1 Product Surface（首版锁定）

| 层级 | 形态 | 职责 |
|------|------|------|
| **Primary** | **HTTPS REST API**（异步长作业） | 创建/查询 Expert Run；下载工件与报告；可选 Webhook |
| **Secondary（同版本可选）** | **Run Observer UI**（只读仪表盘） | 列表/详情/Phase 时间线/预算消耗；面向运维与演示 |
| **Embed** | **宿主应用内 SDK 调用同一 REST**（thin wrapper） | 将 EAAS 作为子系统嵌入既有平台 |

**首版不包含（明确写出以免误解 PRD）**：面向终端用户的闲聊式 Chat 作为主入口、独立消费者 App、语音交互、亚秒级同步对话 UX。

### 2.2 Roles（部署视角）

| Role | 典型身份 | 主要职责 |
|------|----------|----------|
| `PlatformOperator` | 平台/研发 owning EAAS | 维护 `DomainProfileRegistry`、`DomainToolRegistry`、`AuthorityCorpusRetriever` 语料与密钥 |
| `IntegrationClient` | 上游业务、CI、实验室流水线 | 仅通过 API 提交 `ExpertRun`，消费工件与指标 |
| `RunObserver` | SRE、项目经理、演示受众 | 只读 Run 状态与报告（UI 或 API GET） |

### 2.3 Primary User Journeys

#### Journey A — 配置域（一次性或低频）

1. `PlatformOperator` 提交或更新 `DomainProfile`（`domainId`、`domainClass`、`oracleBinding`、`domainExpertThreshold`、`recommendedToolIds`、`authoritySources`）。
2. 若域未注册：`Phase.DomainProfiling` 探测子流程在**首次 Expert Run**时自动触发（+30% Token 溢价由 `ResourceBudgetGuard` 记入 Run 元数据）。
3. 校验：`DomainProfile.callableOracle` 与工具链在 staging 可用 → 标记 `bootstrapStatus: registered | probed`。

（内部映射：`DomainProfileRegistry`、`PhaseOrchestrator` 入口。）

#### Journey B — 发起一次「学得怎么样」的长作业（主路径）

1. `IntegrationClient` 注册或引用 `DomainTestTaskSet`（上传任务包 ID 或内联清单），并 **POST `/v1/expert-runs`**，携带 `resourceBudget`。
2. 系统创建 `expertRunId`，异步执行 Phase 流水线（见 §6）。
3. 客户端轮询 **GET `/v1/expert-runs/{expertRunId}`** 或订阅 **Webhook**（`ExpertRun.PhaseTransitioned`、`ExpertRun.Completed`、`ExpertRun.Aborted`）。
4. 终态：`status: succeeded | incomplete | aborted`，下载 **Run Report + Artifacts**（见 §2.7）。

（内部映射：`PhaseOrchestrator` → `Phase.FinalDelivery` / `HALT`。）

#### Journey C — 运维排障（可选 UI）

1. `RunObserver` 打开 Run Observer UI 或调用 GET。
2. 查看 `currentPhase`、`budgetUtilization`、`lastFailureModeId`（如 `FM.BootstrapInfeasible`）。
3. 若需人工介入：响应体含 `escalationHint`（例如更换 oracle、补充工具白名单）。

### 2.4 Resource Naming（REST）

统一前缀：`/v1/`。核心资源：

| Resource | 语义 |
|----------|------|
| `DomainProfile` | 域配置（注册表项） |
| `DomainTestTaskSet` | 考卷/任务包 |
| `ExpertRun` | 一次端到端 Expert Acquisition 作业 |

### 2.5 REST Contract（规范性示例）

以下为 **API 形状约定**；字段名与 PRD 其余章节 camelCase 对齐，实现时可等价映射到 gRPC/protobuf。

#### 2.5.1 Register or update domain profile

`PUT /v1/domains/{domainId}`

Request body（与 §10 `DomainProfile` 对齐的子集；服务端可拒绝不完整注册）：

```json
{
  "domainClass": "FormallyVerifiable",
  "oracleBinding": {
    "oracleKind": "both",
    "benchmarkIds": ["humaneval-python"],
    "deliveryValidatorCallableId": "validators.python_exec_v2"
  },
  "domainExpertThreshold": 0.85,
  "recommendedToolIds": ["python.executionSandbox.v1"],
  "authoritySources": [{ "uri": "https://example.org/corpus/math", "weight": 0.9 }]
}
```

Response：`200` + `{ "domainId": "...", "bootstrapStatus": "registered" }`

#### 2.5.2 Start Expert Run

`POST /v1/expert-runs`

```json
{
  "domainId": "leetcode-hard-python",
  "domainTestTaskSetRef": "taskset://uuid-or-upload-id",
  "resourceBudget": {
    "tokenCap": 8000000,
    "roundCap": 500,
    "toolCallCap": 2000
  },
  "webhookUrl": "https://customer.example/hooks/eaas",
  "webhookSecretRef": "secret://kms-key-id",
  "clientReferenceId": "ci-job-9042"
}
```

Response `202 Accepted`：

```json
{
  "expertRunId": "run_a3f9c2",
  "status": "queued",
  "pollUrl": "/v1/expert-runs/run_a3f9c2",
  "estimatedComplexity": "KnowledgeIntensive"
}
```

#### 2.5.3 Get Expert Run status（轮询）

`GET /v1/expert-runs/{expertRunId}`

Response（进行中示例）：

```json
{
  "expertRunId": "run_a3f9c2",
  "domainId": "leetcode-hard-python",
  "status": "running",
  "currentPhase": "Phase.DeliberatePracticeLoop",
  "phaseHistory": [
    { "phase": "Phase.DomainProfiling", "startedAt": "...", "endedAt": "..." },
    { "phase": "Phase.OntologyConstruction", "startedAt": "...", "endedAt": "..." }
  ],
  "budgetUtilization": { "tokens": 0.41, "rounds": 0.22, "toolCalls": 0.18 },
  "metricsPreview": {
    "taskPassRate": null,
    "usefulToolCallRatio": 0.61
  },
  "deliveredUnderBudgetCeiling": false
}
```

Response（终态成功示例）：

```json
{
  "expertRunId": "run_a3f9c2",
  "status": "succeeded",
  "currentPhase": "Phase.FinalDelivery",
  "taskPassRate": 0.87,
  "domainExpertThreshold": 0.85,
  "deliveredUnderBudgetCeiling": false,
  "artifactBundleUrl": "https://storage.example/bundles/run_a3f9c2.zip",
  "runReportUrl": "https://storage.example/reports/run_a3f9c2.md"
}
```

Response（预算耗尽未达阈，`INV-7`）：

```json
{
  "expertRunId": "run_a3f9c2",
  "status": "incomplete",
  "taskPassRate": 0.72,
  "domainExpertThreshold": 0.85,
  "deliveredUnderBudgetCeiling": true,
  "artifactBundleUrl": "...",
  "runReportUrl": "..."
}
```

Response（探测失败，`FM.BootstrapInfeasible`）：

```json
{
  "expertRunId": "run_a3f9c2",
  "status": "aborted",
  "abortReason": "FM.BootstrapInfeasible",
  "escalationHint": "Register oracleBinding or provide DomainToolRegistry entries for this domain."
}
```

### 2.6 Webhook Payload（可选）

`POST` 至客户 `webhookUrl`，`Content-Type: application/json`，建议 HMAC（`webhookSecretRef`）。

事件类型：

| `eventType` | 含义 |
|-------------|------|
| `ExpertRun.PhaseTransitioned` | `currentPhase` 变化 |
| `ExpertRun.Completed` | 终态：`succeeded` / `incomplete` |
| `ExpertRun.Aborted` | 不可恢复中止 |

示例：

```json
{
  "eventType": "ExpertRun.Completed",
  "expertRunId": "run_a3f9c2",
  "status": "succeeded",
  "occurredAt": "2026-05-12T06:40:00Z"
}
```

### 2.7 User-Visible Deliverables（人类可读产出）

下列必须能通过 **artifact bundle** 或 **report URL** 获取（具体存储实现不限）：

| 产出物 | 内容 |
|--------|------|
| **Run Summary** | `taskPassRate`、`domainExpertThreshold`、`deliveredUnderBudgetCeiling`、总 Token/Round/Tool 消耗、终态 `currentPhase` |
| **Phase Timeline** | 各 `Phase.*` 起止时间与可选结构化日志指针 |
| **Per-Task Results** | 每题/每任务的 pass/fail、oracle 原始输出摘要 |
| **Artifact Files** | 代码补丁、生成脚本、`ExecutionSandbox` 日志等（依 `DomainClass` 而定） |
| **Verifier Summary** | 三角色聚合：`pass` / `veto` / `abstain` 计数（全文推理可选下沉到 debug 层） |
| **Failure Narrative** | 若 `status: aborted`，人类可读说明 + `failureModeId`（如 `FM.CorpusPollution`） |

调试层（默认不对 `IntegrationClient` 强制开放）：完整 `VerifierVerdict`、`PracticeEpisode` trace，由 `PlatformOperator` 配置 ACL。

### 2.8 Mapping：对外概念 ↔ 内部 PRD

| 对外（§2） | 内部（本文其余章节） |
|------------|----------------------|
| `ExpertRun` | `PhaseOrchestrator` 一次会话实例 |
| `currentPhase` | `Phase.*` FSM 当前状态 |
| `budgetUtilization` | `ResourceBudgetGuard` 聚合 |
| `taskPassRate` / `artifactBundleUrl` | `Phase.FinalDelivery` 产出 |
| `abortReason` | `FM.*`（§12） |

---

## 3. Glossary & Symbols


| 标识                                        | 全称含义（正文/Schema 使用）                          |
| ----------------------------------------- | ------------------------------------------- |
| `domain`                                  | 待掌握的领域实例                                    |
| `DomainClass`                             | `FormallyVerifiable` 或 `KnowledgeIntensive` |
| `DomainTestTaskSet`                       | 该域的测试任务集合                                   |
| `DomainExpertThreshold[domain]`           | 该域专家级通过阈值（0–1）                              |
| `RoundsToExpertise`（R^*）                  | 首次达阈的轮次数                                    |
| `InformationGainAtRound[n]`（IG_n）         | 第 n 轮信息增益代理量                                |
| `ExpertiseConvergenceQuotient`            | `TasksPassed / TokensConsumed`              |
| `UsefulToolCallRatio`（\eta_{\text{tool}}） | 有用工具调用 / 总工具调用                              |
| `MethodologyMemory`                       | 跨域常驻方法论记忆                                   |
| `DomainKnowledgeMemory`                   | 领域知识记忆                                      |
| `CrossDomainAnalogyIndex`                 | 跨域同构类比索引                                    |
| `VerifierEnsemble`                        | 三角色验证子 Agent 集合                             |


**约定**：R^*、IG_n、\eta_{\text{tool}}、\tau_{\text{domain}} 仅出现于数学块；正文与 JSON 使用上表全称或 camelCase 字段。

---

## 4. System Invariants (Non-Negotiable)


| ID    | 不变量                                                                                  |
| ----- | ------------------------------------------------------------------------------------ |
| INV-1 | 写入 `MethodologyMemory` 必须经 `VerifierEnsemble` 至少 **2/3** `pass`                      |
| INV-2 | `CrossDomainAnalogyIndex` 每条必须含非空 `counterEvidence`                                  |
| INV-3 | 每次工具调用必经 `ToolCallSchemaValidator`                                                   |
| INV-4 | 单 Phase 消耗超过该 Phase 配额的 **110%** → 强制终止该 Phase                                       |
| INV-5 | 进入 `Phase.FinalDelivery` 前，`VerifierEnsemble` 三角色各至少一次 `pass`（历史累计）                  |
| INV-6 | `domain` 未在 `DomainProfileRegistry` 注册 → 强制执行 `Phase.DomainProfiling` 内探测子流程         |
| INV-7 | 总预算 ≥ **90%** 且未达 `DomainExpertThreshold` → 提前交付，`deliveredUnderBudgetCeiling: true` |
| INV-8 | `VerifierEnsemble` 三角色上下文**互不可见**（独立会话）                                              |


---

## 5. Top-Level Architecture

### 4.1 Modules


| Module                     | Responsibility                                                                 |
| -------------------------- | ------------------------------------------------------------------------------ |
| `PhaseOrchestrator`        | Phase 调度、转移条件仲裁                                                                |
| `ResourceBudgetGuard`      | Token / Round / ToolCall 配额                                                    |
| `InformationGainMonitor`   | 信息增益代理、`InformationGainPlateau` 早停                                             |
| `DomainProfileRegistry`    | `domain` → 工具、语料、阈值、oracle 绑定                                                  |
| `AgentMemoryStore`         | `MethodologyMemory` / `DomainKnowledgeMemory` / `CrossDomainAnalogyIndex` 统一接口 |
| `VerifierEnsemble`         | `FactualSoundnessChecker` + `LogicalValidityAuditor` + `RobustnessAdversary`   |
| `DomainToolRegistry`       | 域内工具白名单与 JSON Schema                                                           |
| `ToolCallSchemaValidator`  | 工具入参强校验                                                                        |
| `ToolFallbackGateway`      | 工具失败降级链                                                                        |
| `AuthorityCorpusRetriever` | 权威语料检索与引用追溯                                                                    |
| `ExecutionSandbox`         | 代码/仿真隔离执行                                                                      |


### 4.2 Control-Flow Topology (Mermaid)

```mermaid
flowchart TB
    USER([User / Upstream]) -->|domain, DomainTestTaskSet, ResourceBudget| ORCH

    subgraph META["Meta Controller"]
        ORCH[PhaseOrchestrator]
        BUDGET[ResourceBudgetGuard]
        IG[InformationGainMonitor]
    end

    subgraph PHASES["Expert Acquisition Lifecycle FSM"]
        P0[Phase.DomainProfiling] --> P1[Phase.OntologyConstruction]
        P1 --> P2[Phase.CrossDomainAnalogyInjection]
        P2 --> P3[Phase.DeliberatePracticeLoop]
        P3 --> P4[Phase.VerifierEnsembleAudit]
        P4 -->|veto loop| P3
        P4 -->|accept| P5[Phase.MemoryWriteback]
        P5 --> P6[Phase.FinalDelivery]
        P6 -->|info gain not saturated| P2
    end

    subgraph MEM["Memory"]
        MMETA[(MethodologyMemory)]
        MDOM[(DomainKnowledgeMemory)]
        MANA[(CrossDomainAnalogyIndex)]
        REG[(DomainProfileRegistry)]
    end

    subgraph VE["VerifierEnsemble"]
        FC[FactualSoundnessChecker]
        LA[LogicalValidityAuditor]
        RA[RobustnessAdversary]
    end

    subgraph EXEC["Execution"]
        TR[DomainToolRegistry] --> SV[ToolCallSchemaValidator] --> FG[ToolFallbackGateway]
        SBX[ExecutionSandbox]
        RAG[AuthorityCorpusRetriever]
    end

    ORCH --> P0
    P6 --> USER
    BUDGET -.约束.-> PHASES
    IG -.信号.-> ORCH
    P0 -.读.-> REG
    P1 -.写.-> MDOM
    P2 -.读写.-> MANA
    P4 --> FC & LA & RA
    P5 -.写.-> MMETA & MDOM & MANA
    PHASES -.调用.-> EXEC
    FC -.读.-> RAG
```



---

## 6. Phase Machine Specification

### 5.1 Phases


| Phase                               | Entry         | Exit                                          | Default Token Quota   | ReAct 上限 |
| ----------------------------------- | ------------- | --------------------------------------------- | --------------------- | -------- |
| `Phase.DomainProfiling`             | 新会话或域未注册      | `DomainProfile` 完整或 user-escalate             | 5%（**首次域 +30%** 探测溢价） | 4        |
| `Phase.OntologyConstruction`        | Profile 就绪    | Ontology：`nodeCount ≤ 200` 且 `axiomCount ≥ 5` | 20%                   | 8        |
| `Phase.CrossDomainAnalogyInjection` | Ontology 就绪   | ≥1 有效类比且 `counterEvidence` 非空；或进入无类比模式        | 8%                    | 5        |
| `Phase.DeliberatePracticeLoop`      | 上一 Phase 就绪   | `InformationGainPlateau` 或 round cap 或审核触发    | 30%                   | 12       |
| `Phase.VerifierEnsembleAudit`       | 练习回合产出        | 三角色 `VerifierVerdict` 齐全                      | 15%                   | 每角色 3    |
| `Phase.MemoryWriteback`             | 审核通过（≥2 pass） | 写回提交                                          | 5%                    | 2        |
| `Phase.FinalDelivery`               | 写回完成或 IG 饱和   | `passRate` 达阈或预算 90%                          | 17%                   | 6        |


### 5.2 Transitions

- `Phase.DomainProfiling` → `Phase.OntologyConstruction` 当 `domainProfileReady`
- `Phase.DomainProfiling` → `ABORT(reason: "FM.BootstrapInfeasible")` 当 `profileUnsolvable` 或 `domainProfilingBudgetExceeded110Percent`
- `Phase.OntologyConstruction` → `Phase.CrossDomainAnalogyInjection` 当 `ontologyReady`
- `Phase.CrossDomainAnalogyInjection` → `Phase.DeliberatePracticeLoop` 当 `analogyReady` 或 `noValidAnalogy`（无对比模式）
- `Phase.DeliberatePracticeLoop` → `Phase.VerifierEnsembleAudit` 当 `informationGainPlateauForMConsecutiveRounds` 或 `practiceRoundCapHit`
- `Phase.VerifierEnsembleAudit` → `Phase.MemoryWriteback` 当 `verifierAggregateAccept`（≥2 pass）
- `Phase.VerifierEnsembleAudit` → `Phase.DeliberatePracticeLoop` 当 `verifierAggregateReject`（≥2 veto）或部分重试分支
- `Phase.MemoryWriteback` → `Phase.FinalDelivery` 当 `memoryWritebackCommitted`
- `Phase.FinalDelivery` → `HALT(success)` 当 `passRate ≥ DomainExpertThreshold[domain]`
- `Phase.FinalDelivery` → `Phase.CrossDomainAnalogyInjection` 当 `informationGainNotSaturated` 且 `budgetUtilization < 90%`
- `ANY` → `HALT(deliveredUnderBudgetCeiling: true)` 当 `budgetUtilization ≥ 90%` 且未达阈

默认早停：`InformationGainPlateau` 判定沿用 ε=0.08、m=3，直至实测校准。

### 5.3 Phase Exit Assertions（示意）

- `Phase.DomainProfiling`：`domainProfile.toolsNonEmpty`、`domainProfile.callableOracle`、`domainExpertThresholdInZeroOne`
- `Phase.OntologyConstruction`：`nodeCount ≤ 200`、`axiomCount ≥ 5`、`authorityChunkScore ≥ 0.7`（可配置）
- `Phase.CrossDomainAnalogyInjection`：每条类比 `counterEvidence` 非空；`analogyConfidence ≥ 0.5` 或丢弃
- `Phase.DeliberatePracticeLoop`：至少一条 `PracticeEpisode` 且含 `informationGainProxy`
- `Phase.VerifierEnsembleAudit`：三条 `VerifierVerdict` 且各含 `supportingEvidence`
- `Phase.MemoryWriteback`：`verifierEnsembleSignatures` 至少 2×`pass`
- `Phase.FinalDelivery`：交付物数量与 `DomainTestTaskSet` 对齐；报告 `passRate`

---

## 7. AgentMemoryStore Data Model

### 6.1 `MethodologyMemory`

```jsonc
{
  "schemaVersion": "string",
  "entries": [{
    "entryId": "uuid",
    "primitive": "FirstPrinciplesReduction | CrossDomainIsomorphismMapping | LeverageSourceIdentification | DeliberatePracticeWithFeedback | FailureModeCataloging | QuestionQualityCritique",
    "rule": "string",
    "applicableWhen": "predicateExpression",
    "evidenceEpisodeIds": ["episodeId"],
    "ruleConfidence": 0.0,
    "antipatterns": ["string"],
    "createdAtRound": 0,
    "verifierEnsembleSignatures": {
      "FactualSoundnessChecker": "pass | abstain",
      "LogicalValidityAuditor": "pass | abstain",
      "RobustnessAdversary": "pass | abstain"
    }
  }]
}
```

写入需满足 INV-1。

### 6.2 `DomainKnowledgeMemory`

```jsonc
{
  "domainId": "string",
  "ontology": {
    "nodes": [{"id": "string", "term": "string", "definition": "string", "sourceCitation": "string"}],
    "edges": [{"sourceId": "string", "targetId": "string", "relation": "string"}],
    "axioms": ["string"]
  },
  "toolReliabilityHistory": {
    "toolId": {"successCount": 0, "failureCount": 0, "latencyP95Ms": 0}
  },
  "cataloguedFailureModes": [{
    "pattern": "string",
    "signature": "string",
    "remedy": "string",
    "occurrences": 0
  }],
  "benchmarkState": {
    "currentPassRate": 0.0,
    "domainExpertThreshold": 0.0,
    "roundsSoFar": 0
  }
}
```

### 6.3 `CrossDomainAnalogyIndex`

```jsonc
{
  "analogies": [{
    "analogyId": "uuid",
    "sourceDomainId": "string",
    "targetDomainId": "string",
    "mapping": [{"sourceConcept": "string", "targetConcept": "string", "homomorphism": "bijective | partial"}],
    "analogyConfidence": 0.0,
    "counterEvidence": [{"case": "string", "whyBreaks": "string"}],
    "successfulApplicationEpisodeIds": ["episodeId"],
    "failedApplicationEpisodeIds": ["episodeId"]
  }]
}
```

查询接口：`queryCrossDomainAnalogy(targetDomainId, targetProblem, topK)` → 仅返回 `analogyConfidence ≥ 0.5` 且 `counterEvidence` 非空；并注入强制反证 prompt（规划假设：Token 溢价约 +15%）。

---

## 8. VerifierEnsemble API Specification

### 7.1 Request / Response

**Request**

```jsonc
{
  "verifierRole": "FactualSoundnessChecker | LogicalValidityAuditor | RobustnessAdversary",
  "auditSubject": {
    "subjectType": "ontology | analogy | practiceEpisode | deliverable",
    "subjectPayload": {},
    "domainId": "string"
  },
  "evidencePack": {
    "authorityCorpusChunks": [],
    "reasoningTrace": "string",
    "adversarialEdgeCases": []
  },
  "auditBudget": {"maxTokens": 0, "maxRounds": 0}
}
```

- `FactualSoundnessChecker` 主要消费 `authorityCorpusChunks`
- `LogicalValidityAuditor` 主要消费 `reasoningTrace`
- `RobustnessAdversary` 主要消费 `adversarialEdgeCases`

**Response**

```jsonc
{
  "verifierRole": "string",
  "verdict": "pass | veto | abstain",
  "verdictConfidence": 0.0,
  "supportingEvidence": [{"claim": "string", "citationOrTrace": "string"}],
  "refutingEvidence": [{"claim": "string", "citationOrCase": "string"}],
  "actionableFeedbackForRetry": "string",
  "tokensUsed": 0
}
```

### 7.2 Role Mandates（摘要）


| Role                    | 唯一职责             | 禁止越权           |
| ----------------------- | ---------------- | -------------- |
| FactualSoundnessChecker | 事实声明可追溯权威源；无源即可疑 | 不审逻辑链；不构造对抗    |
| LogicalValidityAuditor  | 逐步检验推理有效性        | 不挑战事实；不构造对抗    |
| RobustnessAdversary     | 构造边界/对抗场景试图破坏方案  | 不替代事实稽查；不做形式证明 |


### 7.3 Aggregate Verdict

```
if vetoCount ≥ 2 → REJECT
else if passCount ≥ 2 → ACCEPT
else if vetoCount == 1 && passCount ≥ 1 → PARTIAL_RETRY
else → ABSTAIN_ESCALATE
```

---

## 9. DomainToolRegistry & ToolFallbackGateway

### 8.1 Per-Domain Tool Entry

```jsonc
{
  "domainId": "string",
  "tools": [{
    "toolId": "string",
    "category": "executor | retriever | simulator | validator",
    "inputJsonSchema": {},
    "isPrimary": true,
    "secondaryFallbackForToolId": "string | null",
    "expectedLatencyP95Ms": 0,
    "authentication": "required | optional | none"
  }]
}
```

### 8.2 Fallback Stages（每次调用）

1. Primary tool
2. `RepairPrompt`（1 次，修正参数）
3. Secondary tool（若注册）
4. Degraded LLM-only（`executedInDegradedMode: true`）
5. User escalation

### 8.3 Trigger → Action（摘录）


| Trigger         | Action                                                  |
| --------------- | ------------------------------------------------------- |
| Schema 校验失败     | RepairPrompt；`toolReliabilityHistory.failureCount += 1` |
| 超时 `> 2 × p95`  | 切换 secondary                                            |
| 5xx / panic     | 切换 secondary                                            |
| 召回置信度 < 0.4     | 扩大 `topK` / BroaderSearch                               |
| Verifier 双 veto | 回 `Phase.DeliberatePracticeLoop` 并注入反馈                  |
| 预算 90% 未达阈      | 强制 `Phase.FinalDelivery`，INV-7                          |


---

## 10. DomainProfileRegistry & `Phase.DomainProfiling`

### 9.1 `DomainProfile`

```jsonc
{
  "domainId": "string",
  "domainClass": "FormallyVerifiable | KnowledgeIntensive",
  "oracleBinding": {
    "oracleKind": "benchmark | deliveryCheck | both",
    "benchmarkIds": ["string"],
    "deliveryValidatorCallableId": "string"
  },
  "domainExpertThreshold": 0.0,
  "recommendedToolIds": ["string"],
  "authoritySources": [{"uri": "string", "weight": 0.0}],
  "bootstrapStatus": "registered | probed | unknown"
}
```

### 9.2 Probing Subphase（首次域 +30% Token）

1. Lookup `DomainProfileRegistry`
2. Miss → 运行探测：域类识别、公开 benchmark 候选、`DomainToolRegistry` 过滤、**AuthorityCorpusRetriever** 试召
3. 草稿 `DomainProfile` → 以 `LogicalValidityAuditor` 为主导的快速校验
4. 写回，`bootstrapStatus: probed`
5. 若无 oracle 与工具候选 → `FM.BootstrapInfeasible`

SLA 类量级（待实测）：`FormallyVerifiable` 的 `RoundsToExpertise` 上界约为 `8 × log|DomainTestTaskSet|`；`KnowledgeIntensive` 约为 `24 × log|DomainTestTaskSet|`。

---

## 11. Metrics


| Metric                         | Definition                                                | Collection       |
| ------------------------------ | --------------------------------------------------------- | ---------------- |
| `RoundsToExpertise`            | R^* = \arg\min_n \text{holdout pass} \ge \text{threshold} | Audit / Final    |
| `InformationGainAtRound`       | IG_n：ontology 节点增量与有效类比增量的加权和代理                           | Practice         |
| `ExpertiseConvergenceQuotient` | `passedTasks / tokensConsumed`                            | Final            |
| `UsefulToolCallRatio`          | 被下游实际消费的调用 / 总调用                                          | Practice / Final |
| `TaskPassRate`                 | 通过任务数 / `|DomainTestTaskSet|`                             | Final            |
| `VerifierTruePositiveRate`     | 经 ground truth 认可的 veto / 总 veto                          | Audit            |


---

## 12. Failure Mode → Disposition


| ID                                | Condition              | Disposition                       |
| --------------------------------- | ---------------------- | --------------------------------- |
| FM.BootstrapInfeasible            | 探测无 oracle/tool        | `ABORT` + 上报                      |
| FM.OntologyNodeOverflow           | `nodeCount > 200`      | 聚类压缩 + 重做 OntologyConstruction 一次 |
| FM.AllAnalogiesLowConfidence      | 无合格类比                  | 跳过类比注入                            |
| FM.PracticeNoInformationGain      | plateau 检测触发           | 强制进入 VerifierEnsembleAudit        |
| FM.VerifierEnsembleDoubleVetoLoop | 同 episode ≥3 次双否决      | 标记 dead-end；刷新类比                  |
| FM.BudgetExhausted                | ≥90% 预算                | `deliveredUnderBudgetCeiling`     |
| FM.ToolChainTotalFailure          | 降级链到 escalation        | 用户升级                              |
| FM.CorpusPollution                | 引用追溯反复失败               | 暂停 source；重召                      |
| FM.MisleadingAnalogy              | `failedApplication` 占优 | 将该条 `analogyConfidence → 0`       |
| FM.MethodologySedimentConflict    | 方法论条目互斥                | `VerifierEnsemble` 仲裁；必要时双版本并存    |


---

## 13. Data-Flow Contracts


| Object                 | Producer                    | Consumer                 | Drop-if             |
| ---------------------- | --------------------------- | ------------------------ | ------------------- |
| `DomainProfile`        | DomainProfiling             | Ontology, Audit, Final   | oracle 缺失           |
| `DomainOntology`       | OntologyConstruction        | Analogy, Practice, Audit | `nodeCount` 超限      |
| `CrossDomainAnalogy`   | CrossDomainAnalogyInjection | DeliberatePractice       | `counterEvidence` 空 |
| `PracticeEpisode`      | DeliberatePracticeLoop      | Audit, Writeback         | 无 IG 代理             |
| `VerifierVerdict`      | VerifierEnsembleAudit       | Writeback / Retry        | 无证据                 |
| `MemoryWritebackDelta` | MemoryWriteback             | AgentMemoryStore         | 未过 INV-1            |
| `DomainDeliverable`    | FinalDelivery               | User                     | —                   |


---

## 14. Capability Gates (Agent-Native Milestones)


| Gate                                    | Done-Definition                                                                 |
| --------------------------------------- | ------------------------------------------------------------------------------- |
| Gate.SkeletonFSMWalking                 | Mock registry 上 Phase 全路径可跑；转移日志完整；预算不越界                                        |
| Gate.VerifierEnsembleOnline             | 三角色独立会话；仲裁全覆盖；INV-8 审计通过                                                        |
| Gate.SingleFormallyVerifiableDomainMVP  | 单一参考域端到端 `TaskPassRate` 达阈                                                      |
| Gate.MemoryWritebackClosed              | MemoryWriteback Phase 写回全链路；首条经 `VerifierEnsemble` 通过的 `MethodologyMemory` 条目落库 |
| Gate.CrossDomainAnalogyCompounding      | 第二参考域 `ExpertiseConvergenceQuotient ≥ 1.3×` 第一域                                 |
| Gate.KnowledgeIntensiveDomainSupport    | 引用追溯完整；一参考域达 `KnowledgeIntensive` SLA                                           |
| Gate.FailureModeCoverageComplete        | FM.* 全自动检测+处置；混沌测试 ≥95%                                                         |
| Gate.MultiDomainConcurrencySafe         | ≥3 `domain` 并发；类比索引写无冲突                                                         |
| Gate.SelfImprovementEmpiricallyVerified | 固定域重复 N 次，末次 `RoundsToExpertise` < 首次 × 0.7                                     |


---

## 15. Validation Plan

- **单元**：转移表覆盖；`ResourceBudgetGuard` 边界；`ToolCallSchemaValidator` 四类拒绝；单角色 `VerifierTruePositiveRate`
- **集成**：端到端 mock/real；四类聚合判决可达；`ToolFallbackGateway` 五阶段
- **系统**：多参考域 SLA
- **对抗**：污染语料、逻辑跳跃、对抗输入、误导类比 + 反证机制

---

## 16. Open Risks


| ID  | Risk                          | Mitigation                               |
| --- | ----------------------------- | ---------------------------------------- |
| R-1 | 同模型多 prompt 角色相关性偏高           | Gate.VerifierEnsembleOnline 后测 ρ，必要时异构模型 |
| R-2 | `InformationGainAtRound` 为熵代理 | ε、m 可调；对照实验                              |
| R-3 | `DomainProfileRegistry` 冷启动成本 | Mock + Probing                           |
| R-4 | `MethodologyMemory` 长期偏置      | 定期 prune；冲突双版本                           |
| R-5 | 权威语料时效                        | `validUntil` 元数据                         |
| R-6 | Verifier catch 率先验未校准         | 仪表化回填                                    |


---

## 17. Appendix: Reusable Patterns

### A.1 `VerifierEnsembleSizing`（脱域）

1. 枚举失效模式
2. 相关性聚类 → 正交簇数 K
3. 估计每簇检出率 p
4. 边际覆盖 \Delta P(n) = (1-p)^{n-1} p，拐点 \Delta P(n)/\Delta P(n-1) < 0.4
5. `N = min(n*, K)`；每簇一角色；上下文隔离

### A.2 误差正交三轴

- **FactualSoundness** → `FactualSoundnessChecker`
- **LogicalValidity** → `LogicalValidityAuditor`
- **Robustness** → `RobustnessAdversary`

### A.3 Agent-Native Speed Quadruple

`RoundsToExpertise`、`InformationGainAtRound`、`ExpertiseConvergenceQuotient`、`UsefulToolCallRatio`