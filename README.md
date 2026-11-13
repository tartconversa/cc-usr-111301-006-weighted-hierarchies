# 多归属组织闭包与守恒归集桥表

本项目保存虚构算力设施的仓库源表、湖文件与Python读取工具。`hierarchy_reader`只加载输入并显示记录数；`bridge_builder`在冻结版本上执行桥表展开与归集。所有设备、组织和金额均为合成，不涉及真实付款。

## 字段与单位

边从child指向parent，weight为有理数字符串；节点全在nodes中。charges为整数分，目标集合使用普通闭包或first-hit语义必须显式区分。

material-index.json记录全部输入内容哈希；fixtures保存原始资料与配置，计算输出可放.workspace-output，原始资料不应被覆盖。

## 运行已有读取器

Python 3.11以上，标准库即可，不需安装第三方依赖。

    python -m src.hierarchy_reader fixtures/nodes.json
    python -m compileall -q src tests
    python -m unittest discover -s tests -v

## 构建层级归集桥表

`src/bridge_builder.py`在冻结组织版本上执行多父DAG展开、闭包/frontier桥与逐事实冻结归集：

    python -m src.bridge_builder \
      --nodes fixtures/nodes.json --edges fixtures/edges.csv \
      --charges fixtures/charges.csv --hierarchy fixtures/hierarchy.json \
      --output .workspace-output/bridge

输出包含`closure_bridges.json`（每个起点的全部闭包单元格、路径数与压缩证据）、
`frontier_bridges.json`（每个起点在每个声明目标集合下的first-hit分桶与`UNASSIGNED`）、
`facts_ledger.json`（逐事实舍入明细、冻结摘要与按目标集合汇总）、`report.json`与`manifest.json`。

关键语义：

* 闭包桥是全部有向路径权重乘积之和；路径条数单独保留为证据，事实绝不按路径条数复制。
* roots口径是`sum(weight(source, root))`，与目标层权重分开，不同祖先层次不混加。
* 闭包直接归集要求目标集合无祖先/后代交叠且权重和为1；否则拒绝，改用first-hit frontier。
* frontier归集沿每条向根路径遇到第一个目标即停止，未命中流入`UNASSIGNED`，每起点总权重恒为1。
* 每笔事实按最大余数法单独舍入（并列按目标键升序，`UNASSIGNED`列末），负费用先分配绝对值再恢复符号，
  冻结后才按桶做整数分汇总；汇总会校验冻结摘要与守恒。

## 固定容器

    docker build -t weighted-hierarchies:local .
    docker run --rm weighted-hierarchies:local

容器镜像以完整digest固定，默认仅检查输入契约；实现新增分析功能所需依赖由开发者明确登记。
