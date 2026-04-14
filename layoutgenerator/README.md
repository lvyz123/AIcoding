# Layout Generator

一个简单可维护的芯片版图仿真生成工具，可按参数批量输出 `GDSII (.gds)` 或 `OASIS (.oas)` 文件。

## 1. 功能

- 指定版图尺寸：`--width --height`（单位 um）
- 指定层数：`--layers`
- 指定生成个数：`--count`
- 指定输出格式：`--format gds|oas`
- 支持 `config.json` 输入模式
- 支持简化密度参数：`--density low|medium|high`
- 批量输出文件并生成 JSON 报告

## 2. 安装

```bash
pip install -r requirements.txt
```

## 3. 快速开始（命令行参数）

```bash
python layout_generator.py --width 200 --height 150 --layers 6 --count 5 --format gds --output-dir out
```

## 4. config.json 模式

```bash
python layout_generator.py --config config.json
```

说明：命令行参数优先级高于 `config.json`，可用于临时覆盖。

例如：

```bash
python layout_generator.py --config config.json --count 3 --format gds
```

## 5. 密度简化参数

```bash
python layout_generator.py --width 550 --height 550 --layers 4 --format oas --density high --output-dir out_oas
```

默认映射（每层图形数区间）：

- `low`: 4-10
- `medium`: 8-20
- `high`: 40-80

如果同时设置了 `--min-shapes/--max-shapes`，则以显式数值为准。

## 6. 常用参数

- `--base-name`：文件名前缀（默认 `layout`）
- `--min-shapes`：每层最少图形数量
- `--max-shapes`：每层最多图形数量
- `--margin`：边界留白（默认 `5um`）
- `--seed`：随机种子，便于复现

## 7. 说明

- 当前脚本使用 `gdstk` 进行真实版图写出。
- 如果未安装 `gdstk`，脚本会给出安装提示。
- 生成内容用于仿真/测试数据构造，不代表工艺规则合法性（DRC/LVS 需另行校验）。
