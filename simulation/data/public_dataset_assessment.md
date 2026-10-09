# 公开 AM 数据集可访问性与可用性评估

**评估对象**：NIST AM-Bench、America Makes、Materials Data Facility (MDF)、Zenodo/Figshare/Dryad、NIMS MatNavi、Materials Project/OQMD/NOMAD
**评估目的**：回应审稿人"应使用公开 AM 数据集"的要求，明确每个数据集**可用/不可用**及其证据
**评估日期**：2026-09-28
**评估环境**：Windows 工作站，Python 3 (miniconda)，PowerShell 5.1，curl.exe
**评估数据落盘位置**：`D:\20260618断裂模型论文\simulation\data\nist_ambench\`

---

## 0. 结论摘要

### 0.1 必须首先纠正的一处事实前提

任务描述中的假设 —— "AM-Bench 含 LPBF Ti-6Al-4V 三方向（0°/45°/90°）拉伸数据 + AlSi10Mg 一个方向拉伸数据" —— **经核查不成立**。依据是 NIST 官方 AM-Bench 数据集清单（本次逐文件解析，见 §3）：

| 假设 | 实际情况 | 证据 |
|---|---|---|
| AM-Bench 有 Ti-6Al-4V LPBF 拉伸（0°/45°/90°） | **没有**。AM-Bench 的 Ti-6Al-4V 项目（AMB2025-03）是**高周旋转弯曲疲劳**，不含拉伸 | `mds2-3734`，标题 "AMB2025-03 High-Cycle Rotating Bending Fatigue Tests of PBF-L Ti-6Al-4V" |
| AM-Bench 有 AlSi10Mg 拉伸 | **没有任何 AlSi10Mg 数据集**。对 6 个数据集的 NIST PDR 元数据做全文检索，`AlSi10Mg / AlSi10 / aluminum alloy` 零命中 | §3.6 检索脚本与结果 |
| AM-Bench 有 LPBF 合金多取向拉伸 | **有，但是 IN625 和 IN718**，不是 Ti64/AlSi10Mg | `mds2-2588`(IN625 MaTTO)、`mds2-2587`(IN625 MeTT)、`mds2-2760`(IN625 ASTM E8)、`mds2-3735`(IN718) |

**AM-Bench 金属力学测试的真实材料构成：IN625、IN718（另有 Al 合金/AA5182 仅做吸收率与熔池，无拉伸）。**

### 0.2 六个数据集结论一览

| # | 数据集 | 结论 | 一句话理由 |
|---|---|---|---|
| 1 | **NIST AM-Bench** | **部分可用（受限）** | 元数据完全公开可取；但实测拉伸曲线的二进制文件在本网络**无法下载**（§1.3）；且材料为 IN625/IN718，与本文 Ti64/AlSi10Mg 标定集不匹配 |
| 2 | **America Makes** | **不可用** | 会员制，且**仅限美国机构**；数据资源（CORE）为会员专属，Silver 档 $15K/年 |
| 3 | **Materials Data Facility (MDF)** | **可用（但无对口数据）** | 完全开放，下载无需 Globus 账号；但本次未检索到 LPBF 合金多取向拉伸断裂数据 |
| 4 | **Zenodo / Figshare / Dryad** | **可用** | 开放获取，多为 CC0/CC-BY；可作为**文献数据的补充存档**，需逐个按 DOI 核验许可 |
| 5 | **NIMS MatNavi** | **不可用（对本文）** | 免费但需 DICE 账号 + 机构域名注册 + 用途申请；**明令禁止批量下载与网络抓取**，违者停号 |
| 6 | **Materials Project / OQMD / NOMAD** | **不适用** | 均为第一性原理计算的基础物性库（弹性张量、形成焓、带结构等），**不含拉伸/断裂数据** |

### 0.3 对本文最有价值的两项发现

1. **`mds2-3402`（NIST，非 AM-Bench 本体）是本次最重要的发现**：LPBF Ti-6Al-4V + 多种 HIP 处理，**96 根 mini-tensile 试样，2 个取向（h/v），30 条工程应力—应变曲线，15 种热处理条件**。这是与本文标定集材料完全一致的公开数据，价值远高于 AM-Bench 本体。
2. **AM-Bench 的取向覆盖是"绕 Z 轴旋转"而非"相对构建方向 0/45/90"**：AMB2022-04-MaTTO 的 93 条力学测试记录的取向分布为 `Z`(24)、`Z+60°`(24)、`Z+90°`(24)、`Z+45°`(12)、`Z+30°`(9)、`X`(7)。这一点直接影响能否与本文各向异性标定集对比。

### 0.4 诚实声明（不降级宣称）

- **本文档中所有"已下载并解析"的结论，均基于实际下载到本地的文件**（NIST PDR 元数据 JSON、AM-Bench 官方 GitHub 仓库及其 3.0.0 XML 元数据发布包）。
- **本次未能下载任何"实测数值"文件**（应力—应变 CSV、Reduced_Data.zip 等）。原因与证据见 §1.3。**【R13b 更新：`mds2-3402` 的 `Reduced_Data.zip` 已于 2026-09-28 经 S3 直连成功下载并解析出 30 条发布均值，见 §0.5；本条其余部分仍成立。】**因此 §3 中"数据摘要"给出的是**试样级元数据统计（材料/取向/样本量/条件）**，**不是** ef/UTS 数值。凡是未取得的数值，本文档一律不给数字，不做推测。
- 全部实验数据为文献/公共数据库来源，非原创。

### 0.5 R13b 更新（2026-09-28）：mds2-3402 实测数据已成功下载并纳入扩展数据集

§0.4 第 2 条、§1.3、§3.5 与 §7.1 中“未能下载实测数值文件”的记录写在 2026-09-27，当时为真；其中 `mds2-3402` 这一条现已解决。本节给出更新，**原文一律保留不改**，以便审阅者看到结论的变化过程，而不是被静默替换。

- **通道**：`data.nist.gov/od/ds/...` 返回 301，跟随重定向到 `nist-oar-cache.s3.amazonaws.com/prd/fst4/mds2-3402/...` 的 S3 直连即可取得；Cloudflare 前端路径在本环境下仍被限速到约 4 kB/s，不可用。随附 README 在四条通道上取回字节完全相同。
- **校验**：`Reduced_Data.zip` = 63,085,120 字节，sha256 与 NIST 公布值一致，CRC 全部通过。
- **已解析并采用**：15 个热处理条件（HT0–HT14，含 HIP 与固溶/时效）× 2 取向 = 30 条发布 group mean（各 n = 10）；取向映射 **v → 0°、h → 90°**，依据随附报告 IR 8551 的试样轴定义（属推断而非数据文件自带标签，故每行保留来源标签 `v`/`h`，映射可作为单列编辑回退）。30 条发布均值均可自沉积曲线精确重建（19 无条件剔除、11 标注剔除试样），UTS 与发布表差 ≤ 0.07 MPa，一处 0.01 MPa 修正（HT14-v 949.68 → 949.69 MPa，依据 IR 8551 p63 原文 `949.69 ± 5.78`）。
- **已纳入**：作为 30 点 `nist_mds2_3402` 源写入 `extended_literature_dataset.csv`，扩展集由 24 点 / 6 来源变为 **54 点 / 7 来源**。含 mds2 的 LOSO 由 44.07% 变为 **61.35%**（变差，如实报告，旧值并列保留）；mds2 留存折 ε_f 误差 72.09%、UTS 误差 8.34%，失败在延性不在强度。30 点不是 30 个独立样本（15 条件 × 2 取向各 n = 10），且 20/30 超出任务核验带，两条限制均写入论文。
- **仍未取得（保持诚实声明）**：`Tensile Testing\RAW DATA`（575 个原始机器文件，仅覆盖 5/15 条件）与 `Microscopy` 树；AM-Bench `mds2-2760` 的逐点 CSV 与各数据集 `read me` PDF 亦未取得。§4.1–§4.5 对其余数据源（America Makes、MDF、Zenodo/Figshare/Dryad、MatNavi、Materials Project/OQMD/NOMAD）的结论不变。
- **证据文件**：`nist_ambench/mds2_3402_parsed.md`（解析与逐项核验）、`nist_ambench/download_attempts_r13b.md`（通道记录）、`extraction_log.md` §7（提取记录）。

---

## 1. 评估方法与证据链

### 1.1 采用的核查方式

| 手段 | 用途 |
|---|---|
| NIST PDR 官方元数据 API `/od/id/<id>?format=nerdm` | 获取数据集权威清单：DOI、标题、许可、访问级别、**每个文件的路径/字节数/mediaType/sha256/下载 URL** |
| AM-Bench 官方仓库 `github.com/usnistgov/ambench` | 获取官方数据发布包、XML 元数据、XSD 模式、元数据模型 XLSX |
| `web.fetch` / `general_search` | 核对机构访问政策（America Makes、MatNavi、MDF、Materials Project、NOMAD、OQMD） |
| `curl.exe` 实测连通性 | 判定各仓库从本机是否**真的**可下载 |

### 1.2 本次实际获取并落盘的资产

| 路径 | 内容 | 说明 |
|---|---|---|
| `meta\mds2-2587.json` `mds2-2588.json` `mds2-2760.json` `mds2-3402.json` `mds2-3734.json` `mds2-3735.json` | 6 个 NIST 数据集的官方 nerdm 元数据 | **原始文件**，共 1995 条文件记录 |
| `_repo\ambench-main\` | AM-Bench 官方 GitHub 仓库（57 MB） | 含 `data-releases\`、`AMBench2022\MetadataModel\`、XSD |
| `_repo\...\data-releases\3.0.0_2026-05-05_data-release.zip` | 官方数据发布 **版本 3.0.0**（2026-05-05，18.5 MB） | 解包后 **1025 个 XML 元数据记录** |
| `parsed\nist_pdr_file_inventory.csv` | 1995 行文件清单（路径/字节/mediaType/许可/DOI/sha256/URL） | **解析后的 CSV** |
| `parsed\nist_ambench_specimens.csv` | 100 条力学测试记录（试样 ID/条件/取向/应变率/数据 DOI） | **解析后的 CSV** |
| `parsed\nist_ambench_buildparts.csv` | 346 条构件记录（含 materialClass） | **解析后的 CSV** |
| `parsed\ambench_xml_inventory.csv` | 1025 个 XML 记录索引 | **解析后的 CSV** |
| `parsed\xlsx_*.csv`（43 个） | 元数据模型 XLSX 全部 sheet 导出（`parsed\` 合计 47 个文件 = 4 个清单 CSV + 43 个 sheet 导出） | **解析后的 CSV** |
| `_netdiag.txt` | 下载失败的网络诊断原始记录 | 证据留档 |
| `meta\*.py` `meta\fetch_ps.ps1` | 本次使用的全部脚本 | 可复现 |

### 1.3 关键实测：NIST PDR 的"元数据可读、数据文件不可下载"

这是本次评估最重要的技术发现，直接影响结论强度。

**现象**（`curl.exe` 实测，2026-09-28）：

| 探测目标 | 结果 |
|---|---|
| `https://data.nist.gov/od/id/mds2-3402?format=nerdm`（元数据） | **HTTP 200**，正常返回 |
| `https://data.nist.gov/od/ds/mds2-2760/README_AMBench_2022_Tensile_data.txt`（数据文件） | **超时，0 字节** |
| `https://data.nist.gov/od/ds/ark:/88434/mds2-2588/prediction%20template...xlsx` | **超时，0 字节** |
| `https://data.nist.gov/od/ds/mds2-2760/`（目录） | **超时，0 字节** |
| `https://www.nist.gov/ambench`（其他 NIST 主机） | **HTTP 200** |
| `https://nvlpubs.nist.gov/...`（其他 NIST 主机） | **HTTP 200** |
| `https://data.nist.gov/pdr/lps/ark:/88434/mds2-2760`（落地页） | **HTTP 200** |
| `https://data.nist.gov/pdr/bulkdownload/mds2-3402` | **HTTP 200** |

**排除性验证**（说明这不是简单的"网站挂了"或"我的命令写错了"）：

- 换浏览器 User-Agent、加 `Referer`、强制 `--http1.1`、发 `Range: bytes=0-2047` 分片请求 —— **全部仍然 0 字节超时**。
- 纯 HTTP（80 端口）访问同一路径返回 **301**（服务端应答正常），说明不是 DNS/路由完全不通，而是 **`/od/ds/` 这条路径上的 HTTPS 数据传输被阻断**。
- 同主机其他路径（`/od/id/`、`/pdr/lps/`、`/pdr/bulkdownload/`）**全部 200 正常**。

**附带发现**：本机 Windows schannel 存在 `CRYPT_E_NO_REVOCATION_CHECK (0x80092012)` —— 无法访问证书吊销列表，导致 HTTPS 握手失败。已通过 curl 的 `--ssl-no-revoke` 规避。这是本机环境问题，与 NIST 无关，但需记录，因为它同样会影响其他人复现。

**结论**：在本次评估的网络环境下，**NIST PDR 的实测数据文件无法获取**。这不是 NIST 的访问限制（数据本身是 `accessLevel: public`、NIST 开放许可），而是网络可达性问题。**【R13b 更新：该结论已对 `mds2-3402` 一条解除，见 §0.5。】****因此本次不能声称"已下载 NIST 数据"**。

**给用户的实际行动项**：换一个网络环境（如校园网/VPN/境外节点）重试以下命令即可获取数据，无需任何账号：

```powershell
# 小体积、最对口：AM-Bench 2022 IN625 ASTM E8 拉伸，5 条应力-应变曲线，共约 0.9 MB
curl.exe -L --ssl-no-revoke -o mds2-2760.zip "https://data.nist.gov/od/ds/ark:/88434/mds2-2760"
# 或按文件逐个取
# https://data.nist.gov/od/ds/mds2-2760/SpecimenT3_stress_strain.csv
```

---

## 2. 六个数据集汇总评估表

评分口径：**可访问性** = 是否需登录/许可/付费；**格式**；**内容相关性** = 是否含 AM 合金（LPBF/SLM）断裂应变/UTS/取向数据与工艺参数；**许可**；**对本文价值**。

| 数据集 | 可访问性 | 数据格式 | 内容相关性（LPBF 多取向拉伸） | 许可 | 对本文价值 | 结论 |
|---|---|---|---|---|---|---|
| **NIST AM-Bench** | 元数据：公开、免注册、实测可取；**数据文件：本网络阻断**（非许可问题，`accessLevel=public`） | 元数据 XML/XSD；数据 ZIP（内含 CSV/PDF/图像） | **中**：有 IN625（5 取向）、IN718 拉伸；**无 Ti64 拉伸、无 AlSi10Mg** | NIST 开放许可（`nist.gov/open/license`），允许学术使用与再分发 | 可作**跨材料验证**（IN625）；不能替代 Ti64 标定 | **部分可用（受限）** |
| **America Makes** | **不可用**：会员制，**仅限美国机构**；数据资源会员专属 | 不适用 | 不适用 | 会员协议 | 无 | **不可用** |
| **MDF** | **公开**：浏览/下载**无需 Globus 账号**；发布才需注册 | Foundry-ML（ML-ready）、MDF Forge；HDF5/JSON 等 | **低**：未见 LPBF 多取向拉伸断裂数据 | 依数据集而定（多为 CC 系） | 平台可用，**但无对口数据** | **可用（无对口内容）** |
| **Zenodo / Figshare / Dryad** | **公开**，可匿名下载（浏览器） | 任意（多为 CSV/XLSX/ZIP） | **取决于具体条目**：文献配套数据常有单取向拉伸 | 通常 CC0/CC-BY，需逐条核验 | **可作文献数据存档与 DOI 引用** | **可用** |
| **NIMS MatNavi** | **需注册**：DICE 账号 + 机构域名注册 + 用途申请；**禁止批量下载/抓取** | 网页/专用客户端；非批量导出 | 金属库偏密度/弹性常数/蠕变，**非 AM 断裂** | NIMS 服务条款（禁止抓取） | 无 | **不可用（对本文）** |
| **Materials Project / OQMD / NOMAD** | MP：需 API key；OQMD：开放 REST + 全库可下；NOMAD：**已发布数据无需账号** | JSON / Archive / API | **无**：DFT 基础物性（弹性张量、形成焓、带结构） | MP/NOMAD 有各自条款；OQMD 开放 | 可提供**弹性常数等输入参数**，但**不含断裂数据** | **不适用** |

---

## 3. NIST AM-Bench 深度评估

### 3.1 项目入口与权威清单

| 资源 | URL | 本次可达性 |
|---|---|---|
| AM-Bench 项目主页 | `https://www.nist.gov/ambench` | ✅ 200 |
| AM-Bench CDCS 数据门户 | `https://ambench2022.nist.gov/` | ✅ 200 |
| **AM-Bench 官方数据仓库** | `https://github.com/usnistgov/ambench` | ✅ **200，已完整下载** |
| 当前数据发布版本 | `data-releases/current-release.txt` → **3.0.0，2026-05-05** | ✅ |
| 官方元数据 API 形式 | `https://data.nist.gov/od/id/<mds2-id>?format=nerdm` | ✅ 200 |
| 官方数据文件形式 | `https://data.nist.gov/od/ds/<mds2-id>/<filename>` | ❌ **阻断（0 字节）** |

### 3.2 AM-Bench 具体数据集清单（本次逐文件解析所得）

标题、DOI、体积均取自 NIST PDR 官方元数据 JSON，非推测。

| DOI | 标题 | 文件数 | 总字节 | 与本文关系 |
|---|---|---|---|---|
| `10.18434/mds2-2588` | AM Bench 2022 challenge Macroscale Tensile Tests at Different Orientations (CHAL-AMB2022-04-MaTTO) | 4 | **8.26 GB** | **最对口**（多取向宏观拉伸，但材料 IN625） |
| `10.18434/mds2-2587` | AM Bench 2022 challenge problem Subcontinuum Mesoscale Tensile Test (CHAL-AMB2022-04-MeTT) | 4 | **6.89 GB** | 介观拉伸，含应力—应变 |
| `10.18434/mds2-2760` | AM Bench 2022 ASTM E8 Macroscale Tension at Different Strain Rates on As-built IN625 | 7 | **0.0009 GB** | **体积最小、最易获取**；5 条完整应力—应变曲线 |
| `10.18434/mds2-3402` | Dataset of Additive Manufacturing Laser Powder Bed Fusion **Ti-6Al-4V** Subjected to Various Novel Hot Isostatic Pressing (HIP) Treatments | 1536 | **114.89 GB** | **材料完全对口**（非 AM-Bench 本体） |
| `10.18434/mds2-3734` | AMB2025-03 High-Cycle Rotating Bending **Fatigue** Tests of PBF-L **Ti-6Al-4V** | 439 | **222.18 GB** | 材料对口但为**疲劳**，非拉伸 |
| `10.18434/mds2-3735` | AMB2025-02 Macroscale Quasi-Static Tensile Tests of PBF-L **IN718** | 5 | 0.07 GB | IN718 准静态拉伸 |

**六个数据集合计 1995 个文件记录，许可全部为 `https://www.nist.gov/open/license`，访问级别全部 `public`。**

### 3.3 具体文件级清单（关键数据集）

**`mds2-2760` — IN625 ASTM E8 拉伸（体积最小，最易获取）**

| 文件 | 大小 |
|---|---|
| `SpecimenT3_stress_strain.csv` | 21.1 KB |
| `SpecimenT4_stress_strain.csv` | 20.2 KB |
| `SpecimenT5_stress_strain.csv` | 22.9 KB |
| `SpecimenT6_stress_strain.csv` | 21.0 KB |
| `SpecimenT7_stress_strain.csv` | 22.5 KB |
| `SUMMARY_AMBench_2022_Tensile_data.pdf` | 802.3 KB |
| `README_AMBench_2022_Tensile_data.txt` | 3.2 KB |

→ **单个 CSV 约 21 KB，即"逐点"的应力—应变曲线（而非仅汇总 UTS/ef）**。这是全 AM-Bench 中**最适合直接纳入标定/验证**的数据形态，只是材料是 IN625。

**`mds2-2588` — IN625 MaTTO（多取向宏观拉伸）**

| 文件 | 大小 |
|---|---|
| `Macroscale_CHAL-AMB2022-04-MaTTO.zip` | 8395.08 MB |
| `ANSWERS Macroscale_CHAL-AMB2022-04-MaTTO.zip` | 59.91 MB（**含 XY 扫描 0° 的真应力—应变曲线**） |
| `read me_CHAL-AMB2022-04-MaTTO.pdf` | 2.01 MB |
| `prediction template_CHAL-AMB2022-04-MaTTO.xlsx` | 0.01 MB |

**`mds2-3735` — IN718 准静态拉伸（体积最小）**

| 文件 | 大小 |
|---|---|
| `answers-raw tensile data.zip` | 1.06 MB |
| `calibration data.zip` | 65.33 MB |
| `readme.pdf` | 0.19 MB |
| `AMB2025-02 prediction answers.xlsx` | 0.01 MB |
| `prediction submission template.xlsx` | 0.01 MB |

### 3.4 AMB2022-04 试样级取向构成（从官方 XML 元数据解析）

从 3.0.0 数据发布包的 **1025 个 XML 记录**中解析出 **100 条力学测试记录**：

| challengeID | 描述 | 记录数 |
|---|---|---|
| `CHAL-AMB2022-04-MaTTO` | Macroscale Tensile Tests at Different Orientations | **93** |
| `CHAL-AMB2022-04-MeTT` | Subcontinuum Mesoscale Tensile Test | 1 |
| （空） | 同上类描述 | 6 |

**取向（measurementDirection）分布：**

| 取向 | 记录数 |
|---|---|
| `Z` | 24 |
| `Z+60°` | 24 |
| `Z+90°` | 24 |
| `Z+45°` | 12 |
| `Z+30°` | 9 |
| `X` | 7 |
| **合计** | **100** |

**⚠️ 重要解读**：AM-Bench 的取向定义是**绕 Z 轴旋转角**，且区分 `XY` 扫描策略与 `X` 扫描策略。它**不等于**任务描述中的"相对构建方向的 0°/45°/90°"。跨数据集对标定时必须做取向定义的换算，**不能直接假定两者等价**。

**构件级材料分布**（从 346 条 `AMBuildPart` 记录解析）：

| materialClass | 记录数 |
|---|---|
| `metal; alloy; Ni alloy; Ni-based super alloy; IN625` | 37 |
| `metal; alloy; Ni alloy; Ni-based super alloy; IN718` | 26 |
| （其他/未标注） | 283 |

### 3.5 `mds2-3402`：与本文材料完全对口的 Ti-6Al-4V 数据集（重点）

**标题**：Dataset of Additive Manufacturing Laser Powder Bed Fusion Ti-6Al-4V Subjected to Various Novel Hot Isostatic Pressing (HIP) Treatments
**DOI**：`10.18434/mds2-3402`　**文件数**：1536　**总字节**：114.89 GB

目录构成：

| 目录 | 文件数 |
|---|---|
| `Tensile Testing` | **607**（约 101.7 MB） |
| `X-ray Computed Tomography` | 94（约 100+ GB，tiff 切片栈） |
| `Microscopy` | 821（TEM / EBSD / 金相） |
| `Metallographic preparation` | 10 |
| `Machining Diagrams` | 2 |
| `Rotating Bending Fatigue Testing` | 1 |
| 根目录 | 1（`README.docx`） |

**`Tensile Testing` 内部结构（本次逐文件统计）：**

| 项 | 数量 | 说明 |
|---|---|---|
| `RAW DATA/<条件>-<取向><编号>/specimen.dat` | **96** | 每试样 1 个 `.dat`（182–245 KB），配套 `.log/.mps/.mpp/.prm/.plt` |
| `Engineering stress strain curve images/*.pdf` | **30** | 工程应力—应变曲线图 |
| `Reduced_Data.zip` | 1（60.2 MB） | **规约（processed）数据归档** |
| `mini-tensile specimen dimensions.xlsx` | 1 | 试样尺寸 |
| **RAW DATA 文件合计** | **575** | 96 试样 × 6 文件 − 1 |

**试样级构成（96 根，5 个 HIP 条件 × 2 个取向）：**

| 条件 | h（水平） | v（垂直） | 小计 |
|---|---|---|---|
| HT0 | 10 | 10 | 20 |
| HT1 | 10 | 10 | 20 |
| HT10 | 10 | 10 | 20 |
| HT11 | 10 | 10 | 20 |
| HT12 | 10 | 6 | 16 |
| **合计** | **50** | **46** | **96** |

**取向覆盖度**：只有 **2 个取向（h / v，即垂直/平行于构建方向）**，对应 0°/90°。**无 45° 取向**。因此它可以支撑"两方向各向异性"的标定/验证，**不能**支撑三方向（0/45/90）各向异性验证。

**热处理条件覆盖度**（从文件索引统计）：完整研究覆盖 **ht_0 … ht_14（15 个 HIP 条件）**——30 条应力—应变曲线 = 15 条件 × 2 取向，XCT 覆盖全部 15 个条件，TEM 覆盖 9 个条件；但 **RAW DATA 的 `specimen.dat` 只覆盖 5 个条件**（HT0/HT1/HT10/HT11/HT12）。

**⚠️ 未取得**：`ef`、`UTS`、屈服强度的**数值**。这些数值应在 `Reduced_Data.zip` 与 `SUMMARY...pdf` 内，但受 §1.3 网络阻断影响本次未能下载，**故本报告不给任何 ef/UTS 数字**。**【R13b 更新：已取得，见 §0.5 与 `nist_ambench/mds2_3402_parsed.md`；本节以下内容为当时的元数据结论。】**给用户的重试命令：

```powershell
curl.exe -L --ssl-no-revoke -o Reduced_Data.zip "https://data.nist.gov/od/ds/mds2-3402/Tensile%20Testing/Reduced_Data.zip"
```

### 3.6 AlSi10Mg 存在性检索（否定性证据）

对下载到本地的 6 个 NIST PDR 元数据 JSON 全文做正则检索：

```
AlSi10Mg | AlSi10 | aluminium alloy | aluminum alloy
```

**结果：零命中。** 结论：NIST PDR 的这 6 个 AM 数据集中**不含 AlSi10Mg 数据**。AM-Bench 中涉及铝合金的只有 `A-AMB2022-01`（裸板激光吸收率与熔池动力学，用 AA5182/Al 合金），**无力学拉伸**。

### 3.7 数据格式

| 层级 | 格式 | 说明 |
|---|---|---|
| 元数据（可取） | **XML**（XSD 约束）+ nerdm **JSON** | AM-Bench 3.0.0 发布包 = 1025 个 XML；PDR 元数据 = JSON |
| 元数据模型 | **XLSX** | `AM-Bench 2022 Measurements.xlsx`（39 个 sheet）、`Samples.xlsx`、`Contributors.xlsx` |
| 实测数据（本次未取） | **ZIP → CSV / PDF / XLSX / TIF / VTK / DAT** | 例：`SpecimenT3_stress_strain.csv`（逐点曲线）；`specimen.dat`（原始机台数据）；`Reduced_Data.zip`（规约数据） |
| 图像/体数据 | TIF 切片栈、PNG/JPEG、MP4 | XCT、TEM、EBSD |

**关于"是否含应力—应变曲线"**：**含**。`mds2-2760` 有 5 个逐点 CSV；`mds2-3402` 有 30 个曲线 PDF + 96 个原始 `.dat` + `Reduced_Data.zip`；`mds2-2588/2587` 的 ANSWERS/ZIP 内含真应力—应变曲线。**但本次均未能下载。**【R13b 更新：`mds2-3402` 的 `Reduced_Data.zip` 已下载，见 §0.5；其余仍未下载。】**

---

## 4. 其余数据集逐项评估

### 4.1 America Makes —— **不可用**

来自其官方会员页（`https://www.americamakes.us/membership/`，访问于 2026-09-28）的直接证据：

- 原文：**"Membership is limited to U.S.-based organizations."**
- 数据资源属会员专属：会员权益表列出 **"Access to member-only data and resources (Core)"**，CORE 为"所有技术项目交付物（2012 年至今）"的存放处。
- 会费：Silver **$15K**、Gold $50K、Platinum $200K。会员构成：236 Silver / 44 Gold / 17 Platinum。

**结论**：即便愿意付费，会员资格**限于美国机构**，本文作者（中国境内省属事业单位）**不具备申请资格**。除公开的项目简介与案例研究外，**无法获得项目数据**。**不可用**。

### 4.2 Materials Data Facility (MDF) —— **可用（但无对口数据）**

来自 MDF 官方指南（`https://www.materialsdatafacility.org/guides`）：

- 原文：**"You don't need a Globus account to access and download data!"** —— 下载**无需**任何账号。
- 发布数据才需要免费 Globus 账号并加入 MDF Globus 组。
- 提供两条程序化路径：**Foundry-ML**（结构化、可直接载入 DataFrame 的 ML-ready 数据集）与 **MDF Forge**（通用程序化加载）。
- 站点本次实测可达（HTTP 200）。

**结论**：**平台可用、许可友好**。但本次检索未发现 LPBF 合金**多取向拉伸断裂**数据。**对本研究：无对口数据可直接纳入标定集。** 可作为未来发表本文数据时的**发布渠道**备选。

### 4.3 Zenodo / Figshare / Dryad —— **可用**

- 三者均为开放获取型通用仓储，支持匿名浏览与下载，通常以 **CC0 / CC-BY** 授权，可自由用于学术论文（CC-BY 需署名）。
- 本次实测提示：`curl`（非浏览器 UA/TLS 指纹）访问 Zenodo 返回 **403**，属**机器人防护**而非访问限制；浏览器或 `web.fetch` 方式正常。**不要据此判定 Zenodo 不可用。**
- 价值定位：AM 领域大量论文把**配套拉伸数据**（单取向、含应力—应变曲线与 UTS/ef 汇总）以 CC-BY 存于 Zenodo/Figshare。**逐条按 DOI 检索并核验许可后可用**。

**结论**：**可用**，是本文补充"公开数据"最现实的来源之一；但需**逐条核验**许可与是否含断裂应变，不能笼统引用。

### 4.4 NIMS MatNavi —— **不可用（对本文）**

来自 NIMS 官方门户（`https://mits.nims.go.jp/`）：

- 使用**免费**，但必须完成**三步**：① DICE 账号用户注册（**原则上须隶属某机构**，且须用**机构邮箱**）；② **邮箱域名注册**（域名未登记时须由机构域名管理员申请）；③ **MatNavi 用途申请**（且需电脑浏览器，不支持手机）。
- 明确禁令原文：**"Mass downloading of data is prohibited"**、**"Web scraping of data is prohibited!"**；"The acquisition of large amounts of data, whether by manual or mechanical means, is prohibited by the MatNavi Service Terms of Use"；违者**停号**。
- 内容定位：金属材料库偏**密度、弹性常数、蠕变特性**等，**非 AM 断裂数据**。

**结论**：**对本文不可用**——（a）访问需注册与机构域名审核，非开放获取；（b）用途条款**明确禁止**批量下载与抓取，无法用于构建标定/验证集；（c）内容为传统金属物性，非 AM 断裂。**诚实声明：未尝试绕过其条款。**

（补充：NIMS 另有相对开放的 **MDR 数据仓储**（`mdr.nims.go.jp`，本次实测可达 200），与 MatNavi 不是同一系统，若需要可单独评估。）

### 4.5 Materials Project / OQMD / NOMAD —— **不适用**

| 平台 | 访问方式（已核验） | 内容 |
|---|---|---|
| **Materials Project** | **需 API key**（登录后在 profile dashboard 获取），`pip install mp_api` + `MPRester` | 端点全为**计算物性**：`/materials/elasticity`（弹性张量）、`/materials/thermo`（热力学）、`/materials/dielectric`、`/materials/phonon`、`/materials/electronic_structure` 等 |
| **OQMD** | **开放 REST API**（`oqmd.org`），约 70 万种材料；整库亦可下载 | DFT 热力学与结构性质 |
| **NOMAD** | **"You can access all published data without an account"**；API 对已发布数据**无需授权**；CC 系许可 | 计算材料科学原始数据 + Archive |

**结论：三者均不适用。** 它们是**第一性原理计算的体材料基础物性库**，**不含**拉伸曲线、断裂应变、UTS 或 AM 工艺参数。可作为**弹性常数等本构输入参数的独立来源**（非实验验证数据），但**不能**用于断裂模型的标定或验证。若论文中将其列为"公开数据来源"，会与审稿人要求的"实验数据验证"错配。

---

## 5. 价值判定：可直接纳入 / 仅参考 / 不可用

| 判定 | 数据集 / 具体资源 | 理由与前置条件 |
|---|---|---|
| **✅ 可直接纳入标定/验证集**（材料对口） | **`mds2-3402`**（LPBF Ti-6Al-4V + HIP） | 96 根 mini-tensile、2 取向、30 条工程应力—应变曲线、5 个条件有原始 `.dat`。**前置条件**：① 换网络下载 `Reduced_Data.zip`（60 MB，已验证 URL）；② 仅覆盖 h/v 两方向，**45° 仍需文献补充**；③ 试样为 mini-tensile（非标准 ASTM E8），标定时须说明尺寸效应 |
| **✅ 可直接纳入验证集**（跨材料） | **`mds2-2760`**（IN625，5 条逐点曲线）；**`mds2-2588`**（IN625，5 取向）；**`mds2-3735`**（IN718） | 有完整应力—应变曲线与多取向。**前置条件**：换网络下载。价值在**验证模型的材料无关性**，不能替代 Ti64/AlSi10Mg 标定 |
| **🔶 仅作参考** | Zenodo / Figshare / Dryad 中的文献配套数据；MDF；NIMS MDR；OQMD 弹性常数 | 需逐条核验许可、材料、取向、是否含断裂应变；可用于补充单点或作为参数来源，不宜作为主验证集 |
| **❌ 不可用** | **America Makes**（限美国机构、会员专属）；**NIMS MatNavi**（注册+域名审核+禁止批量下载/抓取） | 受资格或条款限制，非技术问题。**明确声明无法访问的原因，不降级宣称** |
| **❌ 不适用** | Materials Project / OQMD / NOMAD | 无拉伸/断裂数据，与本文验证目标错配 |
| **❌ 本次未取得** | NIST PDR 全部实测数据文件（含 `mds2-3402`、`mds2-2760`） | 本机网络对 `data.nist.gov/od/ds/` 阻断（§1.3）。**元数据与清单已完整取得**，实测数值未取得 |

---

## 6. 对审稿人意见的建议回复口径

审稿人要求"使用公开 AM 数据集"。建议分三层回应，避免过度承诺：

1. **已采纳并新增验证**：明确采用 **NIST AM-Bench 2022（IN625，MaTTO/MeTT/ASTM E8）** 作为**跨材料、多取向**的公开验证数据；采用 **NIST `mds2-3402`（LPBF Ti-6Al-4V）** 作为**同材料两取向**的公开标定/验证数据。给出 DOI。
2. **如实体现在覆盖度上的差异**：AM-Bench 本体**不含** Ti-6Al-4V 拉伸，也**不含** AlSi10Mg；其多取向定义是"绕 Z 轴旋转"，与本文的"相对构建方向 0/45/90"需换算。**不要**在论文中暗示 AM-Bench 提供了 Ti64/AlSi10Mg 多取向拉伸数据——这与官方清单不符，会被二次审稿直接推翻。
3. **声明受限项**：America Makes 限美国机构会员、NIMS MatNavi 明令禁止批量下载 —— 属**访问资格与使用条款**限制，非研究疏漏，应在"数据可用性声明"中写明。

**复习用脚本**：`meta\build_inventory.py` 可重新生成全部清单 CSV；`meta\fetch_ps.ps1` 可在网络允许时批量抓取 NIST PDR 文件。

---

## 7. 附录

### 7.1 未完成事项（诚实记录）

| 事项 | 状态 | 原因 |
|---|---|---|
| 下载 `mds2-3402` 的 `Reduced_Data.zip` 并解析 ef/UTS | **已完成（2026-09-28）** | 经 S3 直连取得，见 §0.5 与 `nist_ambench/mds2_3402_parsed.md` |
| 下载 `mds2-2760` 的 5 个 `SpecimenT*_stress_strain.csv` | **未完成** | 同上 |
| 下载 AM-Bench 各数据集的 `read me` PDF | **未完成** | 同上（PDF 亦在 `/od/ds/`） |
| 遍历 Zenodo/Figshare 找出所有符合条件的 LPBF 拉伸数据集 | **未完成** | 需按关键词逐条检索与核验，超出本轮范围；§4.3 已给出方法与定位 |
| 核实 Materials Project 的具体许可条款文本 | **部分完成** | API key 要求已由官方文档确认；许可条款页 `docs.materialsproject.org/downloading-data/terms-of-use` 本次抓取失败，未据此下结论 |

### 7.2 复现步骤

```
1. 关闭证书吊销检查（本机环境问题，见 §1.3）：
   curl.exe --ssl-no-revoke ...
2. 取元数据（任何网络均可）：
   curl.exe -L --ssl-no-revoke -o mds2-3402.json "https://data.nist.gov/od/id/mds2-3402?format=nerdm"
3. 取官方仓库（任何网络均可）：
   curl.exe -L --ssl-no-revoke -o ambench.zip "https://codeload.github.com/usnistgov/ambench/zip/refs/heads/main"
4. 解析清单与试样构成：
   python meta\build_inventory.py
5. 取实测数据（需换网络）：
   curl.exe -L --ssl-no-revoke -o Reduced_Data.zip "https://data.nist.gov/od/ds/mds2-3402/Tensile%20Testing/Reduced_Data.zip"
   curl.exe -L --ssl-no-revoke -o SpecimenT3.csv "https://data.nist.gov/od/ds/mds2-2760/SpecimenT3_stress_strain.csv"
```

### 7.3 关键 URL 清单

| 资源 | URL |
|---|---|
| AM-Bench 项目主页 | https://www.nist.gov/ambench |
| AM-Bench 官方仓库 | https://github.com/usnistgov/ambench |
| AM-Bench CDCS 门户 | https://ambench2022.nist.gov/ |
| NIST PDR 元数据 API | https://data.nist.gov/od/id/`<mds2-id>`?format=nerdm |
| NIST PDR 数据文件 | https://data.nist.gov/od/ds/`<mds2-id>`/`<filename>` |
| NIST PDR 批量下载说明 | https://data.nist.gov/pdr/bulkdownload/`<mds2-id>` |
| NIST 开放许可 | https://www.nist.gov/open/license |
| America Makes 会员 | https://www.americamakes.us/membership/ |
| MDF 指南 | https://www.materialsdatafacility.org/guides |
| NIMS MatNavi | https://mits.nims.go.jp/ |
| NIMS MDR 仓储 | https://mdr.nims.go.jp/ |
| NOMAD | https://nomad-lab.eu/ |
| OQMD REST API | https://static.oqmd.org/static/docs/restful.html |
| Materials Project API 入门 | https://docs.materialsproject.org/downloading-data/using-the-api/getting-started |
| ORNL Peregrine（L-PBF 316L，6299 只拉伸试样，另见 OSTI 2001425） | https://www.osti.gov/biblio/2001425 |

### 7.4 本次解析产出的清单文件

| 文件 | 行数/规模 | 内容 |
|---|---|---|
| `parsed\nist_pdr_file_inventory.csv` | 1995 行 | 6 个数据集的全部文件：路径、字节、mediaType、许可、DOI、sha256、下载 URL |
| `parsed\nist_ambench_specimens.csv` | 100 行 | 力学测试记录：试样 ID、challengeID、试样条件、测量方向、平均应变率、数据 DOI |
| `parsed\nist_ambench_buildparts.csv` | 346 行 | 构件记录：名称、materialClass、条件、用途 |
| `parsed\ambench_xml_inventory.csv` | 1025 行 | 3.0.0 XML 元数据记录索引 |
| `parsed\xlsx_*.csv` | 43 个 | 官方元数据模型 XLSX 的各 sheet |

### 7.5 交付前校验记录（verifier-hub CLI 实际执行）

| 检查 | 命令 | 结果 |
|---|---|---|
| 报告格式 | `verifier rubric check-file-format public_dataset_assessment.md --expected-ext .md` | ✅ `passed: true`，`text size=31019` |
| 六个数据集与全部结论出现 | `verifier text must-contain --file ... --terms "AM-Bench" "America Makes" "Materials Data Facility" "Zenodo" "NIMS MatNavi" "Materials Project" "OQMD" "NOMAD" "AlSi10Mg" "mds2-3402" "mds2-2760" "mds2-2588" "mds2-3735" "mds2-2587" "mds2-3734" "CRYPT_E_NO_REVOCATION_CHECK" "不可用" "不适用" "仅作参考" "可直接纳入" --mode all` | ✅ `passed: true`，**matched 20/20** |
| 清单 CSV 格式 | `verifier rubric check-file-format parsed\nist_pdr_file_inventory.csv --expected-ext .csv` | ✅ `passed: true`，`size=919980` |
| 试样 CSV 格式 | `verifier rubric check-file-format parsed\nist_ambench_specimens.csv --expected-ext .csv` | ✅ `passed: true`，`size=20338` |
| 占位符审计 | `verifier text placeholder-audit --file public_dataset_assessment.md` | ⚠️ 报 5 处 `ellipsis`，**已逐条人工复核**：全部是路径/命令的**有意缩写**（`_repo\...\`、`https://nvlpubs.nist.gov/...`、`SUMMARY...pdf`、`curl.exe --ssl-no-revoke ...`、URL 中间省略），**非未完成内容**。保留以维持可读性 |

自查脚本 `meta\selfcheck.py` 另行交叉核对：PDR 文件清单 **1995** 行、数据集 **6** 个、许可唯一且为 `https://www.nist.gov/open/license`、访问级别全部 `public`；力学测试记录 **100** 行；`mds2-3402` 文件 **1536**、`.dat` 试样 **96**、曲线 PDF **30**。
（该次自查同时发现报告初稿把 `parsed\` 下 43 个 sheet 导出误写为 47，已修正 —— 47 是 `parsed\` 的文件总数 = 4 个清单 CSV + 43 个 sheet 导出。）

---

*本报告所有体积、数量、URL、许可均来自本次实际下载/抓取的文件与官方页面；未能取得的数值已明确标注为"未取得"，未做任何推测填充。*
