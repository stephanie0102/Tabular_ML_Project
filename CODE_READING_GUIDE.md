# 📚 代码阅读指南 - Tabular ML Project

## 🎯 项目核心概念

**Research Question**: Does a lightweight gradient boosting model (LightGBM) outperform a pretrained transformer-based baseline (TabPFN 2.5) across multiple heterogeneous tabular datasets?

**Research Question Option**: Does a lightweight tree-based models (LightGBM/XGBoost) outperform a pretrained transformer-based baseline (TabPFN 2.5) across multiple heterogeneous tabular datasets?


**答案**: **YES!** 
- LightGBM平均准确率：**0.9143** vs TabPFN baseline：**0.8752** (+4.5% 提升)
- 特别是在HELOC数据集上改进显著：**0.8931 vs 0.7734** (+15.5% 提升)

**核心创新点**: 
- 设计了一个 **dataset-agnostic（数据集无关）** 的统一Pipeline
- 针对TabPFN的50000样本限制，我们的模型能处理大规模数据集（如HIGGS的175k样本）
- 证明了轻量级树模型在表格数据上可以超越大型预训练Transformer

---

## 📊 项目结构与数据流

```
数据流向：
run.py → train.py → data_utils.py → models_tabular.py → predict.py → Kaggle提交
         ↓
    [数据加载] → [预处理] → [模型训练] → [验证评估] → [生成预测] → [合并提交]
```

---

## 🔍 建议的阅读顺序

### 1️⃣ **先读：run.py**（入口脚本）
**位置**: 根目录 `/run.py`  
**作用**: 整个Pipeline的主入口，协调训练和预测

**关键函数**:
```python
def run_full_pipeline(model_type="lgbm", use_cv=True, ...):
    # Step 1: 数据概览
    # Step 2: 训练所有数据集
    # Step 3: 生成预测并保存提交文件
```

**如何运行**:
```bash
python run.py                    # 默认使用LightGBM
python run.py --model xgb        # 使用XGBoost
python run.py --model rf         # 使用Random Forest
python run.py --no-cv            # 跳过交叉验证（更快）
```

---

### 2️⃣ **核心1：src/data_utils.py**（数据处理层）
**作用**: 实现 **dataset-agnostic input layer**，统一三个数据集的接口

#### 设计理念：
每个数据集都有一个独立的Loader类，但都继承自同一个 `DataLoader` 基类，提供统一接口：
- `load_train_data()` → 返回 (X, y, feature_columns)
- `load_test_data()` → 返回 (X, feature_columns)

#### 三个数据集的特点处理：

**CovTypeDataLoader** (森林覆盖类型)
```python
- 55个特征，无缺失值
- 7分类问题（Cover_Type: 1-7）
- ID从1开始
- 数据已清洁，无需特殊处理
```

**HELOCDataLoader** (信用评分)
```python
- 23个特征
- 二分类：Good=1, Bad=0
- ID从3501开始
- ⚠️ 缺失值处理：负数表示缺失
  def _handle_missing(X):
      # 负数 → 用该列的中位数填充
      median_val = np.median(valid_values)
```

**HIGGSDataLoader** (希格斯玻色子)
```python
- 30个特征 + 1个weight列
- 二分类：signal=1, background=0
- ID从4547开始
- ⚠️ 缺失值处理：-999.0表示缺失
  def _handle_missing(X):
      # -999.0 → 用该列的中位数填充
```

#### 统一接口函数：
```python
def get_data_loader(dataset_name):
    """工厂函数：根据数据集名称返回对应的Loader"""
    loaders = {
        'covtype': CovTypeDataLoader,
        'heloc': HELOCDataLoader,
        'higgs': HIGGSDataLoader
    }
    return loaders[dataset_name]()
```

**核心价值**: 
- ✅ 抽象了数据集差异
- ✅ 统一的预处理逻辑
- ✅ 易于扩展新数据集

---

### 3️⃣ **核心2：src/models_tabular.py**（模型层）
**作用**: 实现多种表格数据模型，统一封装接口

#### 基类设计：
```python
class BaseModel:
    def __init__(self, name, model):
        self.name = name
        self.model = model  # 包装sklearn/lgbm/xgb模型
        
    def fit(X, y):           # 训练
    def predict(X):          # 预测
    def predict_proba(X):    # 概率预测
    def cross_validate(X, y, cv=5):  # 交叉验证
```

#### 实现的模型：

**1. TabPFNModel (Baseline - 预训练模型)**
```python
- 来源：HuggingFace Pre-trained Transformer
- 版本：TabPFN 2.5
- ⚠️ 限制：训练样本数必须 ≤ 50,000
- 特点：不需要超参数调优，开箱即用
- Kaggle得分：0.95085
```

**2. LightGBMModel (我们的主要模型)**
```python
- 树模型，适合表格数据
- 超参数：
  n_estimators=500
  learning_rate=0.05
  max_depth=-1
  num_leaves=31
  class_weight="balanced"  # 处理类别不平衡
- Kaggle得分：0.95180（略优于baseline）
- 平均验证准确率：0.9143
```

**3. XGBoostModel**
```python
- 另一个强大的树模型
- 超参数：
  n_estimators=500
  learning_rate=0.05
  max_depth=6
- 平均验证准确率：0.9279（最好！）
```

**4. RandomForestModel**
```python
- 简单的树模型baseline
- 超参数：
  n_estimators=200
  class_weight="balanced"
```

**5. EnsembleModel (集成学习)**
```python
class EnsembleModel:
    def __init__(self, models, voting="soft"):
        # 结合多个模型的预测
        # voting="soft": 基于概率的软投票
```

#### 模型获取函数：
```python
def get_model(model_type, **kwargs):
    """工厂函数：根据模型类型返回模型实例"""
    models = {
        'baseline': TabPFNModel,
        'tabpfn': TabPFNModel,
        'lgbm': LightGBMModel,
        'xgb': XGBoostModel,
        'rf': RandomForestModel,
        # ... 还有LR, MLP, SVM等
    }
```

---

### 4️⃣ **核心3：src/train.py**（训练流程）
**作用**: 训练模型并评估性能

#### 主要函数：

**train_single_dataset()**
```python
def train_single_dataset(dataset_name, model_type="lgbm", 
                         use_cv=True, cv_folds=5, ...):
    # 1. 加载数据
    loader = get_data_loader(dataset_name)
    X_train, y_train, feature_cols = loader.load_train_data()
    
    # 2. TabPFN特殊处理：抽样到50k
    if model_type in {"baseline", "tabpfn"}:
        if X_train.shape[0] > 50000:
            X_train, y_train = downsample(50000)
    
    # 3. 加载最佳超参数
    best_params = get_best_params_per_dataset()
    
    # 4. 构建模型
    model = get_model(model_type, **params)
    
    # 5. 交叉验证（可选）
    if use_cv:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    # 6. 训练/验证分割
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, 
                                                  test_size=0.15)
    
    # 7. 训练模型
    model.fit(X_train, y_train)
    
    # 8. 评估
    val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    
    # 9. 保存模型
    pickle.dump(model, f"{dataset_name}_{model_type}_model.pkl")
    
    return {"model": model, "val_accuracy": val_accuracy, ...}
```

**train_all_datasets()**
```python
def train_all_datasets(model_type="lgbm", ...):
    results = {}
    for dataset_name in ["covtype", "heloc", "higgs"]:
        result = train_single_dataset(dataset_name, model_type, ...)
        results[dataset_name] = result
    
    # 打印汇总
    print("TRAINING SUMMARY")
    for name, result in results.items():
        print(f"{name.upper()}: Validation Accuracy = {result['val_accuracy']:.4f}")
    
    return results
```

---

### 5️⃣ **核心4：src/predict.py**（预测和提交）
**作用**: 生成测试集预测并创建Kaggle提交文件

#### 主要函数：

**predict_single_dataset()**
```python
def predict_single_dataset(dataset_name, model_type="lgbm", ...):
    # 1. 加载训练好的模型
    model = load_model(dataset_name, model_type)
    
    # 2. 加载测试数据
    loader = get_data_loader(dataset_name)
    _, _, train_feature_cols = loader.load_train_data()  # 确定特征列
    X_test, feature_cols = loader.load_test_data()
    
    # 3. 预测
    predictions = model.predict(X_test)
    
    # 4. 创建提交DataFrame
    id_start = loader.id_start  # CovType=1, HELOC=3501, HIGGS=4547
    submission = pd.DataFrame({
        "ID": range(id_start, id_start + len(predictions)),
        "Prediction": predictions
    })
    
    # 5. 保存
    submission.to_csv(f"{dataset_name}_test_submission.csv", index=False)
    
    return submission
```

**predict_all_datasets()**
```python
def predict_all_datasets(model_type="lgbm", save_combined=True, ...):
    # 1. 预测三个数据集
    submissions = {}
    for dataset_name in ["covtype", "heloc", "higgs"]:
        sub = predict_single_dataset(dataset_name, model_type, ...)
        submissions[dataset_name] = sub
    
    # 2. 合并成一个提交文件（Kaggle要求）
    if save_combined:
        combined = pd.concat([
            submissions['covtype'],
            submissions['heloc'],
            submissions['higgs']
        ], ignore_index=True)
        
        combined.to_csv("combined_submission.csv", index=False)
        print(f"Combined submission: {len(combined)} predictions")
```

#### Kaggle提交文件格式：
```csv
ID,Prediction
1,2          # CovType: ID 1-3500
2,3
...
3500,5
3501,0       # HELOC: ID 3501-4546
3502,1
...
4546,0
4547,1       # HIGGS: ID 4547-79546
4548,0
...
```

---

### 6️⃣ **baseline.py**（Baseline运行器）
**作用**: 专门用于运行TabPFN baseline的脚本

```python
def main():
    # 只训练baseline模型
    if args.train_only:
        train_all_datasets(model_type="baseline", ...)
    
    # 只生成预测
    if args.predict_only:
        predict_all_datasets(model_type="baseline", ...)
    
    # 默认：训练+预测
    train_all_datasets(model_type="baseline", ...)
    predict_all_datasets(model_type="baseline", ...)
```

**运行方式**:
```bash
python baseline.py               # 训练+预测baseline
python baseline.py --no-cv       # 跳过交叉验证（更快）
python baseline.py --train-only  # 只训练
```

---

## 📊 实验结果对比

### Validation Accuracy（验证集准确率）

| Model     | CovType | HELOC | HIGGS | **Average** |
|-----------|---------|-------|-------|-------------|
| TabPFN    | 0.9869  | 0.7734| 0.8652| **0.8752**  |
| LightGBM  | 0.9682  | 0.8931| 0.8816| **0.9143**  |
| XGBoost   | 0.9881  | 0.8839| 0.9119| **0.9279**  |

### Kaggle Leaderboard Score

| Model     | Score   |
|-----------|---------|
| TabPFN    | 0.95085 |
| LightGBM  | 0.95180 |

**关键发现**:
- ✅ LightGBM和XGBoost在平均准确率上明显优于TabPFN baseline
- ✅ 我们的模型在HELOC数据集上改进最大（0.7734 → 0.8931）
- ✅ XGBoost表现最好，但LightGBM在Kaggle上略胜

---

## 🎯 核心设计思想

### 1. **Dataset-Agnostic Input Layer（数据集无关输入层）**
```python
# 统一接口设计
loader = get_data_loader("covtype")  # 或 "heloc" 或 "higgs"
X, y, feature_cols = loader.load_train_data()  # 接口一致！
```

**优势**:
- 新增数据集只需继承 `DataLoader` 基类
- 预处理逻辑在Loader内部封装
- 训练代码完全数据集无关

### 2. **Unified Training Pipeline（统一训练流程）**
```python
# 相同的训练流程适用于所有数据集
for dataset_name in ["covtype", "heloc", "higgs"]:
    train_single_dataset(dataset_name, model_type="lgbm")
```

### 3. **针对TabPFN限制的解决方案**
```python
# TabPFN限制：≤50k样本
# 我们的解决方案：
if model_type == "baseline" and X_train.shape[0] > 50000:
    # 分层抽样保持类别比例
    X_train, y_train = stratified_downsample(50000)

# LightGBM/XGBoost没有这个限制，可以用全部数据！
```

---

## 🚀 快速开始

### 1. 训练并预测（推荐）
```bash
# 使用LightGBM（默认）
python run.py

# 使用XGBoost
python run.py --model xgb

# 快速测试（跳过CV）
python run.py --no-cv
```

### 2. 只运行Baseline
```bash
python baseline.py
```

### 3. 分步运行
```bash
# 步骤1：训练
python src/train.py --dataset covtype --model lgbm
python src/train.py --dataset heloc --model lgbm
python src/train.py --dataset higgs --model lgbm

# 步骤2：预测
python src/predict.py --dataset covtype --model lgbm
python src/predict.py --dataset heloc --model lgbm
python src/predict.py --dataset higgs --model lgbm
```

---

## 📝 代码阅读检查清单

完成这些任务后，你就完全理解代码了：

- [ ] **理解数据流**: 数据如何从CSV → DataLoader → 模型 → 预测
- [ ] **理解dataset-agnostic设计**: 如何用统一接口处理不同数据集
- [ ] **理解缺失值处理**: HELOC的负数、HIGGS的-999.0
- [ ] **理解模型封装**: BaseModel如何统一不同模型的接口
- [ ] **理解训练流程**: 交叉验证 → 训练/验证分割 → 训练 → 评估
- [ ] **理解提交文件生成**: ID如何分配、如何合并三个数据集
- [ ] **理解TabPFN限制**: 为什么需要抽样、如何抽样
- [ ] **对比实验结果**: 为什么LightGBM优于TabPFN

---

## 🎓 项目亮点（用于海报）

### 1. **Research Question**
**Does a lightweight gradient boosting model (LightGBM) outperform a pretrained transformer-based baseline (TabPFN 2.5) across multiple heterogeneous tabular datasets?**

**答案**: **YES!** 
- 平均准确率从 **0.8752 → 0.9143** (+4.5%)
- HELOC数据集改进最显著：**0.7734 → 0.8931** (+15.5%)
- Kaggle排行榜得分：**0.95085 → 0.95180**

### 2. **核心创新**
- ✅ Dataset-agnostic统一Pipeline设计
- ✅ 智能缺失值处理（中位数填充）
- ✅ 针对类别不平衡的class_weight调整
- ✅ 突破TabPFN的50k样本限制（HIGGS数据集175k样本）
- ✅ 证明轻量级模型可超越大型预训练Transformer

### 3. **实验对比**
- Simple Baseline: Logistic Regression
- Complex Baseline: TabPFN（预训练）
- Our Models: LightGBM, XGBoost
- 最佳表现: XGBoost (0.9279)

### 4. **计算复杂度对比**（待补充）
需要添加：
- 参数数量
- 训练时间
- 推理速度
- 内存占用

---

## ❓ 常见问题

**Q: 为什么HIGGS数据集要单独处理weight列？**  
A: 物理实验数据带有事件权重，但我们的模型不使用，所以在 `load_train_data()` 中排除了。

**Q: 为什么用中位数而不是均值填充缺失值？**  
A: 中位数对异常值更鲁棒，适合表格数据。

**Q: 为什么LightGBM在Kaggle上略优于XGBoost？**  
A: 可能是过拟合问题，或者XGBoost需要更多调参。

**Q: ID为什么是1, 3501, 4547这样不连续？**  
A: Kaggle要求，用于区分不同数据集的预测。

---

## 📚 推荐阅读顺序总结

1. **run.py** - 理解整体流程
2. **data_utils.py** - 理解数据处理
3. **models_tabular.py** - 理解模型封装
4. **train.py** - 理解训练逻辑
5. **predict.py** - 理解预测和提交
6. **baseline.py** - 理解baseline运行

祝你理解顺利！🎉
