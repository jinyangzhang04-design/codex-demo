from pathlib import Path
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from drugscreen.data.standardize import standardize_smiles
from drugscreen.features.descriptors import featurize
from drugscreen.models.baseline import BaselineModel
import pandas as pd

app = FastAPI(title="DrugScreen")
MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "demo-baseline.joblib"

PAGE = '''<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>DrugScreen</title><style>body{{font:16px system-ui;max-width:900px;margin:40px auto;padding:0 20px}}textarea{{width:100%;height:140px;padding:10px}}button{{padding:10px 22px;margin:12px 0}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:8px;text-align:left}}</style></head><body><h1>小分子成药性预测与虚拟筛选</h1><p>每行输入一个 SMILES：</p><form method="post"><textarea name="smiles">CCO\nCCN\nc1ccccc1</textarea><br><button type="submit">开始筛选</button></form>{result}</body></html>'''

def render(rows):
    if not rows: return ""
    head = "<h2>筛选结果</h2><table><tr><th>SMILES</th><th>任务 1</th><th>任务 2</th><th>综合分数</th></tr>"
    body = "".join(f"<tr><td>{r['smiles']}</td><td>{r['task_0']:.4f}</td><td>{r['task_1']:.4f}</td><td>{r['score']:.4f}</td></tr>" for r in rows)
    return head + body + "</table>"

@app.get("/", response_class=HTMLResponse)
def index(): return PAGE.format(result="")

@app.post("/", response_class=HTMLResponse)
def screen(smiles: str = Form(...)):
    values = [standardize_smiles(x.strip()) for x in smiles.splitlines()]
    values = [x for x in values if x]
    if not values: return PAGE.format(result="<p>没有检测到合法 SMILES。</p>")
    if not MODEL_PATH.exists(): return PAGE.format(result="<p>模型不存在，请先运行训练命令。</p>")
    X = pd.DataFrame([featurize(x) for x in values]).values
    pred = BaselineModel.load(MODEL_PATH).predict(X)
    out = pd.DataFrame(pred, columns=[f"task_{i}" for i in range(pred.shape[1])]); out.insert(0, "smiles", values)
    out["score"] = out[["task_0", "task_1"]].sum(axis=1)
    return PAGE.format(result=render(out.sort_values("score", ascending=False).to_dict("records")))
