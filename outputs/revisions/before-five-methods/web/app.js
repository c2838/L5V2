"use strict";
(() => {
  const datasets = window.AIOT_RESULTS || [];
  const $ = id => document.getElementById(id);
  const make = (tag, text, className) => { const n=document.createElement(tag); if(text!==undefined)n.textContent=text; if(className)n.className=className; return n; };
  const fmt = (n, digits=2) => Number(n).toLocaleString("zh-TW",{minimumFractionDigits:digits,maximumFractionDigits:digits});
  const text = (id,value) => $(id).textContent=value;
  if(!datasets.length){text("dataset-subtitle","尚無分析結果，請先執行 Python 分析。");return;}
  let data, method, k;
  let predictionUrl;
  const palette = ["#29634b", "#b9853a", "#85978b"];
  const option=(value,label)=>{const o=make("option",label);o.value=value;return o;};
  datasets.forEach((d,i)=>$("dataset").append(option(i,d.metadata.title)));
  const currentRows=()=>data.rows.filter(r=>r.method===method);
  const current=()=>currentRows().find(r=>r.k===k);
  function svgElement(tag, attrs={}, value){const n=document.createElementNS("http://www.w3.org/2000/svg",tag);for(const [a,v]of Object.entries(attrs))n.setAttribute(a,v);if(value!==undefined)n.textContent=value;return n;}
  function chart(id,series,{xLabel="編碼後特徵數",yLabel="",scatter=false,reference=null}={}){
    const container=$(id);container.replaceChildren();
    const width=520,height=265,margin={left:64,right:17,top:18,bottom:48};
    const points=series.flatMap(s=>s.points).filter(p=>p.every(Number.isFinite));
    if(!points.length)return;
    let xmin=Math.min(...points.map(p=>p[0])), xmax=Math.max(...points.map(p=>p[0]));
    let ymin=Math.min(...points.map(p=>p[1])), ymax=Math.max(...points.map(p=>p[1]));
    if(reference==="diagonal"){xmin=ymin=Math.min(xmin,ymin);xmax=ymax=Math.max(xmax,ymax);}
    if(reference==="zero"){ymin=Math.min(0,ymin);ymax=Math.max(0,ymax);}
    const dx=xmax-xmin||1,dy=ymax-ymin||1;
    xmin-=dx*.04;xmax+=dx*.04;ymin-=dy*.12;ymax+=dy*.12;
    const sx=x=>margin.left+(x-xmin)/(xmax-xmin)*(width-margin.left-margin.right);
    const sy=y=>height-margin.bottom-(y-ymin)/(ymax-ymin)*(height-margin.top-margin.bottom);
    const svg=svgElement("svg",{viewBox:`0 0 ${width} ${height}`,role:"img","aria-label":`${yLabel} 圖表`});
    svg.append(svgElement("title",{},`${yLabel}；橫軸 ${xLabel}`));
    const tick=n=>yLabel==="R²"?fmt(n,3):Math.abs(n)>=10000?fmt(n/1000,1)+"k":Math.abs(n)>=100?fmt(n,0):fmt(n,2);
    for(let i=0;i<5;i++){
      const value=ymin+(ymax-ymin)*i/4,y=sy(value);
      svg.append(svgElement("line",{x1:margin.left,x2:width-margin.right,y1:y,y2:y,stroke:"#e5e8dd","stroke-width":1}));
      svg.append(svgElement("text",{x:margin.left-10,y:y+4,"text-anchor":"end",fill:"#7d897c","font-size":10},tick(value)));
    }
    const rawX=[...new Set(points.map(p=>p[0]))].sort((a,b)=>a-b);
    const ticks=scatter?Array.from({length:4},(_,i)=>xmin+(xmax-xmin)*i/3):rawX.filter((_,i)=>rawX.length<=7||i%Math.ceil(rawX.length/6)===0||i===rawX.length-1);
    for(const v of ticks)svg.append(svgElement("text",{x:sx(v),y:height-margin.bottom+20,"text-anchor":"middle",fill:"#7d897c","font-size":10},scatter?tick(v):fmt(v,0)));
    svg.append(svgElement("text",{x:(margin.left+width-margin.right)/2,y:height-6,"text-anchor":"middle",fill:"#7d897c","font-size":10},xLabel));
    if(reference==="diagonal")svg.append(svgElement("line",{x1:sx(xmin),y1:sy(xmin),x2:sx(xmax),y2:sy(xmax),stroke:palette[1],"stroke-dasharray":"5 4"}));
    if(reference==="zero")svg.append(svgElement("line",{x1:margin.left,y1:sy(0),x2:width-margin.right,y2:sy(0),stroke:palette[1],"stroke-dasharray":"5 4"}));
    series.forEach((s,i)=>{
      const color=s.color||palette[i%palette.length];
      if(!scatter)svg.append(svgElement("polyline",{points:s.points.map(p=>`${sx(p[0])},${sy(p[1])}`).join(" "),fill:"none",stroke:color,"stroke-width":2.3}));
      s.points.forEach(p=>{const circle=svgElement("circle",{cx:sx(p[0]),cy:sy(p[1]),r:scatter?3.7:4,fill:color,opacity:scatter?.7:1});circle.append(svgElement("title",{},`${s.name}: ${fmt(p[0])}, ${fmt(p[1],4)}`));svg.append(circle);});
    });
    container.append(svg);
    const legend=make("div",undefined,"chart-legend");
    series.forEach((s,i)=>{const item=make("span");const swatch=make("i");swatch.style.backgroundColor=s.color||palette[i%palette.length];item.append(swatch,document.createTextNode(s.name));legend.append(item);});
    container.append(legend);
  }
  function selectDataset(index){
    data=datasets[index];const best=data.rows.find(r=>r.key===data.best_key);method=best.method;k=best.k;
    text("dataset-subtitle",data.metadata.subtitle);
    text("sample-count",`${data.metadata.rows} 筆資料`);text("feature-count",`${data.metadata.raw_feature_count} 個原始欄位`);
    text("best-summary",`CV 選定 · ${best.method==="lasso"?"Lasso":best.method==="forward"?"Forward":"全特徵"} / ${best.k} 個特徵`);
    text("business-note",`以 ${data.metadata.raw_feature_count} 個原始特徵預測 ${data.metadata.target}，目標單位為${data.metadata.unit}。`);
    const missing=Object.values(data.metadata.missing_values).reduce((a,b)=>a+b,0);
    text("data-note",`${data.metadata.rows} 筆有效資料，${missing} 個缺值、${data.metadata.duplicate_rows} 筆重複列；檢查型態與缺值後，保留原始欄位的意義。`);
    const baseline=data.rows.find(r=>r.method==="baseline");
    text("conclusion",`CV 選定 ${best.k} 個特徵，Test RMSE 為 ${fmt(best.test.rmse)}；全特徵基準為 ${fmt(baseline.test.rmse)}。這次切分的結果${best.test.rmse<baseline.test.rmse?"顯示測試誤差降低":"未顯示測試誤差降低"}，仍需更多資料驗證。`);
    const source=$("source-note");source.replaceChildren(document.createTextNode("資料來源："));
    if(data.metadata.source.startsWith("https://")){const link=make("a",data.metadata.source);link.href=data.metadata.source;link.target="_blank";link.rel="noopener noreferrer";source.append(link);}else source.append(document.createTextNode(data.metadata.source));
    text("experiment-note",`random_state=${data.seed}；訓練 ${data.train_rows} 筆、測試 ${data.test_rows} 筆；training CV ${data.cv_folds} folds。SHA-256：${data.metadata.sha256}`);
    $("download-comparison").href=`downloads/${data.metadata.name}/comparison.csv`;
    $("download-report").href=`downloads/${data.metadata.name}/report.md`;
    text("prediction-title",`試算${data.metadata.target==="Profit"?"一家公司的獲利":data.metadata.target==="MEDV"?"一個地區的房價":"一筆新資料"}`);
    text("prediction-label",`預測 ${data.metadata.target}`);text("prediction-unit",data.metadata.unit);
    renderInputs();updateK();render();
    const lc=data.learning_curve;
    chart("learning-chart",[{name:"訓練誤差",points:lc.sizes.map((s,i)=>[s,lc.train_rmse[i]])},{name:"驗證誤差",points:lc.sizes.map((s,i)=>[s,lc.validation_rmse[i]])}],{xLabel:"訓練資料量（筆）",yLabel:"Learning curve RMSE"});
  }
  function updateK(){const rows=currentRows();if(!rows.some(r=>r.k===k))k=rows[0].k;$("feature-k").replaceChildren(...rows.map(r=>option(r.k,`${r.k} 個特徵`)));$("feature-k").value=k;}
  function renderInputs(){
    const container=$("prediction-inputs");container.replaceChildren();
    data.input_schema.forEach((s,i)=>{
      const box=make("div",undefined,"input-item"),label=make("label",s.name),input=s.type==="category"?make("select"):make("input");
      input.id=`input-${i}`;label.htmlFor=input.id;input.dataset.source=s.name;
      if(s.type==="category")s.values.forEach(v=>input.append(option(v,v)));else{input.type="number";input.step="any";box.append(make("small",`訓練範圍 ${fmt(s.min)}–${fmt(s.max)}`));}
      input.value=s.type==="number"?Number(s.default.toPrecision(12)):s.default;
      input.addEventListener("input",predict);box.prepend(label,input);container.append(box);
    });predict();
  }
  function predict(){
    const row=current();if(!row)return;
    const values={},outOfRange=[];
    data.input_schema.forEach((s,i)=>{const input=$(`input-${i}`);values[s.name]=s.type==="number"?(input.value===""?s.default:input.valueAsNumber):input.value;
      if(s.type==="number"&&(values[s.name]<s.min||values[s.name]>s.max))outOfRange.push(s.name);});
    if(Object.values(values).some(v=>typeof v==="number"&&!Number.isFinite(v))){text("predicted-value","請輸入有效數值");return;}
    let prediction=row.model.intercept;
    for(const term of row.model.terms)prediction+=term.coefficient*(term.kind==="number"?values[term.source]:Number(String(values[term.source])===term.value));
    text("predicted-value",fmt(prediction));
    text("range-note",outOfRange.length?`${outOfRange.join("、")} 超出訓練資料範圍，這是外推預測。`:"空白數值使用訓練集中位數。灰色欄位未被目前模型使用。");
  }
  function render(){
    const row=current();if(!row)return;
    document.querySelectorAll("[data-method]").forEach(b=>{b.classList.toggle("active",b.dataset.method===method);b.setAttribute("aria-pressed",b.dataset.method===method);});
    text("selection-status",row.key===data.best_key?"● 目前為 CV 選定模型":"● 正在探索其他特徵組合");
    text("method-detail",method==="lasso"?`以標準化 Lasso 係數排序，alpha=${row.alpha}；再以選定欄位訓練線性回歸。`:method==="forward"?"逐步加入使訓練集內交叉驗證誤差較低的特徵。":"使用全部編碼後的特徵建立線性回歸基準。");
    $("metrics").replaceChildren();
    const specs=[["TEST RMSE",fmt(row.test.rmse),`平均平方誤差的平方根 · ${data.metadata.unit}`],["TEST R²",fmt(row.test.r2,4),"越接近 1，對這次測試資料的解釋程度越高"],["TEST MAE",fmt(row.test.mae),`平均絕對誤差 · ${data.metadata.unit}`],["TEST MSE",fmt(row.test.mse),"大誤差權重較高 · 目標單位的平方"]];
    specs.forEach((s,i)=>{const tile=make("article",undefined,"metric"+(i===0?" featured":""));tile.append(make("div",s[0],"metric-label"),make("div",s[1],"metric-value"),make("div",s[2],"metric-note"));$("metrics").append(tile);});
    $("feature-chips").replaceChildren(...row.features.map(n=>make("span",n,"chip")));
    const rows=currentRows();
    chart("rmse-chart",[{name:"訓練集",points:rows.map(r=>[r.k,r.train.rmse])},{name:"測試集",points:rows.map(r=>[r.k,r.test.rmse])},{name:"Training CV",points:rows.map(r=>[r.k,r.cv_rmse])}],{yLabel:"RMSE"});
    chart("r2-chart",[{name:"測試集 R²",points:rows.map(r=>[r.k,r.test.r2])}],{yLabel:"R²"});
    const tbody=$("comparison-body");tbody.replaceChildren();
    rows.forEach(r=>{const tr=make("tr");tr.classList.toggle("selected",r.key===row.key);const count=make("td"),button=make("button",`${r.k}`);button.setAttribute("aria-label",`查看 ${r.k} 個特徵模型`);button.onclick=()=>{k=r.k;$("feature-k").value=k;render();};count.append(button);if(r.key===data.best_key)count.append(make("span","CV 選定","best-cell"));tr.append(count,make("td",r.features.join(" · "),"features-cell"),make("td",`${fmt(r.cv_rmse)} ± ${fmt(r.cv_rmse_std)}`),make("td",fmt(r.test.rmse)),make("td",fmt(r.test.r2,4)));tbody.append(tr);});
    data.input_schema.forEach((s,i)=>$(`input-${i}`).parentElement.classList.toggle("inactive",!row.model.terms.some(t=>t.source===s.name)));
    const terms=row.model.terms.map(t=>`${t.coefficient>=0?"+":"−"} ${fmt(Math.abs(t.coefficient),4)} × ${t.kind==="category"?`[${t.source} = ${t.value}]`:t.source}`);
    text("equation",`${data.metadata.target} = ${fmt(row.model.intercept,4)} ${terms.join(" ")}`);
    chart("actual-chart",[{name:"保留測試集",points:row.predictions.map(p=>[p.actual,p.predicted])}],{xLabel:"實際值",yLabel:"預測值",scatter:true,reference:"diagonal"});
    chart("residual-chart",[{name:"殘差",points:row.predictions.map(p=>[p.predicted,p.residual])}],{xLabel:"預測值",yLabel:"殘差",scatter:true,reference:"zero"});
    if(predictionUrl)URL.revokeObjectURL(predictionUrl);
    const csv="id,actual,predicted,residual\n"+row.predictions.map(p=>[p.id,p.actual,p.predicted,p.residual].join(",")).join("\n");
    predictionUrl=URL.createObjectURL(new Blob([csv],{type:"text/csv;charset=utf-8"}));$("download-predictions").href=predictionUrl;$("download-predictions").download=`${data.metadata.name}-${row.key}-predictions.csv`;
    predict();
  }
  $("dataset").onchange=e=>selectDataset(Number(e.target.value));
  $("feature-k").onchange=e=>{k=Number(e.target.value);render();};
  document.querySelectorAll("[data-method]").forEach(b=>b.onclick=()=>{method=b.dataset.method;updateK();render();});
  $("reset-inputs").onclick=()=>{renderInputs();render();};
  selectDataset(0);
})();
