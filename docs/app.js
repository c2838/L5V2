"use strict";
(() => {
  const datasets=window.AIOT_RESULTS||[];
  const $=id=>document.getElementById(id);
  const make=(tag,value,className)=>{const el=document.createElement(tag);if(value!==undefined)el.textContent=value;if(className)el.className=className;return el;};
  const text=(id,value)=>$(id).textContent=value;
  const fmt=(value,digits=2)=>value===null||value===undefined?"—":Number(value).toLocaleString("en-US",{minimumFractionDigits:digits,maximumFractionDigits:digits});
  const pvalue=value=>value===null||value===undefined?"—":value<.0001?"< 0.0001":fmt(value,4);
  const names={lasso:"Lasso",forward:"Forward SFS",rfe:"RFE",elasticnet:"ElasticNet",exhaustive:"Exhaustive Search",baseline:"Baseline OLS"};
  const chinese={lasso:"L1 係數排序",forward:"前向選擇",rfe:"遞迴特徵消除",elasticnet:"L1＋L2 係數排序",exhaustive:"全子集合搜尋",baseline:"全特徵基準"};
  const colors={lasso:"#6258c7",forward:"#258960",rfe:"#dc9744",elasticnet:"#cf6088",exhaustive:"#3b90b3",baseline:"#818b9c"};
  const dashes={lasso:"",forward:"7 4",rfe:"2 4",elasticnet:"8 3 2 3",exhaustive:"3 2 1 2"};
  let data,method,k,comparison=false,predictionUrl;
  const option=(value,label)=>{const el=make("option",label);el.value=value;return el;};
  const methodRows=m=>data.rows.filter(r=>r.method===m);
  const current=()=>data.rows.find(r=>r.method===method&&r.k===k);
  const cvSelected=row=>data.cv_best_keys.includes(row.key);
  const bilingual=(element,en,zh)=>{element.replaceChildren(document.createTextNode(en));if(zh)element.append(make("small",zh));};
  if(!datasets.length){text("explorer-title","No analysis results / 尚無分析結果");return;}
  datasets.forEach((d,i)=>$("dataset").append(option(i,d.metadata.title)));
  function svgEl(tag,attrs={},value){const el=document.createElementNS("http://www.w3.org/2000/svg",tag);for(const [key,v]of Object.entries(attrs))el.setAttribute(key,v);if(value!==undefined)el.textContent=value;return el;}
  function chart(id,series,{xLabel="Number of Encoded Features",xZh="編碼後特徵數",yLabel="",yZh="",scatter=false,reference=null}={}){
    const container=$(id);container.replaceChildren();
    const points=series.flatMap(s=>s.points).filter(p=>p.every(Number.isFinite));if(!points.length)return;
    const width=550,height=305,margin={left:76,right:18,top:20,bottom:64};
    let xmin=Math.min(...points.map(p=>p[0])),xmax=Math.max(...points.map(p=>p[0]));
    let ymin=Math.min(...points.map(p=>p[1])),ymax=Math.max(...points.map(p=>p[1]));
    if(reference==="diagonal"){xmin=ymin=Math.min(xmin,ymin);xmax=ymax=Math.max(xmax,ymax);}
    if(reference==="zero"){ymin=Math.min(0,ymin);ymax=Math.max(0,ymax);}
    const dx=xmax-xmin||1,dy=ymax-ymin||1;xmin-=dx*.04;xmax+=dx*.04;ymin-=dy*.12;ymax+=dy*.12;
    const sx=x=>margin.left+(x-xmin)/(xmax-xmin)*(width-margin.left-margin.right);
    const sy=y=>height-margin.bottom-(y-ymin)/(ymax-ymin)*(height-margin.top-margin.bottom);
    const svg=svgEl("svg",{viewBox:`0 0 ${width} ${height}`,role:"img","aria-label":`${yLabel} vs. ${xLabel}`});
    svg.append(svgEl("title",{},`${yLabel} vs. ${xLabel}`));
    const tick=n=>yLabel.includes("R-squared")?fmt(n,3):Math.abs(n)>=10000?fmt(n/1000,1)+"k":Math.abs(n)>=100?fmt(n,0):fmt(n,2);
    for(let i=0;i<5;i++){const value=ymin+(ymax-ymin)*i/4,y=sy(value);svg.append(svgEl("line",{x1:margin.left,x2:width-margin.right,y1:y,y2:y,stroke:"#e8eaf2"}));svg.append(svgEl("text",{x:margin.left-10,y:y+4,"text-anchor":"end",fill:"#788294","font-size":10},tick(value)));}
    const rawX=[...new Set(points.map(p=>p[0]))].sort((a,b)=>a-b);
    const ticks=scatter?Array.from({length:4},(_,i)=>xmin+(xmax-xmin)*i/3):rawX.filter((_,i)=>rawX.length<=7||i%Math.ceil(rawX.length/6)===0||i===rawX.length-1);
    ticks.forEach(v=>svg.append(svgEl("text",{x:sx(v),y:height-margin.bottom+20,"text-anchor":"middle",fill:"#788294","font-size":10},scatter?tick(v):fmt(v,0))));
    const center=(margin.left+width-margin.right)/2;
    svg.append(svgEl("text",{x:center,y:height-20,"text-anchor":"middle",fill:"#526075","font-size":13},xLabel));
    if(xZh)svg.append(svgEl("text",{x:center,y:height-5,"text-anchor":"middle",fill:"#939bae","font-size":9},xZh));
    svg.append(svgEl("text",{transform:`translate(16 ${(margin.top+height-margin.bottom)/2}) rotate(-90)`,"text-anchor":"middle",fill:"#526075","font-size":13},yLabel));
    if(yZh)svg.append(svgEl("text",{transform:`translate(31 ${(margin.top+height-margin.bottom)/2}) rotate(-90)`,"text-anchor":"middle",fill:"#939bae","font-size":9},yZh));
    if(reference==="diagonal")svg.append(svgEl("line",{x1:sx(xmin),y1:sy(xmin),x2:sx(xmax),y2:sy(xmax),stroke:"#a1aabc","stroke-dasharray":"5 4"}));
    if(reference==="zero")svg.append(svgEl("line",{x1:margin.left,y1:sy(0),x2:width-margin.right,y2:sy(0),stroke:"#a1aabc","stroke-dasharray":"5 4"}));
    series.forEach((s,i)=>{const color=s.color||colors[s.method]||"#6258c7";if(!scatter)svg.append(svgEl("polyline",{points:s.points.map(p=>`${sx(p[0])},${sy(p[1])}`).join(" "),fill:"none",stroke:color,"stroke-width":2.2,"stroke-dasharray":dashes[s.method]||""}));s.points.forEach(p=>{const dot=svgEl("circle",{cx:sx(p[0]),cy:sy(p[1]),r:scatter?3.4:Math.max(2.1,4-i*.35),fill:comparison&&!scatter?"#fff":color,stroke:color,"stroke-width":1.2,opacity:scatter?.7:1});dot.append(svgEl("title",{},`${s.name}: ${fmt(p[0])}, ${fmt(p[1],4)}`));svg.append(dot);});});
    container.append(svg);const legend=make("div",undefined,"chart-legend");series.forEach(s=>{const item=make("span"),swatch=make("i");swatch.style.borderColor=s.color||colors[s.method]||"#6258c7";if(dashes[s.method])swatch.style.borderTopStyle="dashed";item.append(swatch,document.createTextNode(s.name));if(s.zh)item.append(make("small",s.zh));legend.append(item);});container.append(legend);
  }
  function setDataset(index){
    data=datasets[index];const best=data.rows.find(r=>r.key===data.best_key);method=data.available_methods.includes(best.method)?best.method:"lasso";k=best.k;comparison=false;
    text("explorer-title",`${data.metadata.title} Multiple Linear Regression Explorer`);
    text("explorer-zh",data.metadata.subtitle);
    text("sample-count",`${data.metadata.rows} Samples / 資料筆數`);
    text("feature-count",`${data.available_methods.length} Selection Methods · ${data.encoded_feature_count} Encoded Features`);
    text("best-summary",`Training CV Selected: k = ${best.k} · RMSE ${fmt(best.cv_rmse)}`);
    text("business-note",`Predict ${data.metadata.target} from ${data.metadata.raw_feature_count} source features. Unit: ${data.metadata.unit}.`);
    text("data-note",`${data.metadata.rows} rows; ${Object.values(data.metadata.missing_values).reduce((a,b)=>a+b,0)} missing values; ${data.metadata.duplicate_rows} duplicate rows.`);
    text("conclusion",`Training CV selects k=${best.k}. Held-out test RMSE=${fmt(best.test.rmse)}; R²=${fmt(best.test.r2,4)}. Results may vary with other splits.`);
    const source=$("source-note");source.replaceChildren(document.createTextNode("Dataset source: "));if(data.metadata.source.startsWith("https://")){const a=make("a",data.metadata.source);a.href=data.metadata.source;a.target="_blank";a.rel="noopener noreferrer";source.append(a);}else source.append(document.createTextNode(data.metadata.source));
    text("experiment-note",`random_state=${data.seed}; Train=${data.train_rows}, Test=${data.test_rows}; Training CV=${data.cv_folds} folds. SHA-256: ${data.metadata.sha256}`);
    $("download-comparison").href=`downloads/${data.metadata.name}/comparison.csv`;
    $("download-report").href=`downloads/${data.metadata.name}/report.md`;
    bilingual($("prediction-title"),data.metadata.target==="Profit"?"Live Profit Predictor":data.metadata.target==="MEDV"?"Live House Price Predictor":"Live Regression Predictor",data.metadata.target==="Profit"?"獲利預測試算":data.metadata.target==="MEDV"?"房價預測試算":"數值預測試算");
    text("prediction-label",`PREDICTED ${data.metadata.target.toUpperCase()}`);text("prediction-unit",data.metadata.unit);
    document.querySelectorAll("[data-method]").forEach(button=>{button.disabled=button.dataset.method!=="baseline"&&!data.available_methods.includes(button.dataset.method);button.title=button.disabled?data.exhaustive_search.limit_reason:"";});
    $("compare-all").firstChild.textContent=`Compare All ${data.available_methods.length} Methods `;
    $("compare-all").querySelector('small').textContent=data.available_methods.length===5?'五種方法一起比較':'所有可用方法一起比較';
    renderInputs();updateK();render();
    const lc=data.learning_curve;chart("learning-chart",[{name:"Training RMSE",zh:"訓練誤差",color:"#6258c7",points:lc.sizes.map((s,i)=>[s,lc.train_rmse[i]])},{name:"Validation RMSE",zh:"驗證誤差",color:"#258960",points:lc.sizes.map((s,i)=>[s,lc.validation_rmse[i]])}],{xLabel:"Training Sample Count",xZh:"訓練資料量",yLabel:"RMSE",yZh:"均方根誤差"});
  }
  function updateK(){const rows=methodRows(method);if(!rows.some(r=>r.k===k))k=rows[0].k;$("feature-k").replaceChildren(...rows.map(r=>option(r.k,`k = ${r.k}`)));$("feature-k").value=k;}
  function renderInputs(){const container=$("prediction-inputs");container.replaceChildren();data.input_schema.forEach((schema,i)=>{const box=make("div",undefined,"input-item"),label=make("label",schema.name),input=make(schema.type==="category"?"select":"input");input.id=`input-${i}`;label.htmlFor=input.id;input.dataset.source=schema.name;let initial=schema.default;if(schema.type==="category")schema.values.forEach(v=>input.append(option(v,v)));else{input.type="number";input.step="any";if(data.metadata.name==="startup"){const defaults={"R&D Spend":100000,"Administration":120000,"Marketing Spend":150000};initial=defaults[schema.name]??initial;}box.append(make("small",`Train range: ${fmt(schema.min)}–${fmt(schema.max)}`));}input.value=schema.type==="number"?Number(initial.toPrecision(12)):initial;input.addEventListener("input",predict);box.prepend(label,input);container.append(box);});predict();}
  function predict(){const row=current();if(!row)return;const values={},outside=[];data.input_schema.forEach((schema,i)=>{const input=$(`input-${i}`);values[schema.name]=schema.type==="number"?(input.validity.badInput?NaN:input.value===""?schema.default:input.valueAsNumber):input.value;if(schema.type==="number"&&row.model.terms.some(t=>t.source===schema.name)&&(values[schema.name]<schema.min||values[schema.name]>schema.max))outside.push(schema.name);});if(Object.values(values).some(v=>typeof v==="number"&&!Number.isFinite(v))){text("predicted-value","Invalid input");return;}let value=row.model.intercept;row.model.terms.forEach(t=>value+=t.coefficient*(t.kind==="number"?values[t.source]:Number(String(values[t.source])===t.value)));text("predicted-value",fmt(value));text("range-note",outside.length?`Outside training range / 超出訓練範圍: ${outside.join(", ")}`:"Blank numbers use training medians. Gray fields are inactive. / 空白數值用訓練中位數；灰色欄位未使用。");}
  function render(){
    const row=current();if(!row)return;const stats=row.statistics;
    document.querySelectorAll("[data-method]").forEach(button=>{button.classList.toggle("active",button.dataset.method===method&&!comparison);button.setAttribute("aria-pressed",button.dataset.method===method&&!comparison);});$("compare-all").classList.toggle("active",comparison);$("compare-all").setAttribute("aria-pressed",comparison);
    $("k-buttons").replaceChildren(...methodRows(method).map(r=>{const button=make("button",`k = ${r.k}`);button.classList.toggle("selected",r.k===k);button.classList.toggle("cv-best",cvSelected(r));button.setAttribute("aria-label",`Select ${r.k} features`);button.setAttribute("aria-pressed",r.k===k);if(cvSelected(r))button.append(make("small","CV Selected"));button.onclick=()=>{k=r.k;$("feature-k").value=k;render();};return button;}));
    text("selection-status",`${cvSelected(row)?"CV Selected":"Exploring"} · ${names[method]} / k=${k}`);
    const descriptions={lasso:`Standardized L1 coefficient ranking; alpha=${row.alpha}. / 標準化後依 Lasso 係數排序，再訓練 OLS。`,forward:"Inner-CV forward selection; final OLS fit. / 內層交叉驗證逐步加入特徵。",rfe:"Recursive elimination by standardized OLS coefficient weights. / 標準化後逐次移除較弱特徵。",elasticnet:`Standardized L1 + L2 ranking; alpha=${row.alpha}, l1_ratio=${row.l1_ratio}. / 混合正則化係數排序。`,exhaustive:`All ${data.exhaustive_search.subsets_evaluated} subsets searched by training CV. / 實際枚舉全部子集合。`,baseline:"All encoded features, ordinary least squares. / 全特徵線性回歸基準。"};
    text("method-detail",descriptions[method]);
    const specs=[['TEST RMSE','測試集均方根誤差',fmt(row.test.rmse),`Lower is better · ${data.metadata.unit}`],['R² SCORE','測試集決定係數',fmt(row.test.r2,4),'Test set · Not classification accuracy'],['ADJUSTED R²','測試集調整後決定係數',fmt(stats.test_adjusted_r2,4),'Test set · Adjusted for k predictors'],['MEAN ABSOLUTE ERROR','測試集平均絕對誤差',fmt(row.test.mae),data.metadata.unit],['F-STATISTIC','訓練集 OLS 整體 F 檢定',fmt(stats.f_stat),`Training OLS · p ${pvalue(stats.f_p_value)}`]];
    $("metrics").replaceChildren(...specs.map(s=>{const tile=make("article",undefined,"metric"),label=make("div",s[0],"metric-label");label.append(make("small",s[1]));tile.append(label,make("div",s[2],"metric-value"),make("div",s[3],"metric-note"));return tile;}));
    $("feature-chips").replaceChildren(...row.features.map(n=>make("span",n,"chip")));
    const baseline=data.rows.find(r=>r.method==="baseline");
    text("parsimony-note",`Current model: ${names[method]}, k=${k}. Training CV RMSE=${fmt(row.cv_rmse)}; Test RMSE=${fmt(row.test.rmse)}. Baseline OLS Test RMSE=${fmt(baseline.test.rmse)}.`);
    text("mse-note",`Test MSE=${fmt(row.test.mse)} · ${comparison?"All five curves use the same train/test split. / 五方法共用同一切分。":"CV selects the model; test scores describe held-out performance. / 用 CV 選模型，以 test 評估。"}`);
    const methods=comparison?data.available_methods:[method];
    chart("rmse-chart",methods.map(m=>({method:m,name:names[m],zh:chinese[m],points:methodRows(m).map(r=>[r.k,r.test.rmse])})),{yLabel:"Test RMSE",yZh:"測試集均方根誤差"});
    chart("r2-chart",methods.map(m=>({method:m,name:names[m],zh:chinese[m],points:methodRows(m).map(r=>[r.k,r.test.r2])})),{yLabel:"Test R-squared",yZh:"測試集決定係數"});
    bilingual($("results-title"),comparison?`Compare All ${data.available_methods.length} Methods — k = ${k}`:"Feature Selection Results",comparison?"相同特徵數下比較所有方法":"每個特徵數量的模型結果");
    text("comparison-note",comparison?"Same final subsets give the same test scores; CV can differ because selection is repeated within folds. Click a method row for details. / 最後子集合相同會有相同 test 分數；各 fold 重新選特徵，因此 CV 仍可能不同。":"Select a feature-count row to inspect that model. / 點選特徵數列查看該模型。");
    const rows=comparison?data.available_methods.map(m=>data.rows.find(r=>r.method===m&&r.k===k)).filter(Boolean):methodRows(method);
    $("comparison-body").replaceChildren(...rows.map(r=>{const tr=make("tr");tr.classList.toggle("selected-row",r.key===row.key);const methodCell=make("td"),button=make("button",names[r.method]);button.setAttribute("aria-label",`Inspect ${names[r.method]} ${r.k} features`);button.onclick=()=>{method=r.method;k=r.k;updateK();render();};methodCell.append(button);if(cvSelected(r))methodCell.append(make("span","CV Selected","cv-label"));tr.append(methodCell,make("td",r.k),make("td",r.features.join(" · "),"features-cell"),make("td",`${fmt(r.cv_rmse)} ± ${fmt(r.cv_rmse_std)}`),make("td",fmt(r.test.rmse)),make("td",fmt(r.test.r2,4)));return tr;}));
    const terms=row.model.terms.map(t=>`${t.coefficient>=0?"+":"−"} ${fmt(Math.abs(t.coefficient),4)} × ${t.kind==="category"?`[${t.source} = ${t.value}]`:t.source}`);text("equation",`${data.metadata.target} = ${fmt(row.model.intercept,4)} ${terms.join(" ")}`);
    $("coefficient-body").replaceChildren(...stats.coefficients.map(c=>{const tr=make("tr");tr.append(make("td",c.predictor),make("td",fmt(c.estimate,4)),make("td",fmt(c.std_error,4)),make("td",fmt(c.t_stat,3)),make("td",pvalue(c.p_value)),make("td",c.ci95?`[${fmt(c.ci95[0],4)}, ${fmt(c.ci95[1],4)}]`:"—"),make("td",c.vif_infinite?"∞":fmt(c.vif)));return tr;}));
    data.input_schema.forEach((schema,i)=>$(`input-${i}`).parentElement.classList.toggle("inactive",!row.model.terms.some(t=>t.source===schema.name)));
    const residuals=[...row.predictions].sort((a,b)=>b.absolute_error-a.absolute_error).slice(0,10);
    text("diagnostic-note",`Top ${residuals.length} of ${row.predictions.length} held-out samples, sorted by |residual|. Above RMSE means absolute error > current Test RMSE; it is a descriptive flag, not an outlier test. / 殘差＝實際值−預測值；標記僅描述誤差大小。`);
    $("residual-body").replaceChildren(...residuals.map((p,i)=>{const tr=make("tr"),status=make("td");status.append(make("span",p.diagnostic,"status"+(p.diagnostic==="Above RMSE"?" high":"")));tr.append(make("td",i+1),make("td",`#${p.id}`),make("td",fmt(p.actual)),make("td",fmt(p.predicted)),make("td",`${p.residual>=0?"+":""}${fmt(p.residual)}`),make("td",fmt(p.absolute_error)),make("td",p.relative_error_percent===null?"—":fmt(p.relative_error_percent)+"%"),status);return tr;}));
    chart("actual-chart",[{name:"Held-out Test Set",zh:"保留測試集",points:row.predictions.map(p=>[p.actual,p.predicted])}],{xLabel:"Actual Value",xZh:"實際值",yLabel:"Predicted Value",yZh:"預測值",scatter:true,reference:"diagonal"});
    chart("residual-chart",[{name:"Residual",zh:"實際值減預測值",points:row.predictions.map(p=>[p.predicted,p.residual])}],{xLabel:"Predicted Value",xZh:"預測值",yLabel:"Residual",yZh:"殘差",scatter:true,reference:"zero"});
    if(predictionUrl)URL.revokeObjectURL(predictionUrl);const csv="id,actual,predicted,residual,absolute_error,relative_error_percent,diagnostic\n"+row.predictions.map(p=>[p.id,p.actual,p.predicted,p.residual,p.absolute_error,p.relative_error_percent??"",p.diagnostic].join(",")).join("\n");predictionUrl=URL.createObjectURL(new Blob([csv],{type:"text/csv;charset=utf-8"}));$("download-predictions").href=predictionUrl;$("download-predictions").download=`${data.metadata.name}-${row.key}-predictions.csv`;
    predict();
  }
  $("dataset").onchange=e=>setDataset(Number(e.target.value));
  $("feature-k").onchange=e=>{k=Number(e.target.value);render();};
  document.querySelectorAll("[data-method]").forEach(button=>button.onclick=()=>{method=button.dataset.method;comparison=false;updateK();render();});
  $("compare-all").onclick=()=>{comparison=!comparison;if(method==="baseline")method="lasso";updateK();render();};
  $("reset-inputs").onclick=()=>{renderInputs();render();};
  setDataset(0);
})();
