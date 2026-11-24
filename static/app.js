const form=document.getElementById('analysisForm');
const dataFile=document.getElementById('dataFile');
const planFile=document.getElementById('planFile');
const targetCol=document.getElementById('targetCol');
const inspectInfo=document.getElementById('inspectInfo');
const loading=document.getElementById('loading');
const runState=document.getElementById('runState');
const sidebar=document.getElementById('sidebar');
const landingView=document.getElementById('landingView');
const appShell=document.getElementById('appShell');
const views=[...document.querySelectorAll('[data-view-panel]')];
const navItems=[...document.querySelectorAll('.nav-item[data-view]')];
let latestResult=null;
let repseudoSource={type:null,jobId:null,file:null,profile:null};
let progressPoll=null;
let progressGeneration=0;

const esc=v=>String(v??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const pct=v=>`${(Number(v||0)*100).toFixed(2)}%`;
const gradeLabel=g=>({H:'고위험',M:'중위험',L:'저위험'}[String(g||'').toUpperCase()]||'-');
const gradeClass=g=>`grade-text-${String(g||'').toUpperCase()}`;
const rawRiskLabel=v=>({'고위험':'고위험','위험':'고위험','중간':'중위험','안심':'저위험'}[String(v||'')]||String(v||'-'));

function setState(text,cls=''){runState.textContent=text;runState.className=`run-state ${cls}`;}
function createJobId(){
  if(globalThis.crypto?.randomUUID)return crypto.randomUUID().replaceAll('-','');
  return Array.from({length:32},()=>Math.floor(Math.random()*16).toString(16)).join('');
}
function showProgress(progress){
  loading.querySelector('b').textContent=progress.title||'분석을 진행하고 있습니다';
  loading.querySelector('p').textContent=progress.description||'현재 분석 단계를 확인하고 있습니다.';
  setState(progress.title||'분석 중',progress.state==='error'?'error':progress.state==='completed'?'done':'running');
}
function startProgressPolling(url,initial){
  stopAnalysisProgress();
  const generation=progressGeneration;
  showProgress(initial);
  const poll=async()=>{
    try{
      const response=await fetch(url,{cache:'no-store'});
      if(generation!==progressGeneration)return;
      if(response.ok){
        const progress=await response.json();
        if(progress.ok)showProgress(progress);
      }
    }catch(_error){}
    if(generation===progressGeneration)progressPoll=setTimeout(poll,700);
  };
  poll();
}
function startAnalysisProgress(jobId){
  startProgressPolling(`/api/progress/${encodeURIComponent(jobId)}`,{
    title:'분석 자료를 준비하고 있습니다',description:'업로드한 데이터셋과 활용 계획서를 확인합니다.'
  });
}
function stopAnalysisProgress(){progressGeneration+=1;if(progressPoll){clearTimeout(progressPoll);progressPoll=null;}}
function setView(name){
  views.forEach(v=>{v.classList.toggle('hidden',v.dataset.viewPanel!==name);v.classList.toggle('active',v.dataset.viewPanel===name)});
  navItems.forEach(n=>n.classList.toggle('active',n.dataset.view===name));
  const labels={risk:['1단계','가명정보 위험 평가'],repseudonymization:['2단계','재가명처리'],comparison:['3단계','재가명처리본 위험도 평가'],overview:['분석 결과','통합 분석 개요'],purpose:['상세 판정','목적 적합성'],identification:['상세 분석','식별 위험'],aggregation:['상세 분석','집계 위험'],exposure:['상세 분석','노출 위험']};
  document.getElementById('topSectionLabel').textContent=labels[name]?.[0]||'';
  document.getElementById('topSectionTitle').textContent=labels[name]?.[1]||'';
  sidebar.classList.remove('open');
  window.scrollTo({top:0,behavior:'smooth'});
}
function openWorkspace(name='risk'){
  landingView.classList.add('hidden');
  appShell.classList.remove('hidden');
  setView(name==='repseudonymization'?'repseudonymization':'risk');
}
function openLanding(){
  appShell.classList.add('hidden');
  landingView.classList.remove('hidden');
  sidebar.classList.remove('open');
  window.scrollTo({top:0,behavior:'smooth'});
}
navItems.forEach(btn=>btn.addEventListener('click',()=>{if(!btn.disabled)setView(btn.dataset.view)}));
document.querySelectorAll('[data-open-workspace]').forEach(btn=>btn.addEventListener('click',()=>openWorkspace(btn.dataset.openWorkspace)));
document.querySelectorAll('[data-landing-home],.landing-return').forEach(btn=>btn.addEventListener('click',openLanding));
document.querySelectorAll('[data-open-detail]').forEach(btn=>btn.addEventListener('click',()=>setView(btn.dataset.openDetail)));
document.querySelectorAll('.back-button').forEach(btn=>btn.addEventListener('click',()=>setView(btn.dataset.view)));
document.getElementById('mobileMenu').addEventListener('click',()=>sidebar.classList.toggle('open'));
document.getElementById('rerunBtn').addEventListener('click',()=>setView('risk'));

const revealItems=[...document.querySelectorAll('.landing-reveal')];
if('IntersectionObserver' in window){
  const revealObserver=new IntersectionObserver(entries=>entries.forEach(entry=>{
    if(entry.isIntersecting){entry.target.classList.add('is-visible');revealObserver.unobserve(entry.target)}
  }),{threshold:.14});
  revealItems.forEach(item=>revealObserver.observe(item));
}else revealItems.forEach(item=>item.classList.add('is-visible'));

dataFile.addEventListener('change',()=>{
  document.getElementById('dataFileName').textContent=dataFile.files[0]?.name||'분석할 데이터셋을 선택해 주세요';
  targetCol.innerHTML='<option value="">먼저 컬럼을 확인해 주세요</option>';
  inspectInfo.classList.add('hidden');
});
planFile.addEventListener('change',()=>document.getElementById('planFileName').textContent=planFile.files[0]?.name||'목적 적합성 판단에 사용할 자료를 선택해 주세요');

async function inspect(){
  if(!dataFile.files[0]){alert('데이터셋을 먼저 선택해 주세요.');return;}
  const fd=new FormData();fd.append('data_file',dataFile.files[0]);setState('컬럼 확인 중','running');
  try{
    const r=await fetch('/api/inspect',{method:'POST',body:fd});const d=await r.json();
    if(!r.ok||!d.ok)throw new Error(d.message||'컬럼을 확인하지 못했습니다.');
    targetCol.innerHTML='<option value="">타깃 컬럼을 선택해 주세요</option>'+d.columns.map(c=>`<option value="${esc(c.name)}">${esc(c.name)} (${esc(c.dtype)}, 고유값 ${c.unique})</option>`).join('');
    inspectInfo.classList.remove('hidden');
    inspectInfo.innerHTML=`<span>레코드<b>${d.rows.toLocaleString()}건</b></span><span>전체 컬럼<b>${d.columns.length}개</b></span><span>이진 타깃 후보<b>${d.target_candidates.length}개</b></span>`;
    setState('컬럼 확인 완료','done');
  }catch(e){setState('확인 실패','error');alert(e.message)}
}
document.getElementById('inspectBtn').addEventListener('click',inspect);

form.addEventListener('submit',async e=>{
  e.preventDefault();
  if(!dataFile.files[0]){alert('데이터셋을 선택해 주세요.');return;}
  if(!planFile.files[0]){alert('활용 계획서 PDF를 선택해 주세요.');return;}
  if(!targetCol.value){alert('데이터셋 컬럼 확인 후 집계 분석 타깃 컬럼을 선택해 주세요.');return;}
  const formData=new FormData(form);const clientJobId=createJobId();formData.append('client_job_id',clientJobId);
  loading.classList.remove('hidden');startAnalysisProgress(clientJobId);
  try{
    const r=await fetch('/api/analyze',{method:'POST',body:formData});const d=await r.json();
    if(!r.ok||!d.ok)throw new Error(d.message||'분석을 완료하지 못했습니다.');
    renderAll(d);
    document.querySelectorAll('.result-nav').forEach(x=>x.disabled=false);
    setState(d.pipeline_halted?'목적 적합성 반려':'분석 완료',d.pipeline_halted?'error':'done');
    setView('overview');
  }catch(err){setState('분석 실패','error');alert(err.message)}finally{stopAnalysisProgress();loading.classList.add('hidden')}
});

function renderAll(d){
  latestResult=d;
  localStorage.setItem('priviewLatestJobId',d.job_id);
  const ds=d.dataset||{};
  document.getElementById('datasetSummary').textContent=`${ds.filename||'데이터셋'}, ${Number(ds.rows||0).toLocaleString()}개 레코드, ${ds.columns||0}개 컬럼`;
  document.getElementById('summaryRows').textContent=`${Number(ds.rows||0).toLocaleString()}건`;
  document.getElementById('summaryColumns').textContent=`${ds.columns||0}개`;
  document.getElementById('summaryPurpose').textContent=d.gate?.decision||'-';
  const report=document.getElementById('reportBtn');
  if(d.report?.available){report.href=d.report.docx_url;report.classList.remove('disabled')}else{report.href='#';report.classList.add('disabled')}
  renderIntegrated(d);
  renderPurpose(d.gate||{});
  renderIdentification(d.modules?.identification||{},d.plots?.identification||[],d.pipeline_halted);
  renderAggregation(d.modules?.aggregation||{},d.plots?.aggregation||[],d.pipeline_halted);
  renderExposure(d.modules?.exposure||{},d.plots?.exposure||[],d.pipeline_halted);
  const grade=String(d.integrated?.grade||'').toUpperCase();
  document.getElementById('repseudoNext').classList.toggle('hidden',d.pipeline_halted||grade==='L'||!grade);
  document.getElementById('riskSourceStatus').textContent=`최근 결과 ${d.job_id.slice(0,8)}, 통합 위험도 ${d.integrated?.grade_label||gradeLabel(grade)}`;
  document.getElementById('repseudoRiskGrade').textContent=d.integrated?.grade_label||gradeLabel(grade)||'결과 확인 전';
}

function renderIntegrated(d){
  const el=document.getElementById('integratedGrade');
  const evidence=document.getElementById('integratedEvidence');evidence.classList.add('hidden');evidence.innerHTML='';
  if(d.pipeline_halted){
    el.textContent='중단';el.className='decision-reject';
    document.getElementById('overallTitle').textContent='후속 위험 분석을 중단했습니다';
    document.getElementById('overallMessage').textContent='목적 적합성이 반려되어 식별 위험, 집계 위험 및 노출 위험을 분석하지 않았습니다.';
    return;
  }
  const i=d.integrated;
  if(!i){
    el.textContent='확인';el.className='';
    document.getElementById('overallTitle').textContent='통합 위험도를 확인할 수 없습니다';
    document.getElementById('overallMessage').textContent='식별 위험, 집계 위험 및 노출 위험의 실행 결과를 확인해 주세요.';
    return;
  }
  el.textContent=i.grade_label||gradeLabel(i.grade);el.className=gradeClass(i.grade);
  document.getElementById('overallTitle').textContent=`통합 위험도는 ${i.grade_label||gradeLabel(i.grade)}입니다`;
  document.getElementById('overallMessage').textContent=i.rationale||'세 위험 축의 등급 중 가장 높은 등급으로 통합했습니다.';
  const componentCards=(i.components||[]).map(component=>`<article><span>${esc(component.label)}</span><strong class="${gradeClass(component.grade)}">${esc(component.grade_label||gradeLabel(component.grade))}</strong><p><b>${esc(component.top_attribute||'-')}</b></p><small>${esc(component.basis||'모듈 자체 판정')}</small></article>`).join('');
  evidence.innerHTML=`<div class="section-heading"><div><h3>통합 위험도 산출 근거</h3><p>세 모듈의 데이터셋 등급 중 가장 높은 등급을 적용합니다. 각 모듈의 판정 근거는 상세 화면에서 확인할 수 있습니다.</p></div></div><div class="integrated-component-grid">${componentCards}</div>`;evidence.classList.remove('hidden');
}

function setCardStatus(id,text,ok){const el=document.getElementById(id);el.textContent=text;el.style.color=ok?'var(--success)':'var(--danger)';}
function listHtml(arr,empty='해당 항목이 없습니다.'){return (arr||[]).length?`<ul class="plain-list">${arr.map(x=>`<li>${esc(x)}</li>`).join('')}</ul>`:`<span style="font-size:11px;color:var(--muted)">${empty}</span>`}
function plotGrid(plots){
  const items=(plots||[]).slice(0,4);
  if(!items.length)return '<div class="halted-box">생성된 시각화가 없습니다.</div>';
  return `<div class="plot-grid">${items.map(p=>`<figure class="plot-card"><img src="${esc(p.url)}" alt="${esc(p.title)}"><figcaption><b>${esc(p.title)}</b><span>${esc(p.caption)}</span></figcaption></figure>`).join('')}</div>`;
}
function moduleError(contentId,gradeId,metricId,statusId,overviewGradeId,overviewReasonId,m){
  document.getElementById(gradeId).textContent='확인 필요';document.getElementById(metricId).textContent=m.hint||'실행 환경을 확인해 주세요.';setCardStatus(statusId,'실행 오류',false);
  document.getElementById(overviewGradeId).textContent='확인 필요';document.getElementById(overviewGradeId).className='grade-text-H';document.getElementById(overviewReasonId).textContent=m.message||'모듈 실행 중 오류가 발생했습니다.';
  document.getElementById(contentId).className='detail-content';document.getElementById(contentId).innerHTML=`<div class="error-box"><b>${esc(m.message||'모듈 실행 중 오류가 발생했습니다.')}</b>${esc(m.hint||'')}</div>`;
}
function haltedModule(contentId,gradeId,metricId,statusId,overviewGradeId,overviewReasonId){
  document.getElementById(gradeId).textContent='미실행';document.getElementById(gradeId).className='';
  document.getElementById(metricId).textContent='목적 적합성 반려로 분석하지 않았습니다.';setCardStatus(statusId,'미실행',false);
  document.getElementById(overviewGradeId).textContent='미실행';document.getElementById(overviewGradeId).className='';
  document.getElementById(overviewReasonId).textContent='목적 적합성 반려로 후속 분석을 중단했습니다.';
  document.getElementById(contentId).className='detail-content';
  document.getElementById(contentId).innerHTML='<div class="halted-box">목적 적합성이 통과된 경우에만 이 분석을 실행합니다.</div>';
}

function renderPurpose(m){
  if(m.status!=='success'){moduleError('purposeContent','purposeDecision','purposeMetric','purposeStatus','purposeOverviewGrade','purposeOverviewReason',m);return;}
  const pass=m.decision==='통과';
  ['purposeDecision','purposeOverviewGrade'].forEach(id=>{const el=document.getElementById(id);el.textContent=m.decision;el.className=pass?'decision-pass':'decision-reject'});
  document.getElementById('purposeMetric').textContent=`선택 목적은 ${m.purpose_label||'-'}입니다.`;setCardStatus('purposeStatus','판정 완료',pass);
  document.getElementById('purposeOverviewReason').textContent=pass?`${m.purpose_label||'선택 목적'}가 활용 계획에 적합함을 확인했습니다.`:(m.rejection_reasons||[]).join(', ')||'반려 사유를 확인해 주세요.';
  const danger=Object.entries(m.danger_keywords||{}).flatMap(([k,v])=>(v||[]).map(x=>`${k}: ${x}`));
  const a=m.analysis_summary||{};
  const details=m.display_details||{},llm=details.llm||{};
  const criteria=details.criteria||[];
  const total=Number(a.검토항목수||criteria.length||0),satisfied=Number(a.충족항목수??Math.max(0,total-(m.unmet_items||[]).length));
  const rate=Math.max(0,Math.min(100,Number(a.충족률||0)));
  const morph=Number(a.형태소수||0),dangerCount=Number(a.위험표현수||0),activityMax=Math.max(morph,dangerCount,1);
  const criteriaHtml=criteria.length?criteria.map((item,index)=>`<article class="purpose-criterion"><header><span>${String(index+1).padStart(2,'0')}</span><h4>${esc(item.display_title||item.criterion)}</h4><em class="${item.status==='미충족'?'fail':''}">${esc(item.status)}</em></header><div class="purpose-plan-evidence"><strong>계획서에서 확인한 내용</strong><p>${esc(item.evidence||'관련 내용을 확인하지 못했습니다.').replaceAll('\n','<br>')}</p></div></article>`).join(''):'<p class="purpose-empty">표시할 세부 기준이 없습니다.</p>';
  document.getElementById('purposeContent').className='detail-content';
  document.getElementById('purposeContent').innerHTML=`<div class="decision-layout"><div class="decision-card"><span>목적 적합성 판정</span><strong class="${pass?'decision-pass':'decision-reject'}">${esc(m.decision)}</strong><p>${pass?`${esc(m.purpose_label||'선택 목적')}와 활용 계획서의 적합성을 확인했습니다.`:'활용 계획서가 선택 목적의 기준을 충족하지 않아 후속 위험 분석을 중단했습니다.'}</p></div><div class="purpose-analysis"><div><span>검토 항목</span><b>${satisfied}/${total}개 충족</b></div><div><span>형태소 분석</span><b>${morph.toLocaleString()}개</b></div><div><span>위험 표현</span><b>${dangerCount}개</b></div><div><span>실행 판정 방식</span><b>${esc(llm.mode||'규칙 기반 판정')}</b></div></div></div><div class="purpose-visuals"><figure><figcaption><b>검토 항목 충족 현황</b><span>선택 목적에 적용되는 기준을 확인한 결과입니다.</span></figcaption><div class="purpose-donut" style="--purpose-rate:${rate}"><strong>${rate.toFixed(1)}%</strong><small>${satisfied}개 충족</small></div></figure><figure><figcaption><b>계획서 텍스트 분석량</b><span>판정에 사용한 형태소와 탐지된 위험 표현의 수입니다.</span></figcaption><div class="purpose-bars"><div><span>형태소</span><i><em style="width:${(morph/activityMax*100).toFixed(1)}%"></em></i><b>${morph.toLocaleString()}</b></div><div><span>위험 표현</span><i><em class="danger" style="width:${(dangerCount/activityMax*100).toFixed(1)}%"></em></i><b>${dangerCount}</b></div></div></figure></div><section class="purpose-review"><div class="section-heading"><div><h3>계획서 기준별 판정 근거</h3><p>${esc(m.purpose_label||'선택 목적')}에 적용되는 기준과 계획서 원문에서 확인한 내용입니다.</p></div></div><div class="purpose-criteria">${criteriaHtml}</div></section><section class="purpose-review purpose-review-summary"><div><h3>LLM 문맥 판별</h3><p>${esc(llm.description||'현재 실행 판정은 규칙 기반으로 수행됩니다.')}</p><small>${llm.api_call_active?'OpenAI GPT 보조 판정 사용':'규칙 기반 판정 사용'}</small></div><div><h3>추가 확인 결과</h3>${listHtml([...(m.unmet_items||[]),...danger], '미충족 항목이나 위험 표현이 확인되지 않았습니다.')}</div><div><h3>최종 판정</h3><p>${pass?'계획서의 활용 목적과 판단 기준이 적합하여 통과했습니다.':'미충족 기준 또는 위험 표현이 확인되어 반려했습니다.'}</p></div></section>`;
}

function renderIdentification(m,plots,halted){
  if(halted){haltedModule('idContent','idGrade','idMetric','idStatus','idOverviewGrade','idOverviewReason');return;}
  if(m.status!=='success'){moduleError('idContent','idGrade','idMetric','idStatus','idOverviewGrade','idOverviewReason',m);return;}
  const grade=m.dataset_grade||'L',label=m.dataset_grade_label||gradeLabel(grade);
  ['idGrade','idOverviewGrade'].forEach(id=>{const el=document.getElementById(id);el.textContent=label;el.className=gradeClass(grade)});
  document.getElementById('idMetric').textContent=`유일성 ${pct(m.uniqueness)}, 정확 매칭률 ${pct(m.correctness)}`;setCardStatus('idStatus','분석 완료',true);
  document.getElementById('idOverviewReason').textContent=`상위 조합 ${m.top_combinations?.length||0}개와 속성별 등급을 확인했습니다.`;
  const counts=m.attribute_grade_counts||{};
  const attrRows=(m.attributes||[]).map(x=>`<tr><td><b>${esc(x.attribute)}</b></td><td>${Number(x.individual_risk_score||0).toFixed(2)}</td><td class="${gradeClass(x.individual_grade)}">${gradeLabel(x.individual_grade)}</td><td>${pct(x.top_appearance_ratio)}</td><td class="${gradeClass(x.combo_grade)}">${gradeLabel(x.combo_grade)}</td><td class="${gradeClass(x.final_grade)}"><b>${gradeLabel(x.final_grade)}</b></td></tr>`).join('');
  const comboRows=(m.top_combinations||[]).slice(0,10).map(x=>`<tr><td><b>${esc((x.combination||[]).join(' + '))}</b></td><td>${x.size}</td><td>${pct(x.uniqueness)}</td><td>${pct(x.correctness)}</td></tr>`).join('');
  document.getElementById('idContent').className='detail-content';
  document.getElementById('idContent').innerHTML=`<div class="metric-row"><div class="metric"><span>데이터셋 등급</span><b class="${gradeClass(grade)}">${esc(label)}</b></div><div class="metric"><span>유일성</span><b>${pct(m.uniqueness)}</b></div><div class="metric"><span>정확 매칭률</span><b>${pct(m.correctness)}</b></div><div class="metric"><span>조합 분석 범위</span><b>1~${m.max_combo_size}개</b></div></div><div class="subsection"><h3>식별 위험 시각화</h3>${plotGrid(plots)}</div><div class="subsection" style="margin-top:28px"><h3>속성별 최종 식별 등급</h3><p style="font-size:11px;color:var(--muted)">고위험 ${counts.H||0}개, 중위험 ${counts.M||0}개, 저위험 ${counts.L||0}개입니다.</p><table class="data-table"><thead><tr><th>컬럼</th><th>개별 위험 점수</th><th>개별 등급</th><th>상위 조합 등장률</th><th>조합 등급</th><th>최종 등급</th></tr></thead><tbody>${attrRows}</tbody></table></div><div class="subsection" style="margin-top:28px"><h3>상위 식별 위험 조합</h3><table class="data-table id-combination-table"><thead><tr><th>컬럼 조합</th><th>크기</th><th>유일성</th><th>정확 매칭률</th></tr></thead><tbody>${comboRows}</tbody></table></div>`;
}

function renderAggregation(m,plots,halted){
  if(halted){haltedModule('aggContent','aggGrade','aggMetric','aggStatus','aggOverviewGrade','aggOverviewReason');return;}
  if(m.status!=='success'){moduleError('aggContent','aggGrade','aggMetric','aggStatus','aggOverviewGrade','aggOverviewReason',m);return;}
  const grade=m.dataset_grade||'L',label=m.dataset_grade_label||gradeLabel(grade);
  ['aggGrade','aggOverviewGrade'].forEach(id=>{const el=document.getElementById(id);el.textContent=label;el.className=gradeClass(grade)});
  document.getElementById('aggMetric').textContent=`타깃 ${m.target_col}, ${Number(m.total_combinations||0).toLocaleString()}개 조합을 분석했습니다.`;setCardStatus('aggStatus','분석 완료',true);
  document.getElementById('aggOverviewReason').textContent=`타깃 ${m.target_col||'-'} 기준으로 ${Number(m.total_combinations||0).toLocaleString()}개 조합을 확인했습니다.`;
  const hml=m.hml_distribution||{},hp=hml.H?.pct||0,mp=hml.M?.pct||0,lp=hml.L?.pct||0;
  const rows=(m.top_combinations||[]).slice(0,12).map(x=>`<tr><td><b>${esc(String(x.Feature_Combination||'').split('+').join(' + '))}</b></td><td>${(Number(x.Mean_Accuracy||0)*100).toFixed(2)}%</td><td>${(Number(x.Mean_F1||0)*100).toFixed(2)}%</td><td>${esc(rawRiskLabel(x.Final_Risk))}</td></tr>`).join('');
  document.getElementById('aggContent').className='detail-content';
  document.getElementById('aggContent').innerHTML=`<div class="metric-row"><div class="metric"><span>데이터셋 집계 등급</span><b class="${gradeClass(grade)}">${esc(label)}</b></div><div class="metric"><span>민감정보 타깃</span><b>${esc(m.target_col)}</b></div><div class="metric"><span>분석 조합 수</span><b>${Number(m.total_combinations||0).toLocaleString()}</b></div><div class="metric"><span>사용 모델</span><b>${(m.models||[]).length}종</b></div></div><div class="subsection"><h3>집계 위험 시각화</h3>${plotGrid(plots)}</div><div class="subsection aggregation-distribution" style="margin-top:28px"><h3>등급 분포</h3><div class="distribution"><i class="grade-h" style="width:${hp}%"></i><i class="grade-m" style="width:${mp}%"></i><i class="grade-l" style="width:${lp}%"></i></div><div class="legend"><span><i class="grade-h"></i>고위험 ${hp}%</span><span><i class="grade-m"></i>중위험 ${mp}%</span><span><i class="grade-l"></i>저위험 ${lp}%</span></div></div>${aggAttributeTable(m)}<div class="subsection"><h3>상위 집계 위험 조합</h3><table class="data-table aggregation-table"><thead><tr><th>특성 조합</th><th>평균 정확도</th><th>평균 F1</th><th>최종 위험</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

function aggAttributeTable(m){
  if(m.attribute_grade_method!=='label_share'||!m.attribute_scores)return '';
  const grades=m.attribute_grades||{};
  const rows=Object.entries(m.attribute_scores).sort((a,b)=>b[1]-a[1]).map(([attr,share])=>`<tr><td><b>${esc(attr)}</b></td><td>${(Number(share||0)*100).toFixed(1)}%</td><td class="${gradeClass(grades[attr])}"><b>${gradeLabel(grades[attr])}</b></td></tr>`).join('');
  return `<div class="subsection"><h3>속성별 H 비율</h3><p class="required-column-note">그 속성이 들어간 조합 중 위험 이상 판정을 받은 조합의 비율입니다. 등급은 H, M, L 비율 중 가장 큰 쪽이며, 같은 등급이어도 H 비율이 높을수록 우선 처리 대상입니다.</p><table class="data-table"><thead><tr><th>속성</th><th>H 비율</th><th>속성 등급</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

function renderExposure(m,plots,halted){
  if(halted){haltedModule('exposureContent','exposureGrade','exposureMetric','exposureStatus','exposureOverviewGrade','exposureOverviewReason');return;}
  if(m.status!=='success'){moduleError('exposureContent','exposureGrade','exposureMetric','exposureStatus','exposureOverviewGrade','exposureOverviewReason',m);return;}
  const grade=m.dataset_grade||'L',label=m.dataset_grade_label||gradeLabel(grade);
  ['exposureGrade','exposureOverviewGrade'].forEach(id=>{const el=document.getElementById(id);el.textContent=label;el.className=gradeClass(grade)});
  document.getElementById('exposureMetric').textContent=`민감속성 ${(m.sensitive_columns||[]).length}개, 임계 초과 ${m.n_exceed||0}개를 확인했습니다.`;setCardStatus('exposureStatus','분석 완료',true);
  document.getElementById('exposureOverviewReason').textContent=`속성별 노출 위험 임계 초과 비율은 ${(Number(m.rho||0)*100).toFixed(1)}%입니다.`;
  const categoryLabels=m.category_labels||{};
  const actionRows=(m.top_actions||[]).map(item=>`<tr><td>${item.rank}</td><td><b>${esc(item.attribute)}</b></td><td>${(Number(item.risk||0)*100).toFixed(1)}%</td></tr>`).join('');
  const matrixRows=(m.exposure_matrix||[]).map(row=>`<tr><td><b>${esc(row.attribute)}</b><small class="cell-note">${esc(categoryLabels[row.category]||row.category||'-')}</small></td><td>${Number(row.emd||0).toFixed(3)}</td><td>${Number(row.coverage||0).toFixed(3)}</td><td>${Number(row.combo||0).toFixed(3)}</td><td>${Number(row.ee_match||0).toFixed(3)}</td><td>${Number(row.granularity||0).toFixed(3)}</td></tr>`).join('');
  const tierKo={CONFIRMED:'확정',PROBABLE:'유력',WEAK:'검토 필요',NONE:'없음'};
  const detectionRows=(m.detection||[]).filter(row=>row.is_sensitive||row.review_required).map(row=>`<tr><td><b>${esc(row.column)}</b></td><td>${esc(row.category_ko||categoryLabels[row.category]||row.category||'-')}</td><td>${row.track==='A'?'법정':row.track==='B'?'비법정':'-'}</td><td>${esc(tierKo[row.tier]||row.tier||'-')}</td><td>${row.review_required?'사용자 확인 필요':'자동 판정 완료'}</td></tr>`).join('');
  document.getElementById('exposureContent').className='detail-content';
  document.getElementById('exposureContent').innerHTML=`<div class="metric-row"><div class="metric"><span>데이터셋 노출 등급</span><b class="${gradeClass(grade)}">${esc(label)}</b></div><div class="metric"><span>민감속성</span><b>${(m.sensitive_columns||[]).length}개</b></div><div class="metric"><span>임계 초과 비율</span><b>${(Number(m.rho||0)*100).toFixed(1)}%</b></div><div class="metric"><span>사용자 확인 필요</span><b>${(m.review_columns||[]).length}개</b></div></div><div class="subsection"><h3>노출 위험 시각화</h3>${plotGrid(plots)}</div><div class="subsection" style="margin-top:28px"><h3>민감정보 탐지 결과</h3><table class="data-table"><thead><tr><th>컬럼</th><th>정보 유형</th><th>구분</th><th>탐지 수준</th><th>판정 상태</th></tr></thead><tbody>${detectionRows}</tbody></table></div>`;
}

const reDataFile=document.getElementById('reDataFile');
const repseudoSettings=document.getElementById('repseudoSettings');
const strategyRows=document.getElementById('strategyRows');
const repseudoResult=document.getElementById('repseudoResult');

function parameterFor(method){return ['rare_group','numeric_bin'].includes(method)?5:''}
function methodPicker(methods,selected,column){
  const current=methods[selected]||Object.values(methods)[0]||'처리 방법 선택';
  const options=Object.entries(methods).map(([value,label])=>`<button type="button" class="method-picker-option ${value===selected?'selected':''}" data-value="${esc(value)}" role="option" aria-selected="${value===selected}">${esc(label)}</button>`).join('');
  return `<div class="method-picker"><button type="button" class="method-picker-trigger" aria-label="${esc(column)} 처리 방법" aria-expanded="false"><span>${esc(current)}</span><i></i></button><div class="method-picker-menu hidden" role="listbox">${options}</div><input type="hidden" class="strategy-method" value="${esc(selected)}"></div>`;
}
function updateStrategyParameter(row,method){
  const input=row.querySelector('.strategy-param');const value=parameterFor(method);
  input.disabled=value==='';input.value=value;
}
function renderStrategyRows(profile){
  const methods=profile.methods||{};
  document.getElementById('repseudoDatasetInfo').textContent=`${profile.filename}, ${Number(profile.rows).toLocaleString()}개 레코드, ${profile.column_count}개 컬럼`;
  strategyRows.innerHTML=(profile.columns||[]).map((column,index)=>{
    const checked=column.recommended?'checked':'';
    return `<div class="strategy-row" data-column="${esc(column.name)}"><span><input type="checkbox" class="strategy-check" ${checked} aria-label="${esc(column.name)} 선택"></span><b>${esc(column.name)}${column.reason?`<small>${esc(column.reason)}</small>`:''}</b><em>${esc(column.dtype)}, 고유값 ${column.unique}</em>${methodPicker(methods,column.suggested_method,column.name)}<input class="strategy-param" type="number" min="2" max="20" value="${parameterFor(column.suggested_method)}" ${parameterFor(column.suggested_method)===''?'disabled':''} aria-label="${esc(column.name)} 설정값"></div>`;
  }).join('');
  strategyRows.querySelectorAll('.method-picker-trigger').forEach(trigger=>trigger.addEventListener('click',event=>{
    event.stopPropagation();const picker=trigger.closest('.method-picker');const menu=picker.querySelector('.method-picker-menu');
    strategyRows.querySelectorAll('.method-picker-menu').forEach(item=>{if(item!==menu)item.classList.add('hidden')});
    const opening=menu.classList.contains('hidden');menu.classList.toggle('hidden',!opening);trigger.setAttribute('aria-expanded',String(opening));
  }));
  strategyRows.querySelectorAll('.method-picker-option').forEach(option=>option.addEventListener('click',event=>{
    event.stopPropagation();const picker=option.closest('.method-picker'),row=option.closest('.strategy-row');
    picker.querySelector('.strategy-method').value=option.dataset.value;picker.querySelector('.method-picker-trigger span').textContent=option.textContent;
    picker.querySelectorAll('.method-picker-option').forEach(item=>{item.classList.toggle('selected',item===option);item.setAttribute('aria-selected',String(item===option))});
    picker.querySelector('.method-picker-menu').classList.add('hidden');picker.querySelector('.method-picker-trigger').setAttribute('aria-expanded','false');
    updateStrategyParameter(row,option.dataset.value);
  }));
  repseudoSettings.classList.remove('hidden');repseudoResult.classList.add('hidden');
}
document.addEventListener('click',()=>strategyRows.querySelectorAll('.method-picker-menu').forEach(menu=>menu.classList.add('hidden')));

async function loadRiskDataset(){
  const jobId=latestResult?.job_id||localStorage.getItem('priviewLatestJobId');
  if(!jobId){alert('먼저 위험 평가를 실행하거나 별도 데이터셋을 선택해 주세요.');return;}
  setState('분석 데이터 불러오는 중','running');
  try{
    const response=await fetch(`/api/repseudonymize/prepare/${encodeURIComponent(jobId)}`);const profile=await response.json();
    if(!response.ok||!profile.ok)throw new Error(profile.message||'위험 평가 데이터를 불러오지 못했습니다.');
    repseudoSource={type:'risk',jobId,file:null,profile};renderStrategyRows(profile);
    document.getElementById('riskSourceStatus').textContent=`${profile.filename}, 추천 컬럼 ${profile.recommended_columns.length}개`;
    document.getElementById('repseudoRiskGrade').textContent=profile.integrated_grade_label||gradeLabel(profile.integrated_grade)||'결과 확인 전';
    setState('처리 방법 선택','done');
  }catch(error){setState('불러오기 실패','error');alert(error.message)}
}

async function inspectRepseudoUpload(file){
  const fd=new FormData();fd.append('data_file',file);setState('컬럼 확인 중','running');
  try{
    const response=await fetch('/api/repseudonymize/inspect',{method:'POST',body:fd});const profile=await response.json();
    if(!response.ok||!profile.ok)throw new Error(profile.message||'데이터셋을 확인하지 못했습니다.');
    repseudoSource={type:'upload',jobId:null,file,profile};renderStrategyRows(profile);setState('처리 방법 선택','done');
  }catch(error){setState('확인 실패','error');alert(error.message)}
}

document.getElementById('loadRiskResult').addEventListener('click',loadRiskDataset);
document.getElementById('resultToRepseudo').addEventListener('click',()=>{setView('repseudonymization');loadRiskDataset()});
reDataFile.addEventListener('change',()=>{
  const file=reDataFile.files[0];document.getElementById('reDataFileName').textContent=file?.name||'재가명처리할 파일 선택';
  if(file)inspectRepseudoUpload(file);
});
document.getElementById('selectRecommended').addEventListener('click',()=>{
  const recommended=new Set(repseudoSource.profile?.recommended_columns||[]);
  strategyRows.querySelectorAll('.strategy-row').forEach(row=>row.querySelector('.strategy-check').checked=recommended.has(row.dataset.column));
  if(!recommended.size)alert('위험 평가에서 추천된 컬럼이 없습니다. 처리할 컬럼을 직접 선택해 주세요.');
});

function renderRiskComparison(result){
  const comparison=result.comparison||{},content=document.getElementById('comparisonContent');
  const moduleRows=(comparison.modules||[]).map(row=>`<tr><td><b>${esc(row.label)}</b></td><td class="${gradeClass(row.before_grade)}">${row.before_grade?gradeLabel(row.before_grade):'평가 없음'}</td><td class="${gradeClass(row.after_grade)}">${row.after_grade?gradeLabel(row.after_grade):'재평가 실패'}</td><td>${row.reduction==null?'비교 불가':row.reduction>0?'등급 하락':row.reduction<0?'등급 상승':'등급 유지'}</td><td>${esc(row.before_attribute||'-')} → ${esc(row.after_attribute||'-')}</td></tr>`).join('');
  const attrRows=(comparison.improved_attributes||[]).map(row=>`<tr><td><b>${esc(row.attribute)}</b></td><td>${esc(row.module_label)}<br><small>${esc(row.metric_label||'모듈 수치')}</small></td><td>${Number(row.before_score||0).toFixed(3)}</td><td>${Number(row.after_score||0).toFixed(3)}</td><td><b>-${Number(row.reduction||0).toFixed(3)}</b></td></tr>`).join('');
  const processRows=(comparison.processing_summary||[]).map(row=>`<tr><td><b>${esc(row.column)}</b></td><td>${esc(row.method_label)}</td><td>${Number(row.changed_rows||0).toLocaleString()}건</td><td>${Number(row.before_unique||0).toLocaleString()}개 → ${Number(row.after_unique||0).toLocaleString()}개</td></tr>`).join('');
  content.className='detail-content';
  const improvementText=!comparison.comparable?'일부 모듈 재평가 실패':comparison.absolute_reduction>0?'통합 등급 하락':comparison.absolute_reduction<0?'통합 등급 상승':'통합 등급 유지';
  const reductionText='등급은 순서 척도이므로 감소율은 계산하지 않습니다.';
  content.innerHTML=`<div class="comparison-summary"><div><span>처리 전</span><strong class="${gradeClass(comparison.before_grade)}">${esc(comparison.before_grade_label||gradeLabel(comparison.before_grade))}</strong><small>통합 판정</small></div><i>→</i><div><span>처리 후</span><strong class="${gradeClass(comparison.after_grade)}">${esc(comparison.after_grade_label||gradeLabel(comparison.after_grade))}</strong><small>재평가 판정</small></div><div class="comparison-improvement"><span>전후 비교</span><b>${improvementText}</b><small>${reductionText}</small></div></div><section class="subsection"><h3>영역별 위험 등급 변화</h3><table class="data-table"><thead><tr><th>영역</th><th>처리 전</th><th>처리 후</th><th>판정 변화</th><th>대표 속성 변화</th></tr></thead><tbody>${moduleRows}</tbody></table></section><section class="subsection"><h3>수치가 낮아진 속성</h3>${attrRows?`<table class="data-table"><thead><tr><th>속성</th><th>영역</th><th>처리 전</th><th>처리 후</th><th>감소폭</th></tr></thead><tbody>${attrRows}</tbody></table>`:'<div class="halted-box">동일 속성에서 비교 가능한 수치 감소를 확인하지 못했습니다.</div>'}</section><section class="subsection"><h3>재가명처리 산출물</h3><table class="data-table"><thead><tr><th>처리 컬럼</th><th>처리 방법</th><th>변경 레코드</th><th>고유값 변화</th></tr></thead><tbody>${processRows}</tbody></table></section>`;
  content.insertAdjacentHTML('beforeend',`<p class="required-column-note">${esc(comparison.comparison_note||'영역별 등급과 속성별 수치를 함께 확인해 주세요.')}</p>`);
}

function renderRounds(comparison){
  const info=comparison.rounds;if(!info||!(info.rounds||[]).length)return '';
  const rows=info.rounds.map(r=>`<tr><td><b>${r.round}라운드</b></td><td>${(r.processed_columns||[]).map(esc).join(', ')||'-'}</td><td class="${gradeClass(r.grade_before)}">${r.grade_before?gradeLabel(r.grade_before):'-'}</td><td class="${gradeClass(r.grade_after)}">${r.grade_after?gradeLabel(r.grade_after):'재평가 실패'}</td><td>${r.attributes_above_target==null?'-':r.attributes_above_target+'개'}</td></tr>`).join('');
  return `<section class="subsection"><h3>라운드별 재가명처리 (목표: ${esc(gradeLabel(info.target_level))}, 최대 ${info.max_rounds}라운드)</h3><table class="data-table"><thead><tr><th>라운드</th><th>처리 컬럼</th><th>라운드 전 통합 등급</th><th>라운드 후 통합 등급</th><th>목표 초과 속성</th></tr></thead><tbody>${rows}</tbody></table><p class="required-column-note">멈춘 이유: ${esc(info.stop_reason||'-')}</p></section>`;
}

function renderUtilityTv(tv){
  if(!tv||tv.status!=='success'||!(tv.queries||[]).length)return '';
  const rows=tv.queries.map(q=>`<tr><td><b>${esc(q.column)}</b></td><td>${esc(q.query)}</td><td>${Number(q.tv).toFixed(3)}</td><td>${Number(q.allowed).toFixed(3)}<br><small>재표본 95% ${Number(q.bootstrap_q95).toFixed(3)} × √2</small></td><td class="${q.pass?'tv-pass':'tv-fail'}">${q.pass?'기준 이내':'기준 초과'}</td></tr>`).join('');
  return `<section class="subsection"><h3>유용성 (TV 분포 변화) — 참고용</h3><p class="required-column-note">${esc(tv.criterion)}. ${esc(tv.note)} 처리 컬럼 ${tv.n_columns}개 중 ${tv.n_columns_pass}개가 기준 이내입니다.</p><table class="data-table"><thead><tr><th>처리 컬럼</th><th>질의</th><th>TV</th><th>허용 폭</th><th>결과</th></tr></thead><tbody>${rows}</tbody></table></section>`;
}

async function runFollowupAssessment(result){
  if(!result.reassess_url){alert('위험 평가와 연결된 재가명처리 결과만 전후 비교할 수 있습니다.');return;}
  loading.classList.remove('hidden');
  startProgressPolling(`/api/repseudonymize/reassess/progress/${encodeURIComponent(result.task_id)}`,{
    title:'위험도 재평가를 준비하고 있습니다',description:'재가명처리본과 기존 위험 평가 기준을 확인합니다.'
  });
  try{
    const response=await fetch(result.reassess_url,{method:'POST'});const followup=await response.json();
    if(!response.ok||!followup.ok)throw new Error(followup.message||'재평가를 완료하지 못했습니다.');
    renderRiskComparison(followup);document.getElementById('comparisonNav').disabled=false;setState('전후 비교 완료','done');setView('comparison');
  }catch(error){setState('재평가 실패','error');alert(error.message)}finally{stopAnalysisProgress();loading.classList.add('hidden')}
}

document.getElementById('runRepseudo').addEventListener('click',async()=>{
  const strategies=[...strategyRows.querySelectorAll('.strategy-row')].filter(row=>row.querySelector('.strategy-check').checked).map(row=>({
    column:row.dataset.column,method:row.querySelector('.strategy-method').value,parameter:row.querySelector('.strategy-param').disabled?null:Number(row.querySelector('.strategy-param').value)
  }));
  if(!strategies.length){alert('재가명처리할 컬럼을 한 개 이상 선택해 주세요.');return;}
  const fd=new FormData();fd.append('strategies',JSON.stringify(strategies));fd.append('target_level',document.getElementById('targetLevel')?.value||'L');
  const clientTaskId=createJobId();fd.append('client_task_id',clientTaskId);
  if(repseudoSource.type==='risk')fd.append('job_id',repseudoSource.jobId);
  else if(repseudoSource.file)fd.append('data_file',repseudoSource.file);
  else{alert('처리할 데이터셋을 먼저 불러와 주세요.');return;}
  loading.classList.remove('hidden');
  startProgressPolling(`/api/repseudonymize/progress/${encodeURIComponent(clientTaskId)}`,{
    title:'재가명처리를 준비하고 있습니다',description:'선택한 컬럼과 처리 방법을 확인합니다.'
  });
  try{
    const response=await fetch('/api/repseudonymize',{method:'POST',body:fd});const result=await response.json();
    if(!response.ok||!result.ok)throw new Error(result.message||'재가명처리를 완료하지 못했습니다.');
    const rows=(result.summary||[]).map(item=>`<tr><td><b>${esc(item.column)}</b></td><td>${esc(item.method_label)}</td><td>${Number(item.changed_rows).toLocaleString()}건</td><td>${item.before_unique}개 → ${item.after_unique}개</td></tr>`).join('');
    repseudoResult.innerHTML=`<div class="repseudo-result-head"><div><h3>재가명처리를 완료했습니다.</h3><p>${result.processed_columns}개 컬럼을 처리했습니다. 위험도를 다시 평가해 개선 정도를 확인할 수 있습니다.</p></div><div class="repseudo-result-actions"><a class="primary-button" href="${esc(result.download_url)}" download>결과 파일 내려받기</a>${result.reassess_url?'<button type="button" class="secondary-button" id="recheckRisk">위험도 재평가</button>':''}</div></div><table class="data-table"><thead><tr><th>처리 컬럼</th><th>처리 방법</th><th>변경 레코드</th><th>고유값 변화</th></tr></thead><tbody>${rows}</tbody></table>`;
    document.getElementById('recheckRisk')?.addEventListener('click',()=>runFollowupAssessment(result));
    repseudoResult.classList.remove('hidden');setState('재가명처리 완료','done');
  }catch(error){setState('처리 실패','error');alert(error.message)}finally{stopAnalysisProgress();loading.classList.add('hidden')}
});

const storedJobId=localStorage.getItem('priviewLatestJobId');
if(storedJobId)document.getElementById('riskSourceStatus').textContent=`최근 위험 평가 결과 ${storedJobId.slice(0,8)}를 불러올 수 있습니다.`;
