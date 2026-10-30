'use client';
import { useEffect, useState } from 'react';
import AppShell from '../../components/AppShell';
import { CheckCircle2, Download, FileText } from 'lucide-react';
import type { AnalysisResult } from '../../lib/api';

export default function Reports(){
  const [analysis,setAnalysis]=useState<AnalysisResult|null>(null);
  useEffect(()=>{const raw=localStorage.getItem('hemovision-analysis'); if(raw) setAnalysis(JSON.parse(raw));},[]);
  return <AppShell><div className="page"><div className="heading"><div><small>WORKSPACE / REPORTS</small><h1>Analysis Reports</h1><p>Review and export the latest structured analysis summary.</p></div></div>{analysis ? <><div className="report"><span><FileText size={23}/></span><div><h3>{analysis.sample_id} <b><CheckCircle2 size={12}/> Complete</b></h3><p>Blood smear quantitative analysis • YOLO11n detection + Stage 2 RBC morphology</p><small>{analysis.total_cells.toLocaleString()} detected cells • {analysis.elapsed_seconds}s inference</small></div><button className="btn secondary" onClick={()=>window.print()}><Download size={14}/> Print / PDF</button></div><div className="empty" style={{minHeight:360,alignItems:'stretch',padding:24,textAlign:'left'}}><h2>Summary</h2><p>RBC: {analysis.counts.RBC} • WBC: {analysis.counts.WBC} • Platelet: {analysis.counts.Platelet}</p><h3 style={{fontFamily:'Manrope'}}>RBC morphology</h3>{Object.entries(analysis.morphology).map(([name,v])=><p key={name}>{name}: {v.count} ({v.percentage.toFixed(1)}%)</p>)}<p style={{marginTop:16}}>Average RBC morphology confidence: {(analysis.rbc_morphology_average_confidence*100).toFixed(1)}%</p></div></> : <div className="empty"><span><FileText size={23}/></span><h2>No analysis report yet</h2><p>Run an image analysis first.</p></div>}</div></AppShell>
}
