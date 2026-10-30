'use client';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import AppShell from '../../components/AppShell';
import { ArrowRight, FileText, Image as ImageIcon, ShieldCheck, UploadCloud, Zap } from 'lucide-react';
import { analyzeImage } from '../../lib/api';

export default function Analysis() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  async function startAnalysis() {
    if (!file) {
      setError('Please select a blood-smear image first.');
      return;
    }
    setBusy(true);
    setError('');
    try {
      const result = await analyzeImage(file);
      localStorage.setItem('hemovision-analysis', JSON.stringify(result));
      router.push('/dashboard');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed.');
    } finally {
      setBusy(false);
    }
  }

  return <AppShell>
    <div className="page">
      <div className="heading"><div><small>WORKSPACE / NEW ANALYSIS</small><h1>New Analysis</h1><p>Upload a microscopic blood-smear image to begin AI-assisted analysis.</p></div></div>
      <div className="analysis-grid">
        <div>
          <label className="upload">
            <input type="file" accept="image/png,image/jpeg,image/tiff,image/tif" onChange={e => { setFile(e.target.files?.[0] || null); setError(''); }} />
            <span><UploadCloud size={28}/></span>
            <h2>{file?.name || 'Drop your smear image here'}</h2>
            <p>{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB • Ready for analysis` : 'or click to browse from your computer'}</p>
            <small>JPG • PNG • TIFF • Maximum 20 MB</small>
          </label>
          {error && <div className="security" style={{color:'#b42318', background:'#fff5f4'}}>{error}</div>}
        </div>
        <div>
          <section className="panel options">
            <div className="panel-head"><div><h3>Analysis pipeline</h3><p>Stages connected to the validated models.</p></div></div>
            {[
              ['Cell detection','YOLO11n • RBC, WBC, Platelet'],
              ['RBC morphology','Stage 2 • 11 morphology classes'],
              ['Quantitative analysis','Counts, percentages & confidence'],
              ['Annotated result','Detection + morphology visualization']
            ].map(x=><div className="toggle-row" key={x[0]}><span><b>{x[0]}</b><small>{x[1]}</small></span><i className="toggle on"/></div>)}
            <button className="btn primary full" onClick={startAnalysis} disabled={busy}>
              {busy ? 'Running AI analysis…' : 'Start Analysis'} <ArrowRight size={16}/>
            </button>
          </section>
          <div className="security"><ShieldCheck size={17}/><div><b>Research workspace</b><p>AI output is for research/education and visual verification, not definitive diagnosis.</p></div></div>
        </div>
      </div>
      <div className="features"><Feature icon={<Zap/>} title="Real inference" text="Final YOLO11n pipeline"/><Feature icon={<ImageIcon/>} title="Visual verification" text="Inspect annotated detections"/><Feature icon={<FileText/>} title="Structured results" text="Counts, morphology & confidence"/></div>
    </div>
  </AppShell>
}

function Feature({icon,title,text}:{icon:React.ReactNode;title:string;text:string}){return <div className="feature"><span>{icon}</span><div><b>{title}</b><p>{text}</p></div></div>}
